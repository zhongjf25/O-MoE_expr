import torch
import threading
from vllm.logger import init_logger
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.utils.platform_utils import is_pin_memory_available
import weakref
import json
import time
import os
from vllm.v1.worker.block_table import ExpertBlockTable

logger = init_logger(__name__)


class StreamContext:

    memory_stream: torch.cuda.Stream = None
    compute_stream: torch.cuda.Stream = None
    prefetch_stream: torch.cuda.Stream = None
    initialized = False

    @classmethod
    def init(cls):
        if not cls.initialized:
            cls.memory_stream = torch.cuda.Stream()
            cls.prefetch_stream = torch.cuda.Stream()
            cls.compute_stream = torch.cuda.current_stream()
            cls.initialized = True


class BackendExpertManager:

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BackendExpertManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        StreamContext.init()
        self.vllm_config = get_current_vllm_config()
        self.hf_config = self.vllm_config.model_config.hf_text_config
        self.tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        self.use_tp = not self.vllm_config.parallel_config.enable_expert_parallel and self.tp_size > 1

        self.tp_rank = get_tensor_model_parallel_rank()
        
        self.num_experts = self.hf_config.num_experts if hasattr(self.hf_config, 'num_experts') \
            else self.hf_config.n_routed_experts
        self.first_k_dense_replace = self.hf_config.first_k_dense_replace if \
            hasattr(self.hf_config, 'first_k_dense_replace') else 0
        self.num_hidden_layers = self.hf_config.num_hidden_layers

        self.moe_modules = {}
        self.gate_modules = {}
        
        self.w13_weight_1 = None
        self.w2_weight_1 = None
        self.w13_weight_2 = None
        self.w2_weight_2 = None
        self.expert_params = {}
        self.comp_flag = 1

        self.w13_blocks = None
        self.w2_blocks = None

        self.block_table: ExpertBlockTable = None

        self.layer_prefixes = []

        self.prefetch_daemon = PrefetchDaemon(self)
        self.pin_memory = is_pin_memory_available()
        # 当前 step 内按 layer 聚合的 topk 专家，用于通过 RPC 回传 CPU 前台。
        self.pending_topk_updates = {}
        self.pending_topk_lock = threading.Lock()

        # Per-layer targets from dynamic cache delta (set by apply_cache_delta).
        self._evict_targets: dict[int, set[int]] = {}
        self._load_targets: dict[int, set[int]] = {}
    
    def _flatten_topk_ids(self, topk_ids) -> list[int]:
        if topk_ids is None:
            return []
        if isinstance(topk_ids, torch.Tensor):
            return topk_ids.reshape(-1).tolist()
        if isinstance(topk_ids, list):
            flat_ids = []
            for item in topk_ids:
                if isinstance(item, list):
                    flat_ids.extend(item)
                else:
                    flat_ids.append(item)
            return flat_ids
        return list(topk_ids)

    def _record_topk_ids_for_rpc(self, layer_id: int, topk_ids) -> None:
        expert_ids = self._flatten_topk_ids(topk_ids)
        if not expert_ids:
            return
        with self.pending_topk_lock:
            if layer_id not in self.pending_topk_updates:
                self.pending_topk_updates[layer_id] = []
            self.pending_topk_updates[layer_id].extend(expert_ids)

    def apply_cache_delta(self, delta) -> None:
        """Receive an ExpertCacheDelta from the frontend and decompose it
        into per-layer load / evict targets that will be consumed during
        the forward pass by ``get_experts_with_topk_ids``."""
        self._evict_targets.clear()
        self._load_targets.clear()

        for layer_id, expert_id in delta.experts_to_evict:
            self._evict_targets.setdefault(layer_id, set()).add(expert_id)

        for layer_id, expert_id in delta.experts_to_load:
            self._load_targets.setdefault(layer_id, set()).add(expert_id)

        if delta.new_expert_to_block:
            if not hasattr(self, 'expert_to_block'):
                self.expert_to_block = {}
            self.expert_to_block.update(delta.new_expert_to_block)
            self.block_table.update_and_commit_experts(self.expert_to_block)

    def drain_topk_updates(self) -> dict[int, list[int]]:
        """取出并清空本 step 累积的 topk 更新（用于 worker->engine RPC）。"""
        with self.pending_topk_lock:
            if not self.pending_topk_updates:
                return {}
            updates = dict(self.pending_topk_updates)
            self.pending_topk_updates.clear()
        return updates
    
    def get_layer_prefix(self, layer_id: int):
        return self.layer_prefixes[layer_id - self.first_k_dense_replace]
    
    def get_layer_index(self, layer_prefix: str):
        return self.layer_prefixes.index(layer_prefix) + self.first_k_dense_replace
    
    def initialize_experts(self, expert_to_block: dict[(int, int, str), int]):
        self.expert_to_block = expert_to_block
        if self.w13_blocks is None or self.w2_blocks is None:
            assert False, "Failed to initialize experts, w13_blocks or w2_blocks is None"
        
        for (layer_id, expert_id, w123) in expert_to_block:
            block_id = expert_to_block[(layer_id, expert_id, w123)]
            layer_prefix = self.get_layer_prefix(layer_id)
            layer_experts = self.expert_params.get(layer_prefix, {})
            param = layer_experts[expert_id][w123]
            if w123 == "w1" or w123 == "w3":
                blocks = self.w13_blocks
            elif w123 == "w2":
                blocks = self.w2_blocks
            else:
                assert False, "Invalid w123"

            assert blocks[block_id].shape == param.shape, "Shape mismatch"
            blocks[block_id].copy_(param.data, non_blocking=True)

    def init_w13_w2_weight(self, w13_weight, w2_weight):
        device = torch.cuda.current_device()
        if self.w13_weight_1 is None and self.w13_weight_2 is None:
            self.w13_weight_1 = torch.empty_like(w13_weight, device=device)
            self.w2_weight_1 = torch.empty_like(w2_weight, device=device)
            self.w13_weight_2 = torch.empty_like(w13_weight, device=device)
            self.w2_weight_2 = torch.empty_like(w2_weight, device=device)
            print(f"[debug] init w13_weight and w2_weight on {device=}, {self.w13_weight_1.shape=}, {self.w2_weight_1.shape=}")

    # def get_gate_layer_prefix(self, layer_prefix):
    #     parts = layer_prefix.split('.')
                  
    #     layer_idx = int(parts[2]) + 1
    #     parts[2] = str(layer_idx)
    #     if parts[4] == 'experts':
    #         parts[4] = 'gate'
    #     else:
    #         raise ValueError("Suffix is not 'experts' but expected to be")
        
    #     new_layer_prefix = '.'.join(parts)
    #     return new_layer_prefix
       
    def get_gate_layer_prefix(self, layer_prefix):
        from vllm.model_executor.models.utils import extract_layer_index

        layer_idx = extract_layer_index(layer_prefix)
        next_layer_idx = layer_idx + 1
        parts = layer_prefix.split(".")

        replaced_layer_idx = False
        for i, part in enumerate(parts):
            if part.isdigit() and int(part) == layer_idx:
                parts[i] = str(next_layer_idx)
                replaced_layer_idx = True
                break

        if not replaced_layer_idx:
            raise ValueError(
                f"Failed to find layer index {layer_idx} in layer_prefix: {layer_prefix}"
            )

        if "experts" not in parts:
            raise ValueError(
                f"Suffix does not contain 'experts' but expected to be in: {layer_prefix}"
            )

        i = parts.index("experts") 
        parts[i] = "gate"
        new_layer_prefix = ".".join(parts)
        return new_layer_prefix
    
    def get_pred_router_logits(self, hidden_states, layer_prefix):
        
        layer_prefix = self.get_gate_layer_prefix(layer_prefix)
        
        if layer_prefix is None:
            return None
        
        module_ref = self.gate_modules.get(layer_prefix)            
        if module_ref is None:
            return None
        
        next_gate = module_ref()
        if next_gate is None:
            return None
        
        pred_router_logits, _ = next_gate(hidden_states)
        
        return pred_router_logits

    # 通过路由信息获取权重并进行预取
    # 参数(layer_prefix: 层ID, topk_ids: 当前层专家ID列表, topk_ids_pred: 下一层预测专家ID列表)
    def get_experts_with_topk_ids(self, layer_prefix, topk_ids, topk_ids_pred):
        module_ref = self.moe_modules.get(layer_prefix)
        if module_ref is None:
            logger.error(f"module with prefix {layer_prefix} not found")
            return None, None

        module = module_ref()
        if module is None:
            logger.error(f"module with prefix {layer_prefix} has been gc")
            return None, None
            
        # 仅做后台聚合，后续由 ModelRunnerOutput 通过 RPC 传回 CPU 前台更新。
        layer_idx = self.get_layer_index(layer_prefix)
        self._record_topk_ids_for_rpc(layer_idx, topk_ids)

        # 专家ID去重，得到需要加载的全局专家ID列表
        unique_ids = torch.unique(topk_ids).tolist()
        
        if self.w13_weight_1 is None or self.w2_weight_1 is None or self.w13_weight_2 is None or self.w2_weight_2 is None:
            logger.error(f"w13_weight or w2_weight not initialized for {layer_prefix}")
            return None, None

        if layer_prefix not in self.expert_params:
            logger.error(f"No expert params found for {layer_prefix}")
            return None, None
        
        # 获取当前层专家权重字典
        # key: 本地专家ID, value: 该专家权重字典
        # 字典内容:  {"w13": w13_param, "w2": w2_param}
        layer_experts = self.expert_params[layer_prefix]

        stream = StreamContext.memory_stream

        prefetched_experts = self.prefetch_daemon.notify_layer_loading(layer_idx)
        if prefetched_experts is None:
            prefetched_experts = []
        # print(layer_prefix, "prefetched_experts: ", prefetched_experts)

        w13_weight_comp = self.w13_weight_1 if self.comp_flag == 1 else self.w13_weight_2
        w2_weight_comp = self.w2_weight_1 if self.comp_flag == 1 else self.w2_weight_2
        self.comp_flag = 1 if self.comp_flag == 2 else 2
        on_demand_loaded_count = 0

        with torch.cuda.stream(stream):
            # 遍历所有本次需要的专家ID
            for expert_id in unique_ids:
                # # 已放入计算矩阵，判定是否需要del or cache
                if expert_id in prefetched_experts:
                #     local_cached_experts = self.local_cached_experts_params[layer_prefix]
                #     # 缓存专家是否需要驱逐
                #     if expert_id_int in local_cached_experts:
                #         if not self.global_cached_experts_id[layer_prefix].exist(expert_id_int):
                #             del local_cached_experts[expert_id_int]
                #     # 预取专家是否需要加入cache
                #     else:
                #         if self.global_cached_experts_id[layer_prefix].exist(expert_id_int):
                #             # .clone(non_blocking=True).detach()
                #             device = torch.cuda.current_device()
                #             w13_param = torch.empty_like(w13_weight_comp[expert_id_int], device=device)
                #             w2_param = torch.empty_like(w2_weight_comp[expert_id_int], device=device)
                #             w13_param.copy_(w13_weight_comp[expert_id_int].data, non_blocking=True)
                #             w2_param.copy_(w2_weight_comp[expert_id_int].data, non_blocking=True)
                #             local_cached_experts[expert_id_int] = w13_param, w2_param
                    continue
                
                # 缓存命中
                w1_block_id, w2_block_id, w3_block_id = self.block_table.get_device_block_id(layer_idx, expert_id)
                blocks_ready = self.w13_blocks is not None and self.w2_blocks is not None
                if w1_block_id != -1 and blocks_ready:
                    w1_param = self.w13_blocks[w1_block_id]
                    w2_param = self.w2_blocks[w2_block_id]
                    w3_param = self.w13_blocks[w3_block_id]
                
                # 否则从CPU按需加载专家权重
                else:
                    on_demand_loaded_count += 1
                    w1_param = self.load_param(layer_experts[expert_id]["w1"])
                    w2_param = self.load_param(layer_experts[expert_id]["w2"])
                    w3_param = self.load_param(layer_experts[expert_id]["w3"])

                # 将加载的权重复制到GPU张量中
                [intermediate_size, _] = w1_param.shape
                w13_weight_comp[expert_id][:intermediate_size].copy_(w1_param.data, non_blocking=True)
                w13_weight_comp[expert_id][intermediate_size:].copy_(w3_param.data, non_blocking=True)
                w2_weight_comp[expert_id].copy_(w2_param.data, non_blocking=True)
        
        stream.synchronize()

        # 处理下一层需要预取的专家ID
        if topk_ids_pred == []:
            prefetch_expert_ids = topk_ids_pred
        else:
            topk_ids_pred = torch.unique(topk_ids_pred)
            prefetch_expert_ids = set(topk_ids_pred.tolist())

        try:
            # current_idx = self.layer_prefixes.index(layer_prefix)
            if layer_idx >= self.num_hidden_layers:
                next_layer_idx = self.first_k_dense_replace
            else:
                next_layer_idx = layer_idx + 1
            # Inject dynamic-cache expansion targets into the prefetch list
            # so that PrefetchDaemon loads them asynchronously alongside
            # the normal next-layer experts.
            next_load = self._load_targets.pop(next_layer_idx, None)
            if next_load:
                existing = set(prefetch_expert_ids)
                for eid in next_load:
                    if eid not in existing:
                        prefetch_expert_ids.add(eid)

            self.prefetch_daemon.schedule_prefetch(next_layer_idx, prefetch_expert_ids)
        except ValueError:
            print(f"[warning] can not find layer_prefix: {layer_prefix}")

        # Dynamic-cache shrinkage: invalidate evicted experts in the
        # block table after this layer's computation is done (cheap
        # metadata-only operation).
        evict_set = self._evict_targets.pop(layer_idx, None)
        if evict_set:
            for expert_id in evict_set:
                self.block_table.block_table.np[layer_idx, expert_id, :] = -1
            self.block_table.block_table.copy_to_gpu()
        
        return w13_weight_comp, w2_weight_comp
    
    def register_moe_module(self, layer_prefix, module):
        if layer_prefix in self.moe_modules:
            logger.error(f"Module with layer_prefix {layer_prefix} already registered.")
            return
        self.moe_modules[layer_prefix] = weakref.ref(module)
        self.layer_prefixes.append(layer_prefix)
        # self.initialize_cached_experts(layer_prefix)
    
    def register_gate_module(self, layer_prefix, module):
        if layer_prefix in self.gate_modules:
            logger.error(f"Module with layer_prefix {layer_prefix} already registered.")
            return
        self.gate_modules[layer_prefix] = weakref.ref(module)
    
    def get_all_moe_modules(self):
        return {prefix: ref() for prefix, ref in self.moe_modules.items() if ref() is not None}
    
    def split_w13_w2_weight(self, layer_prefix):
        """
        将 MoE 层的 w13_weight 和 w2_weight 拆分并卸载到 CPU
        初始化的静态预加载
        
        Args:
            layer_prefix: 层标识符，如 "model.layers.0.mlp.experts"
        
        调用时机:
            - 在 process_weights_after_loading() 中被调用
            - 权重加载完成后，模型初始化的最后阶段
        """ 
        
        module_ref = self.moe_modules.get(layer_prefix)        
        module = module_ref()
        if module is None:
            logger.error(f"Module with prefix {layer_prefix} has been gc")
            return
        
        if not hasattr(module, "w13_weight") or not hasattr(module, "w2_weight"):
            logger.error(f"Module {layer_prefix} does not have w13_weight and w2_weight")
            return
        
        if module.w13_weight.device.type == "cpu" and module.w2_weight.device.type == "cpu":
            num_experts = module.w13_weight.size(0)

            if layer_prefix not in self.expert_params:
                self.expert_params[layer_prefix] = {}

            for i in range(num_experts):
                w1, w3 = module.w13_weight[i].chunk(2, dim=0)
                w2 = module.w2_weight[i]
                if self.pin_memory:
                    w1 = w1.pin_memory()
                    w2 = w2.pin_memory()
                    w3 = w3.pin_memory()
                
                self.expert_params[layer_prefix][i] = {
                    "w1": torch.nn.Parameter(w1, requires_grad=False),
                    "w2": torch.nn.Parameter(w2, requires_grad=False),
                    "w3": torch.nn.Parameter(w3, requires_grad=False)
                }
            
            self.init_w13_w2_weight(w13_weight=module.w13_weight, w2_weight=module.w2_weight)

            # 从GPU卸载权重
            module.w13_weight = None
            module.w2_weight = None

        else:
            print(f"[debug] weights for {layer_prefix} are not on CPU!!")

        
    def load_param(self, param):
        if param is None:
            return None
        
        device_type = param.device.type
        if device_type == "cuda":
            return param
        
        param_device = param.cuda(non_blocking=True)

        return param_device
    
    def __del__(self):
        if hasattr(self, 'prefetch_daemon'):
            self.prefetch_daemon.shutdown()


class PrefetchDaemon:
    """运行时预取专家权重"""

    def __init__(self, manager: BackendExpertManager):
        self.manager = manager
        self.prefetch_queue = []
        self.loaded_queue = []
        self.next_layer_idx = None
        self.shutdown_flag = threading.Event()
        self.lock = threading.Lock()

        self.daemon_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.daemon_thread.start()

    
    def schedule_prefetch(self, layer_idx: int, prefetch_expert_ids_pred):
        with self.lock:
            self.prefetch_queue = []
            self.loaded_queue = []
            self.next_layer_idx = layer_idx

            self.prefetch_queue = list(prefetch_expert_ids_pred)

    def notify_layer_loading(self, layer_idx: int):
        with self.lock:
            if layer_idx == self.next_layer_idx:
                # rest_len = len(self.prefetch_queue)
                self.prefetch_queue = []
                # print(f"clear list, the number of the rest experts is : {rest_len}")
                return self.loaded_queue
    
    def shutdown(self):
        self.shutdown_flag.set()
        if self.daemon_thread.is_alive():
            self.daemon_thread.join(timeout=1)
    
    def _prefetch_worker(self):
        with torch.cuda.stream(StreamContext.prefetch_stream):
            while not self.shutdown_flag.is_set():
                expert_id = None
                layer_idx = None

                with self.lock:
                    if self.prefetch_queue and self.next_layer_idx:
                        expert_id = self.prefetch_queue.pop(0)
                        layer_idx = self.next_layer_idx
                
                if expert_id is not None and layer_idx is not None:
                    success = self._load_expert(layer_idx, expert_id)
                    if not success:
                        continue
                else:
                    time.sleep(0.0005)

    def _load_expert(self, layer_idx: int, expert_id: int):
        try:
            with self.lock:
                if layer_idx != self.next_layer_idx:
                    return False
                if not self.prefetch_queue:
                    return False
            
            layer_prefix = self.manager.get_layer_prefix(layer_idx)
            layer_experts = self.manager.expert_params.get(layer_prefix, {})
            
            w13_weight_comm = self.manager.w13_weight_1 if self.manager.comp_flag == 1 else self.manager.w13_weight_2
            w2_weight_comm = self.manager.w2_weight_1 if self.manager.comp_flag == 1 else self.manager.w2_weight_2
            
            w1_block_id, w2_block_id, w3_block_id = self.manager.block_table.get_device_block_id(layer_idx, expert_id)
            blocks_ready = (self.manager.w13_blocks is not None
                            and self.manager.w2_blocks is not None)
            if w1_block_id != -1 and blocks_ready:
                w1_param = self.manager.w13_blocks[w1_block_id]
                w2_param = self.manager.w2_blocks[w2_block_id]
                w3_param = self.manager.w13_blocks[w3_block_id]
            else:
                w1_param = layer_experts[expert_id]["w1"]
                w2_param = layer_experts[expert_id]["w2"]
                w3_param = layer_experts[expert_id]["w3"]
            
            [intermediate_size, _] = w1_param.shape
            w13_weight_comm[expert_id][:intermediate_size].copy_(w1_param.data, non_blocking=True)
            w13_weight_comm[expert_id][intermediate_size:].copy_(w3_param.data, non_blocking=True)
            w2_weight_comm[expert_id].copy_(w2_param.data, non_blocking=True)

            StreamContext.prefetch_stream.synchronize()
            self.loaded_queue.append(expert_id)
            return True
        
        except Exception as e:
            print(f"[error] fail to prefetch expert {expert_id} from layer {layer_idx}: {e}")
            return False