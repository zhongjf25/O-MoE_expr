import queue
import torch
import threading
from vllm.logger import init_logger
from typing import (Dict)
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.model_executor.expert_ARC_cache import ARC_Cache
from vllm.config import get_current_vllm_config
from contextlib import contextmanager
import psutil
import weakref
import json
import time
import os
import gc
import re

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
            cls.compute_stream = torch.cuda.Stream()
            cls.prefetch_stream = torch.cuda.current_stream()
            cls.initialized = True

class PrefetchDaemon:

    def __init__(self, manager):
        self.manager = manager
        self.prefetch_queue = []
        self.loaded_queue = []
        self.next_layer_prefix = None
        self.shutdown_flag = threading.Event()
        self.lock = threading.Lock()

        self.daemon_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.daemon_thread.start()

    
    def shcedule_prefetch(self, layer_prefix : str, prefetch_expert_ids_pred):
        with self.lock:
            self.manager.offload_prefetched_experts()
            self.prefetch_queue = []
            self.loaded_queue = []
            self.next_layer_prefix = layer_prefix

            count = 0
            for expert_id in list(prefetch_expert_ids_pred):
                if not self.manager.global_cached_experts_id[layer_prefix].exist(expert_id):
                    self.prefetch_queue.append(expert_id)
                    count += 1

    def notify_layer_loading(self, layer_prefix : str):
        with self.lock:
            if layer_prefix == self.next_layer_prefix:
                rest_len = len(self.prefetch_queue)
                self.prefetch_queue = []
                # print(f"clear list, the number of the rest experts is : {rest_len}")
                loaded_experts = self.loaded_queue
                self.loaded_queue = []
                return loaded_experts
    
    def shutdown(self):
        self.shutdown_flag.set()
        if self.daemon_thread.is_alive():
            self.daemon_thread.join(timeout=1)
    
    def _prefetch_worker(self):
        with torch.cuda.stream(StreamContext.prefetch_stream):
            while not self.shutdown_flag.is_set():
                expert_id = None
                layer_prefix = None

                with self.lock:
                    if self.prefetch_queue and self.next_layer_prefix:
                        expert_id = self.prefetch_queue.pop(0)
                        layer_prefix = self.next_layer_prefix
                
                if expert_id is not None and layer_prefix is not None:
                    success = self._load_expert(layer_prefix, expert_id)
                    if not success:
                        continue
                else:
                    time.sleep(0.0005)

    def _load_expert(self, layer_prefix: str, expert_id: int):
        try:
            with self.lock:
                if layer_prefix != self.next_layer_prefix:
                    return False
                if not self.prefetch_queue:
                    return False
                
            layer_experts = self.manager.expert_params.get(layer_prefix, {})
            local_expert_id = self.manager.get_local_expert_id(expert_id)

            if local_expert_id not in layer_experts:
                return False
            
            w13_param = layer_experts[local_expert_id]["w13"]
            w2_param = layer_experts[local_expert_id]["w2"]

            loaded_w13 = self.manager.load_param(w13_param)
            loaded_w2 = self.manager.load_param(w2_param)
            StreamContext.prefetch_stream.synchronize()

            self.manager.prefetched_params[expert_id] = (loaded_w13, loaded_w2)
            self.loaded_queue.append(expert_id)

            return True
        
        except Exception as e:
            print(f"[error] fail to prefetch expert {expert_id} from layer {layer_prefix}: {e}")
            return False
        

class DeepSeekModuleManager:

    _instance = None

    @staticmethod
    def _resolve_num_experts_from_hf_config(hf_config) -> int:
        """Support DeepSeek/Qwen2 MoE (top-level) and Qwen3.5 MoE (nested ``text_config``)."""
        if getattr(hf_config, "num_experts", None) is not None:
            return hf_config.num_experts
        if getattr(hf_config, "n_routed_experts", None) is not None:
            return hf_config.n_routed_experts
        text = getattr(hf_config, "text_config", None)
        if text is not None:
            if getattr(text, "num_experts", None) is not None:
                return text.num_experts
            if getattr(text, "n_routed_experts", None) is not None:
                return text.n_routed_experts
        raise ValueError(
            "Cannot determine num_experts from hf_config (tried num_experts / "
            "n_routed_experts on root and text_config)."
        )

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeepSeekModuleManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        StreamContext.init()
        self.vllm_config = get_current_vllm_config()
        self.hf_config = self.vllm_config.model_config.hf_config
        self.tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        from vllm.distributed import get_tensor_model_parallel_rank
        self.tp_rank = get_tensor_model_parallel_rank()
        self.num_experts = self._resolve_num_experts_from_hf_config(self.hf_config)

        self.moe_modules = {}
        self.gate_modules = {}
        
        self.w13_weight = None
        self.w2_weight = None
        self.expert_params = {}

        self.is_expert_offload_enabled = os.getenv("DS_EXPERT_OFFLOAD", "1") == "1"

        self.expert_map = None
        # MoE: TP shards FFN per rank (full local expert table); EP partitions experts.
        # Matches parallel_config: enable_expert_parallel uses EP for MoE, else TP.
        self.use_tp = not self.vllm_config.parallel_config.enable_expert_parallel
        self.intermediate_size_per_partition = None

        if self.use_tp:
            self.expert_per_device = self.num_experts
            intermediate_size = getattr(
                self.hf_config, "moe_intermediate_size",
                getattr(self.hf_config, "intermediate_size", None),
            )
            if intermediate_size is not None:
                assert intermediate_size % self.tp_size == 0, (
                    f"intermediate_size ({intermediate_size}) must be divisible by "
                    f"tp_size ({self.tp_size})"
                )
                self.intermediate_size_per_partition = intermediate_size // self.tp_size
                logger.info(
                    "[expert offload] MoE tensor parallel (enable_expert_parallel=False): "
                    "rank %s/%s, %s experts, FFN shard dim %s per rank",
                    self.tp_rank,
                    self.tp_size,
                    self.num_experts,
                    self.intermediate_size_per_partition,
                )
            else:
                logger.warning(
                    "expert offload (MoE TP): intermediate_size not found in hf_config; "
                    "skipping TP divisibility check",
                )
        else:
            self.expert_per_device = self.num_experts // self.tp_size

        self.cached_experts_count = int(os.getenv("DS_CACHED_EXPERTS_COUNT", "0"))
        self.global_cached_experts_id = {}
        self.local_cached_experts_params = {}

        self.layer_prefixes = []
        self.prefetched_params = {}

        self.prefetch_daemon = PrefetchDaemon(self)

    def register_expert_map(self, expert_map):
        if expert_map is None:
            # logger.warning("Expert map is None, skipping registration expert_map for EP.")
            return
        self.expert_map = expert_map.cpu().tolist()

    def get_local_expert_id(self, global_expert_id: int):
        """
        global_expert_id -> loacl_expert_id
        """
        if self.use_tp:
            return global_expert_id
        if self.expert_map is None:
            return global_expert_id
        return self.expert_map[global_expert_id]
    
    def initialize_cached_experts(self, layer_prefix):
        cache_expert_per_device = self.cached_experts_count // self.tp_size

        if self.use_tp:
            if layer_prefix == self.layer_prefixes[0]:
                init_list = list(range(self.num_experts))
            elif cache_expert_per_device > 0:
                init_list = list(range(cache_expert_per_device))
            else:
                init_list = []
        else:
            device = int(torch.cuda.current_device())
            start_expert_id = device * self.expert_per_device

            if layer_prefix == self.layer_prefixes[0]:
                init_list = [index for index in range(start_expert_id, start_expert_id+self.expert_per_device)]
            elif cache_expert_per_device > 0:
                init_list = [index for index in range(start_expert_id, start_expert_id+cache_expert_per_device)]
            else:
                init_list = []
        self.global_cached_experts_id[layer_prefix] = ARC_Cache(init_list)
        self.local_cached_experts_params[layer_prefix] = {}
        
    def init_w13_w2_weight(self, w13_weight, w2_weight):
        device = int(torch.cuda.current_device())
        if self.w13_weight is None and self.w2_weight is None:
            self.w13_weight = torch.empty_like(w13_weight, device=device)
            self.w2_weight = torch.empty_like(w2_weight, device=device)
            print(f"[debug] init w13_weight and w2_weight on {device=}, {self.w13_weight.shape=}, {self.w2_weight.shape=}")
                  
    def get_gate_layer_prefix(self, layer_prefix):
        """Map experts block prefix to the *next* layer's gate prefix for prefetch.

        Supports:
        - ``...layers.<idx>.mlp.experts`` (e.g. Qwen3.5: ``language_model.model.layers.0...``)
        - Legacy ``...<idx>....experts`` with layer index at ``parts[2]`` (no ``layers`` segment).
        """
        parts = layer_prefix.split(".")
        if "layers" in parts:
            i = parts.index("layers")
            if i + 1 >= len(parts):
                raise ValueError(
                    f"Invalid layer prefix (no index after 'layers'): {layer_prefix!r}"
                )
            parts[i + 1] = str(int(parts[i + 1]) + 1)
        else:
            if len(parts) < 5:
                raise ValueError(f"Unexpected layer prefix: {layer_prefix!r}")
            parts[2] = str(int(parts[2]) + 1)
        if "experts" in parts:
            parts[parts.index("experts")] = "gate"
        else:
            raise ValueError(
                "Expected 'experts' in layer prefix to map to gate: " + repr(layer_prefix)
            )
        return ".".join(parts)
    
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
    
    def update_expert_param_cache(self, layer_prefix):
        if self.use_tp:
            global_experts_current_rank = list(range(self.num_experts))
        else:
            device = int(torch.cuda.current_device())
            start_expert_id = device * self.expert_per_device
            global_experts_current_rank = [
                index for index in range(start_expert_id, start_expert_id + self.expert_per_device)
            ]

        dirty_list = list(self.local_cached_experts_params[layer_prefix])
        for global_experts_id_int in global_experts_current_rank:
            local_expert_id_int = self.get_local_expert_id(global_experts_id_int)
            
            if not self.global_cached_experts_id[layer_prefix].exist(global_experts_id_int) and local_expert_id_int in dirty_list:
                  del self.local_cached_experts_params[layer_prefix][local_expert_id_int]

    def get_experts_with_topk_ids(self, layer_prefix, topk_ids, topk_ids_pred):
        module_ref = self.moe_modules.get(layer_prefix)
        if module_ref is None:
            logger.error(f"module with prefix {layer_prefix} not found")
            return None, None

        module = module_ref()
        if module is None:
            logger.error(f"module with prefix {layer_prefix} has been gc")
            return None, None

        global_unique_ids = torch.unique(topk_ids).tolist()

        if len(self.global_cached_experts_id[layer_prefix].get()) > 0:
            for global_expert_id_int in global_unique_ids:
                self.global_cached_experts_id[layer_prefix].update(global_expert_id_int)
        
        if self.w13_weight is None or self.w2_weight is None:
            logger.error(f"w13_weight or w2_weight not initialized for {layer_prefix}")
            return None, None

        if layer_prefix not in self.expert_params:
            logger.error(f"No expert params found for {layer_prefix}")
            return None, None
        
        layer_experts = self.expert_params[layer_prefix]

        stream = StreamContext.memory_stream

        self.prefetch_daemon.notify_layer_loading(layer_prefix)
        prefetched_experts = list(self.prefetched_params.keys())

        on_demand_loaded_count = 0

        with torch.cuda.stream(stream):
            for global_expert_id_int in global_unique_ids:
                local_expert_id_int = self.get_local_expert_id(global_expert_id_int)

                if local_expert_id_int not in layer_experts:
                    continue

                if local_expert_id_int in self.local_cached_experts_params[layer_prefix]:
                    w13_param, w2_param = self.local_cached_experts_params[layer_prefix][local_expert_id_int]

                    if not self.global_cached_experts_id[layer_prefix].exist(global_expert_id_int):
                        del self.local_cached_experts_params[layer_prefix][local_expert_id_int]
                elif global_expert_id_int in prefetched_experts:
                    w13_param, w2_param = self.prefetched_params[global_expert_id_int]

                    del self.prefetched_params[global_expert_id_int]
                    if self.global_cached_experts_id[layer_prefix].exist(global_expert_id_int):
                        self.local_cached_experts_params[layer_prefix][local_expert_id_int] = w13_param, w2_param
                else:
                    if layer_experts[local_expert_id_int]["w13"].device.type != 'cuda':
                        on_demand_loaded_count += 1
                    
                    w13_param = self.load_param(layer_experts[local_expert_id_int]["w13"])
                    w2_param = self.load_param(layer_experts[local_expert_id_int]["w2"])

                    if self.global_cached_experts_id[layer_prefix].exist(global_expert_id_int):
                        self.local_cached_experts_params[layer_prefix][local_expert_id_int] = (w13_param, w2_param)

                self.w13_weight[local_expert_id_int].copy_(w13_param.data, non_blocking=True)
                self.w2_weight[local_expert_id_int].copy_(w2_param.data, non_blocking=True)
        
        self.update_expert_param_cache(layer_prefix)
        stream.synchronize()

        if topk_ids_pred == []:
            prefetch_global_id_current_rank = topk_ids_pred
        else:
            topk_ids_pred = torch.unique(topk_ids_pred)
            prefetch_global_expert_ids = set(topk_ids_pred.tolist())
            prefetch_global_id_current_rank = [id for id in prefetch_global_expert_ids if self.get_local_expert_id(id) != -1]

        try:
            current_idx = self.layer_prefixes.index(layer_prefix)
            if current_idx >= len(self.layer_prefixes) - 1:
                next_layer_idx = 0
            else:
                next_layer_idx = current_idx + 1

            next_layer_prefix = self.layer_prefixes[next_layer_idx]
            self.prefetch_daemon.shcedule_prefetch(next_layer_prefix, prefetch_global_id_current_rank)
        except ValueError:
            print(f"[warning] can not find layer_prefix: {layer_prefix}")
        
        return self.w13_weight, self.w2_weight
    
    def offload_prefetched_experts(self):
        if self.prefetched_params:
            self.prefetched_params.clear()
    
    def register_moe_module(self, layer_prefix, module):
        if layer_prefix in self.moe_modules:
            logger.error(f"Module with layer_prefix {layer_prefix} already registered.")
            return
        self.moe_modules[layer_prefix] = weakref.ref(module)
        self.layer_prefixes.append(layer_prefix)
        self.initialize_cached_experts(layer_prefix)
    
    def register_gate_module(self, layer_prefix, module):
        if layer_prefix in self.gate_modules:
            logger.error(f"Module with layer_prefix {layer_prefix} already registered.")
            return
        self.gate_modules[layer_prefix] = weakref.ref(module)
    
    def get_all_moe_modules(self):
        return {prefix: ref() for prefix, ref in self.moe_modules.items() if ref() is not None}
    
    def split_w13_w2_weight(self, layer_prefix):
        if not self.is_expert_offload_enabled:
            print(f"didn't enable expert offloading.")
            return
        
        module_ref = self.moe_modules.get(layer_prefix)
        if module_ref is None:
            logger.error(f"Module with prefix {layer_prefix} not found")
            return
        
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
                w13_pinned = module.w13_weight[i].pin_memory()
                w2_pinned = module.w2_weight[i].pin_memory()
                
                self.expert_params[layer_prefix][i] = {
                    "w13": torch.nn.Parameter(w13_pinned, requires_grad=False),
                    "w2": torch.nn.Parameter(w2_pinned, requires_grad=False)
                }

            cached_experts_list = self.global_cached_experts_id[layer_prefix].get()
            if len(cached_experts_list) > 0:
                global_id_current_rank = [id for id in cached_experts_list if self.get_local_expert_id(id) != -1]

                for expert_id in global_id_current_rank:
                    local_expert_id_int = self.get_local_expert_id(expert_id)

                    w13_param = self.load_param(self.expert_params[layer_prefix][local_expert_id_int]["w13"])
                    w2_param = self.load_param(self.expert_params[layer_prefix][local_expert_id_int]["w2"])

                    self.local_cached_experts_params[layer_prefix][local_expert_id_int] = (w13_param, w2_param)

            self.init_w13_w2_weight(w13_weight=module.w13_weight, w2_weight=module.w2_weight)

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