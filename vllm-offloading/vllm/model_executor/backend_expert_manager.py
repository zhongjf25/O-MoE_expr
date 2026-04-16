import torch
import threading
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import ClassVar
from vllm.logger import init_logger
from vllm.config import get_current_vllm_config
import weakref
import time
from vllm.v1.worker.block_table import ExpertBlockTable
from vllm.utils.expert_numa import maybe_bind_prefetch_thread_expert_numa
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

from vllm.model_executor.offloaded_expert_copy_extension import (
    ensure_offloaded_expert_copy_op_loaded,
)

ensure_offloaded_expert_copy_op_loaded()


logger = init_logger(__name__)


class StreamContext:

    memory_stream: torch.cuda.Stream = None
    prefetch_stream: torch.cuda.Stream = None
    initialized = False

    @classmethod
    def init(cls):
        if not cls.initialized:
            cls.memory_stream = torch.cuda.Stream()
            cls.prefetch_stream = torch.cuda.Stream()
            cls.initialized = True


@dataclass
class OffloadedExpertComputeInputs:
    SOURCE_INVALID: ClassVar[int] = -1
    SOURCE_COMP: ClassVar[int] = 0
    SOURCE_CACHE: ClassVar[int] = 1

    w13_weight_comp: torch.Tensor | None = None
    w2_weight_comp: torch.Tensor | None = None
    w13_blocks: torch.Tensor | None = None
    w2_blocks: torch.Tensor | None = None
    expert_source: torch.Tensor | None = None
    comp_expert_to_slot: torch.Tensor | None = None
    cache_w1_block_ids: torch.Tensor | None = None
    cache_w2_block_ids: torch.Tensor | None = None
    cache_w3_block_ids: torch.Tensor | None = None


@dataclass
class OffloadedLayerWeights:
    w13_cpu: torch.Tensor
    w2_cpu: torch.Tensor
    intermediate_size: int
    w13_uva: torch.Tensor | None = None
    w2_uva: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.w13_cpu.device.type != "cpu" or self.w2_cpu.device.type != "cpu":
            raise ValueError("OffloadedLayerWeights expects CPU tensors.")
        if self.w13_cpu.dim() != 3 or self.w2_cpu.dim() != 3:
            raise ValueError("OffloadedLayerWeights expects rank-3 tensors.")
        if self.w13_cpu.size(1) != 2 * self.intermediate_size:
            raise ValueError("w13_cpu shape does not match intermediate_size.")
        if self.w2_cpu.size(2) != self.intermediate_size:
            raise ValueError("w2_cpu shape does not match intermediate_size.")
        if not self.w13_cpu.is_contiguous():
            self.w13_cpu = self.w13_cpu.contiguous()
        if not self.w2_cpu.is_contiguous():
            self.w2_cpu = self.w2_cpu.contiguous()
        if not self.w13_cpu.is_pinned():
            self.w13_cpu = self.w13_cpu.pin_memory()
        if not self.w2_cpu.is_pinned():
            self.w2_cpu = self.w2_cpu.pin_memory()
        if self.w13_uva is None:
            self.w13_uva = get_accelerator_view_from_cpu_tensor(self.w13_cpu)
        if self.w2_uva is None:
            self.w2_uva = get_accelerator_view_from_cpu_tensor(self.w2_cpu)


@dataclass
class OffloadedExpertSelection:
    routed_expert_ids: torch.Tensor
    cached_expert_ids: torch.Tensor
    uncached_expert_ids: torch.Tensor
    cached_block_rows: torch.Tensor


@dataclass
class CompPrefetchRequest:
    context_id: int
    version: int
    layer_idx: int
    buffer_idx: int | None
    topk_ids_pred: object
    producer_event: torch.cuda.Event | None


@dataclass
class OffloadedPrepareState:
    seen_buffer: torch.Tensor
    miss_expert_ids: torch.Tensor
    miss_count: torch.Tensor
    miss_slot_ids: torch.Tensor | None = None
    pending_mask: torch.Tensor | None = None
    slot_live_flags: torch.Tensor | None = None
    epoch: int = 0


@dataclass
class CompBufferState:
    expert_to_slot: torch.Tensor
    slot_to_expert: torch.Tensor
    assigned_slot_count: int = 0
    layer_idx: int | None = None


@dataclass
class BlockLoadState:
    layer_idx: int | None = None
    buffer_idx: int | None = None
    target_ids: set[int] = field(default_factory=set)
    queue: list[int] = field(default_factory=list)
    inflight: set[int] = field(default_factory=set)
    ready_events: dict[int, torch.cuda.Event | None] = field(
        default_factory=dict)
    error: str | None = None


@dataclass
class CompPrefetchState:
    layer_idx: int | None = None
    buffer_idx: int | None = None
    version: int = 0
    pending_request: CompPrefetchRequest | None = None
    queue: list[tuple[int, int]] = field(default_factory=list)
    loaded_queue: list[tuple[int, torch.cuda.Event | None]] = field(
        default_factory=list)
    error: str | None = None


@dataclass
class PrefetchContext:
    context_id: int
    block_load: BlockLoadState = field(default_factory=BlockLoadState)
    comp_prefetch: CompPrefetchState = field(default_factory=CompPrefetchState)


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
        self.layer_prefix_to_idx: dict[str, int] = {}
        self.layer_idx_to_prefix: dict[int, str] = {}

        self.expert_offload_config = self.vllm_config.expert_offload_config
        self.dynamic_cache_enabled = bool(
            getattr(self.expert_offload_config, "dynamic_cache_enabled", False))
        self.use_legacy_copy_prefetch = not self.dynamic_cache_enabled
        self.prefetch_daemon = None
        self.legacy_prefetch_daemon = None
        if self.use_legacy_copy_prefetch:
            self.legacy_prefetch_daemon = LegacyPrefetchDaemon(self)
        else:
            self.prefetch_daemon = PrefetchDaemon(self)

        # Per-layer targets from dynamic cache delta (set by apply_cache_delta).
        self._evict_targets: dict[int, set[int]] = {}
        self._load_targets: dict[int, set[int]] = {}
        self._pending_reserved_blocks: dict[int, dict[int, tuple[int, int, int]]] = {}
        self._active_cache_delta_id: int | None = None
        self._active_prefetch_context_id: int | None = None
        self._evict_commit_mode: str = "row"
        self._selection_masks: dict[int, torch.Tensor] = {}
        self._offloaded_prepare_states: dict[int, OffloadedPrepareState] = {}
        self._comp_buffer_states: dict[int, CompBufferState] = {}

    def no_copy_compute_enabled(self) -> bool:
        return self.expert_offload_config.expert_no_copy_compute

    def compact_comp_buffer_enabled(self) -> bool:
        limit = max(0, getattr(self.expert_offload_config,
                               "offload_expert_limit", 0) or 0)
        return (
            self.no_copy_compute_enabled()
            and limit > 0
            and limit < self.num_experts
        )

    def comp_buffer_capacity(self) -> int:
        if self.compact_comp_buffer_enabled():
            return max(
                1,
                int(getattr(self.expert_offload_config,
                            "offload_expert_limit", 0)),
            )
        return self.num_experts

    def _get_comp_buffer_state(
        self,
        buffer_idx: int,
        device: torch.device,
    ) -> CompBufferState:
        state = self._comp_buffer_states.get(buffer_idx)
        capacity = self.comp_buffer_capacity()
        if state is None or state.expert_to_slot.device != device:
            state = CompBufferState(
                expert_to_slot=torch.full(
                    (self.num_experts,),
                    -1,
                    dtype=torch.int32,
                    device=device,
                ),
                slot_to_expert=torch.full(
                    (capacity,),
                    -1,
                    dtype=torch.int32,
                    device=device,
                ),
            )
            self._comp_buffer_states[buffer_idx] = state
        elif state.slot_to_expert.numel() != capacity:
            state.slot_to_expert = torch.full(
                (capacity,),
                -1,
                dtype=torch.int32,
                device=device,
            )
            state.assigned_slot_count = 0
            state.layer_idx = None
        return state

    def _reset_comp_buffer_state(
        self,
        buffer_idx: int,
        layer_idx: int,
        device: torch.device,
    ) -> CompBufferState:
        state = self._get_comp_buffer_state(buffer_idx, device)
        if state.assigned_slot_count > 0:
            active_experts = state.slot_to_expert.narrow(
                0, 0, state.assigned_slot_count)
            valid_mask = active_experts >= 0
            if torch.any(valid_mask):
                state.expert_to_slot.index_fill_(
                    0,
                    active_experts[valid_mask].to(dtype=torch.long),
                    -1,
                )
        state.assigned_slot_count = 0
        state.layer_idx = layer_idx
        return state

    def _acquire_comp_buffers(self) -> tuple[int, torch.Tensor, torch.Tensor]:
        buffer_idx = self.comp_flag
        w13_weight_comp = self.w13_weight_1 if self.comp_flag == 1 else self.w13_weight_2
        w2_weight_comp = self.w2_weight_1 if self.comp_flag == 1 else self.w2_weight_2
        self.comp_flag = 1 if self.comp_flag == 2 else 2
        return buffer_idx, w13_weight_comp, w2_weight_comp

    def _flatten_topk_ids(self, topk_ids) -> list[int]:
        if topk_ids is None:
            return []
        if isinstance(topk_ids, torch.Tensor):
            return topk_ids.reshape(-1).tolist()
        if isinstance(topk_ids, np.ndarray):
            return np.asarray(topk_ids).reshape(-1).tolist()
        if isinstance(topk_ids, list):
            flat_ids: list[int] = []
            for item in topk_ids:
                if isinstance(item, list):
                    flat_ids.extend(item)
                else:
                    flat_ids.append(item)
            return flat_ids
        return list(topk_ids)

    def _dynamic_daemon_or_raise(self):
        if self.prefetch_daemon is None:
            raise RuntimeError(
                "PrefetchDaemon is unavailable in legacy copy-prefetch mode.")
        return self.prefetch_daemon

    def _legacy_daemon_or_raise(self):
        if self.legacy_prefetch_daemon is None:
            raise RuntimeError(
                "LegacyPrefetchDaemon is unavailable in dynamic cache mode.")
        return self.legacy_prefetch_daemon

    def _schedule_prefetch(self, layer_idx: int, topk_ids_pred) -> None:
        if self.use_legacy_copy_prefetch:
            return
        try:
            next_layer_idx = layer_idx + 1
            if next_layer_idx >= self.num_hidden_layers:
                return
            if self._active_prefetch_context_id is None:
                raise RuntimeError("Missing active prefetch context.")
            self._dynamic_daemon_or_raise().schedule_prefetch(
                self._active_prefetch_context_id,
                next_layer_idx,
                topk_ids_pred,
                self._load_targets.get(next_layer_idx),
            )
        except ValueError:
            # logger.warning("Cannot find layer_prefix for layer_idx=%s", layer_idx)
            pass

    def _schedule_prefetch_legacy_copy_path(self, layer_idx: int, topk_ids_pred) -> None:
        """Legacy lightweight prefetch scheduling used by copy path ablation."""
        if not self.use_legacy_copy_prefetch:
            return
        if isinstance(topk_ids_pred, torch.Tensor):
            if topk_ids_pred.numel() == 0:
                prefetch_expert_ids: set[int] = set()
            else:
                prefetch_expert_ids = set(
                    torch.unique(topk_ids_pred).tolist())
        elif topk_ids_pred is None or topk_ids_pred == []:
            prefetch_expert_ids = set()
        else:
            prefetch_expert_ids = set(topk_ids_pred)

        try:
            if layer_idx >= self.num_hidden_layers:
                next_layer_idx = self.first_k_dense_replace
            else:
                next_layer_idx = layer_idx + 1
            if next_layer_idx >= self.num_hidden_layers:
                return

            # Keep legacy semantics for copy-path ablation: merge dynamic-load
            # targets into predicted expert IDs and avoid block-load scheduling.
            next_load = self._load_targets.pop(next_layer_idx, None)
            if next_load:
                prefetch_expert_ids.update(next_load)

            self._legacy_daemon_or_raise().schedule_prefetch(
                next_layer_idx,
                sorted(prefetch_expert_ids),
            )
        except ValueError:
            pass

    def _prepare_layer_prefetch(
        self,
        layer_idx: int,
        buffer_idx: int,
        num_experts: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.cuda.Event | None, torch.Tensor | None,
               bool]:
        if self.use_legacy_copy_prefetch:
            raise RuntimeError(
                "_prepare_layer_prefetch should not be used in legacy mode.")
        if self._active_prefetch_context_id is None:
            raise RuntimeError("Missing active prefetch context.")
        ready_mask, handoff_event, pending_mask, has_block_loads = (
            self._dynamic_daemon_or_raise().notify_layer_loading(
                self._active_prefetch_context_id,
                layer_idx,
                buffer_idx=buffer_idx,
                num_experts=num_experts,
                device=device,
            ))
        if has_block_loads:
            self._activate_pending_loads(layer_idx)
        return ready_mask, handoff_event, pending_mask, has_block_loads

    def _build_offloaded_compute_inputs(
        self,
        buffer_idx: int,
        w13_weight_comp: torch.Tensor,
        w2_weight_comp: torch.Tensor,
    ) -> OffloadedExpertComputeInputs:
        return self._get_offloaded_compute_inputs(
            buffer_idx,
            w13_weight_comp,
            w2_weight_comp,
        )

    def begin_prefetch_context(self, context_id: int) -> None:
        self._active_prefetch_context_id = context_id
        if self.use_legacy_copy_prefetch:
            return
        daemon = self._dynamic_daemon_or_raise()
        daemon.create_context(context_id)
        daemon.set_foreground_context(context_id)

    def finish_prefetch_context(self, context_id: int) -> None:
        if not self.use_legacy_copy_prefetch:
            self._dynamic_daemon_or_raise().clear_context(context_id)
        if self._active_prefetch_context_id == context_id:
            self._active_prefetch_context_id = None

    def _get_offloaded_compute_inputs(
        self,
        buffer_idx: int,
        w13_weight_comp: torch.Tensor,
        w2_weight_comp: torch.Tensor,
    ) -> OffloadedExpertComputeInputs:
        num_experts = self.num_experts
        device = w13_weight_comp.device
        int_kwargs = {
            "dtype": torch.int32,
            "device": device,
        }
        cache_attr = f"_offloaded_compute_inputs_{buffer_idx}"
        compute_inputs = getattr(self, cache_attr, None)
        needs_alloc = (
            compute_inputs is None
            or compute_inputs.expert_source is None
            or compute_inputs.expert_source.size(0) != num_experts
            or compute_inputs.expert_source.device != device
        )
        if needs_alloc:
            compute_inputs = OffloadedExpertComputeInputs(
                w13_weight_comp=w13_weight_comp,
                w2_weight_comp=w2_weight_comp,
                w13_blocks=self.w13_blocks,
                w2_blocks=self.w2_blocks,
                expert_source=torch.empty((num_experts,), **int_kwargs),
                comp_expert_to_slot=torch.empty((num_experts,), **int_kwargs),
                cache_w1_block_ids=torch.empty((num_experts,), **int_kwargs),
                cache_w2_block_ids=torch.empty((num_experts,), **int_kwargs),
                cache_w3_block_ids=torch.empty((num_experts,), **int_kwargs),
            )
            setattr(self, cache_attr, compute_inputs)

        compute_inputs.w13_weight_comp = w13_weight_comp
        compute_inputs.w2_weight_comp = w2_weight_comp
        compute_inputs.w13_blocks = self.w13_blocks
        compute_inputs.w2_blocks = self.w2_blocks
        buffer_state = self._get_comp_buffer_state(buffer_idx, device)
        compute_inputs.comp_expert_to_slot = buffer_state.expert_to_slot
        return compute_inputs

    def _get_selection_mask_buffer(
        self,
        buffer_idx: int,
        num_experts: int,
        device: torch.device,
    ) -> torch.Tensor:
        selection_masks = getattr(self, "_selection_masks", None)
        if selection_masks is None:
            selection_masks = {}
            self._selection_masks = selection_masks

        selected_mask = selection_masks.get(buffer_idx)
        if (selected_mask is None or selected_mask.numel() != num_experts
                or selected_mask.device != device):
            selected_mask = torch.empty((num_experts, ),
                                        dtype=torch.bool,
                                        device=device)
            selection_masks[buffer_idx] = selected_mask
        selected_mask.zero_()
        return selected_mask

    def _get_offloaded_prepare_state(
        self,
        buffer_idx: int,
        num_experts: int,
        miss_capacity: int,
        device: torch.device,
        compact_capacity: int | None = None,
    ) -> OffloadedPrepareState:
        prepare_states = getattr(self, "_offloaded_prepare_states", None)
        if prepare_states is None:
            prepare_states = {}
            self._offloaded_prepare_states = prepare_states

        state = prepare_states.get(buffer_idx)
        if (
            state is None
            or state.seen_buffer.numel() != num_experts
            or state.seen_buffer.device != device
        ):
            state = OffloadedPrepareState(
                seen_buffer=torch.zeros((num_experts, ),
                                        dtype=torch.int32,
                                        device=device),
                miss_expert_ids=torch.empty((miss_capacity, ),
                                            dtype=torch.long,
                                            device=device),
                miss_count=torch.empty((1, ),
                                       dtype=torch.int32,
                                       device=device),
            )
            prepare_states[buffer_idx] = state
        elif state.miss_expert_ids.numel() < miss_capacity:
            state.miss_expert_ids = torch.empty((miss_capacity, ),
                                               dtype=torch.long,
                                               device=device)

        if compact_capacity is not None:
            if (state.miss_slot_ids is None
                    or state.miss_slot_ids.numel() < miss_capacity
                    or state.miss_slot_ids.device != device):
                state.miss_slot_ids = torch.empty((miss_capacity, ),
                                                  dtype=torch.long,
                                                  device=device)
            if (state.pending_mask is None
                    or state.pending_mask.numel() != num_experts
                    or state.pending_mask.device != device):
                state.pending_mask = torch.empty((num_experts, ),
                                                 dtype=torch.bool,
                                                 device=device)
            if (state.slot_live_flags is None
                    or state.slot_live_flags.numel() != compact_capacity
                    or state.slot_live_flags.device != device):
                state.slot_live_flags = torch.empty((compact_capacity, ),
                                                    dtype=torch.int32,
                                                    device=device)

        if state.epoch >= torch.iinfo(torch.int32).max:
            state.seen_buffer.zero_()
            state.epoch = 1
        else:
            state.epoch += 1
        return state

    def _prepare_and_copy_compact_offloaded_compute_inputs(
        self,
        buffer_idx: int,
        compute_inputs: OffloadedExpertComputeInputs,
        layer_weights: OffloadedLayerWeights,
        layer_idx: int,
        topk_ids: torch.Tensor,
        prefetched_ready_mask: torch.Tensor,
        pending_prefetch_mask: torch.Tensor,
    ) -> None:
        assert compute_inputs.expert_source is not None
        assert compute_inputs.comp_expert_to_slot is not None
        assert compute_inputs.cache_w1_block_ids is not None
        assert compute_inputs.cache_w2_block_ids is not None
        assert compute_inputs.cache_w3_block_ids is not None
        assert compute_inputs.w13_weight_comp is not None
        assert compute_inputs.w2_weight_comp is not None
        assert layer_weights.w13_uva is not None
        assert layer_weights.w2_uva is not None

        device = compute_inputs.expert_source.device
        miss_capacity = max(1, int(topk_ids.numel()))
        compact_capacity = compute_inputs.w13_weight_comp.size(0)
        state = self._get_offloaded_prepare_state(
            buffer_idx,
            self.num_experts,
            miss_capacity,
            device,
            compact_capacity=compact_capacity,
        )
        assert state.miss_slot_ids is not None
        assert state.pending_mask is not None
        assert state.slot_live_flags is not None

        if prefetched_ready_mask.numel() == 0:
            prefetched_ready_mask = torch.empty((0, ),
                                                dtype=torch.bool,
                                                device=device)
        if pending_prefetch_mask.numel() == 0:
            state.pending_mask.zero_()
        else:
            state.pending_mask.copy_(pending_prefetch_mask)

        buffer_state = self._get_comp_buffer_state(buffer_idx, device)
        if buffer_state.assigned_slot_count > compact_capacity:
            raise RuntimeError(
                "Compact expert buffer state exceeded capacity before compute: "
                f"buffer={buffer_idx}, assigned={buffer_state.assigned_slot_count}, "
                f"capacity={compact_capacity}."
            )

        torch.ops._C.prepare_and_copy_offloaded_compute_inputs_compact(
            compute_inputs.expert_source,
            compute_inputs.comp_expert_to_slot,
            compute_inputs.cache_w1_block_ids,
            compute_inputs.cache_w2_block_ids,
            compute_inputs.cache_w3_block_ids,
            state.miss_expert_ids,
            state.miss_slot_ids,
            state.miss_count,
            state.seen_buffer,
            state.slot_live_flags,
            self.block_table.block_table.gpu[layer_idx],
            topk_ids.contiguous(),
            prefetched_ready_mask,
            state.pending_mask,
            buffer_state.slot_to_expert,
            buffer_state.assigned_slot_count,
            compute_inputs.w13_weight_comp,
            compute_inputs.w2_weight_comp,
            layer_weights.w13_uva,
            layer_weights.w2_uva,
            state.epoch,
        )
        buffer_state.assigned_slot_count = compact_capacity

    def _prepare_offloaded_compute_inputs(
        self,
        buffer_idx: int,
        compute_inputs: OffloadedExpertComputeInputs,
        layer_idx: int,
        topk_ids: torch.Tensor,
        prefetched_ready_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        assert compute_inputs.expert_source is not None
        assert compute_inputs.cache_w1_block_ids is not None
        assert compute_inputs.cache_w2_block_ids is not None
        assert compute_inputs.cache_w3_block_ids is not None

        device = compute_inputs.expert_source.device
        num_experts = compute_inputs.expert_source.size(0)
        state = self._get_offloaded_prepare_state(
            buffer_idx,
            num_experts,
            num_experts,
            device,
        )
        ready_mask = prefetched_ready_mask
        if ready_mask is None:
            ready_mask = torch.empty((0, ), dtype=torch.bool, device=device)

        torch.ops._C.prepare_offloaded_compute_inputs(
            compute_inputs.expert_source,
            compute_inputs.cache_w1_block_ids,
            compute_inputs.cache_w2_block_ids,
            compute_inputs.cache_w3_block_ids,
            state.miss_expert_ids,
            state.miss_count,
            state.seen_buffer,
            self.block_table.block_table.gpu[layer_idx],
            topk_ids.contiguous(),
            ready_mask,
            state.epoch,
        )
        return state.miss_expert_ids

    def _prepare_and_copy_offloaded_compute_inputs(
        self,
        buffer_idx: int,
        compute_inputs: OffloadedExpertComputeInputs,
        layer_weights: OffloadedLayerWeights,
        layer_idx: int,
        topk_ids: torch.Tensor,
        prefetched_ready_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        assert compute_inputs.expert_source is not None
        assert compute_inputs.cache_w1_block_ids is not None
        assert compute_inputs.cache_w2_block_ids is not None
        assert compute_inputs.cache_w3_block_ids is not None
        assert compute_inputs.w13_weight_comp is not None
        assert compute_inputs.w2_weight_comp is not None
        assert layer_weights.w13_uva is not None
        assert layer_weights.w2_uva is not None

        device = compute_inputs.expert_source.device
        num_experts = compute_inputs.expert_source.size(0)
        miss_capacity = max(1, int(topk_ids.numel()))
        state = self._get_offloaded_prepare_state(
            buffer_idx,
            num_experts,
            miss_capacity,
            device,
        )
        ready_mask = prefetched_ready_mask
        if ready_mask is None:
            ready_mask = torch.empty((0, ), dtype=torch.bool, device=device)

        torch.ops._C.prepare_and_copy_offloaded_compute_inputs(
            compute_inputs.expert_source,
            compute_inputs.cache_w1_block_ids,
            compute_inputs.cache_w2_block_ids,
            compute_inputs.cache_w3_block_ids,
            state.miss_expert_ids,
            state.miss_count,
            state.seen_buffer,
            self.block_table.block_table.gpu[layer_idx],
            topk_ids.contiguous(),
            ready_mask,
            compute_inputs.w13_weight_comp,
            compute_inputs.w2_weight_comp,
            layer_weights.w13_uva,
            layer_weights.w2_uva,
            state.epoch,
        )
        return state.miss_expert_ids

    def _copy_expert_to_comp_buffer(
        self,
        layer_weights: OffloadedLayerWeights,
        expert_id: int,
        row_idx: int,
        w13_weight_comp: torch.Tensor,
        w2_weight_comp: torch.Tensor,
    ) -> None:
        assert layer_weights.w13_uva is not None
        assert layer_weights.w2_uva is not None
        w13_weight_comp[row_idx].copy_(layer_weights.w13_uva[expert_id],
                                       non_blocking=True)
        w2_weight_comp[row_idx].copy_(layer_weights.w2_uva[expert_id],
                                      non_blocking=True)

    def _copy_expert_to_blocks(
        self,
        layer_weights: OffloadedLayerWeights,
        expert_id: int,
        block_row: tuple[int, int, int],
    ) -> None:
        assert layer_weights.w13_uva is not None
        assert layer_weights.w2_uva is not None
        w1_block_id, w2_block_id, w3_block_id = block_row
        w13_weight = layer_weights.w13_uva[expert_id]
        intermediate_size = layer_weights.intermediate_size
        self.w13_blocks[w1_block_id].copy_(w13_weight[:intermediate_size],
                                           non_blocking=True)
        self.w13_blocks[w3_block_id].copy_(w13_weight[intermediate_size:],
                                           non_blocking=True)
        self.w2_blocks[w2_block_id].copy_(layer_weights.w2_uva[expert_id],
                                          non_blocking=True)

    def _select_offloaded_experts(
        self,
        layer_idx: int,
        topk_ids: torch.Tensor,
        buffer_idx: int,
        prefetched_ready_mask: torch.Tensor | None,
        device: torch.device,
    ) -> OffloadedExpertSelection:
        layer_table = self.block_table.block_table.gpu[layer_idx]
        empty_rows = torch.empty((0, layer_table.size(1)),
                                 dtype=layer_table.dtype,
                                 device=device)
        empty_ids = torch.empty((0, ), dtype=torch.long, device=device)
        if topk_ids.numel() == 0:
            return OffloadedExpertSelection(
                routed_expert_ids=empty_ids,
                cached_expert_ids=empty_ids,
                uncached_expert_ids=empty_ids,
                cached_block_rows=empty_rows,
            )

        flat_ids = topk_ids.reshape(-1).to(device=device, dtype=torch.long)
        selected_mask = self._get_selection_mask_buffer(
            buffer_idx,
            layer_table.size(0),
            device,
        )
        selected_mask.index_fill_(0, flat_ids, True)
        routed_expert_ids = torch.nonzero(selected_mask,
                                          as_tuple=False).flatten()
        if routed_expert_ids.numel() == 0:
            return OffloadedExpertSelection(
                routed_expert_ids=routed_expert_ids,
                cached_expert_ids=empty_ids,
                uncached_expert_ids=empty_ids,
                cached_block_rows=empty_rows,
            )

        block_rows = torch.index_select(layer_table, 0, routed_expert_ids)
        if prefetched_ready_mask is not None and prefetched_ready_mask.numel() > 0:
            active_mask = ~torch.index_select(prefetched_ready_mask, 0,
                                              routed_expert_ids)
            active_expert_ids = routed_expert_ids[active_mask]
            active_block_rows = block_rows[active_mask]
        else:
            active_expert_ids = routed_expert_ids
            active_block_rows = block_rows

        if active_expert_ids.numel() == 0:
            return OffloadedExpertSelection(
                routed_expert_ids=routed_expert_ids,
                cached_expert_ids=empty_ids,
                uncached_expert_ids=empty_ids,
                cached_block_rows=empty_rows,
            )

        cached_rows_mask = active_block_rows[:, 0] != -1
        cached_expert_ids = active_expert_ids[cached_rows_mask]
        uncached_expert_ids = active_expert_ids[~cached_rows_mask]
        return OffloadedExpertSelection(
            routed_expert_ids=routed_expert_ids,
            cached_expert_ids=cached_expert_ids,
            uncached_expert_ids=uncached_expert_ids,
            cached_block_rows=active_block_rows[cached_rows_mask],
        )

    def _copy_cached_experts_to_comp_buffer(
        self,
        selection: OffloadedExpertSelection,
        w13_weight_comp: torch.Tensor,
        w2_weight_comp: torch.Tensor,
    ) -> None:
        cache_expert_ids = selection.cached_expert_ids
        if cache_expert_ids.numel() == 0:
            return
        cache_block_rows = selection.cached_block_rows
        w1_block_ids = cache_block_rows[:, 0].to(dtype=torch.long)
        w2_block_ids = cache_block_rows[:, 1].to(dtype=torch.long)
        w3_block_ids = cache_block_rows[:, 2].to(dtype=torch.long)
        intermediate_size = self.w13_blocks.size(1)
        # Write cached w1/w3 blocks directly into the packed destination halves.
        w13_weight_comp[:, :intermediate_size].index_copy_(
            0,
            cache_expert_ids,
            torch.index_select(self.w13_blocks, 0, w1_block_ids),
        )
        w13_weight_comp[:, intermediate_size:].index_copy_(
            0,
            cache_expert_ids,
            torch.index_select(self.w13_blocks, 0, w3_block_ids),
        )
        w2_weight_comp.index_copy_(
            0,
            cache_expert_ids,
            torch.index_select(self.w2_blocks, 0, w2_block_ids),
        )

    def _copy_uncached_experts_to_comp_buffer(
        self,
        layer_weights: OffloadedLayerWeights,
        selection: OffloadedExpertSelection,
        w13_weight_comp: torch.Tensor,
        w2_weight_comp: torch.Tensor,
    ) -> None:
        self._copy_uncached_expert_ids_to_comp_buffer(
            layer_weights,
            selection.uncached_expert_ids,
            w13_weight_comp,
            w2_weight_comp,
        )

    def _copy_uncached_expert_ids_to_comp_buffer(
        self,
        layer_weights: OffloadedLayerWeights,
        uncached_ids: torch.Tensor,
        w13_weight_comp: torch.Tensor,
        w2_weight_comp: torch.Tensor,
    ) -> None:
        if uncached_ids.numel() == 0:
            return
        assert layer_weights.w13_uva is not None
        assert layer_weights.w2_uva is not None
        torch.ops._C.copy_uncached_experts_to_comp(
            w13_weight_comp,
            w2_weight_comp,
            layer_weights.w13_uva,
            layer_weights.w2_uva,
            uncached_ids,
        )

    def _set_cpu_block_row(
        self,
        layer_idx: int,
        expert_id: int,
        block_row: tuple[int, int, int],
    ) -> None:
        self.block_table.block_table.cpu[layer_idx, expert_id] = torch.tensor(
            block_row,
            dtype=torch.int32,
        )

    def _clear_cpu_block_row(
        self,
        layer_idx: int,
        expert_id: int,
    ) -> None:
        self.block_table.block_table.cpu[layer_idx, expert_id].fill_(-1)

    def _reserved_block_row(
        self,
        layer_idx: int,
        expert_id: int,
    ) -> tuple[int, int, int] | None:
        return self._pending_reserved_blocks.get(layer_idx, {}).get(expert_id)

    def _has_pending_cache_delta_state(self) -> bool:
        return bool(
            self._active_cache_delta_id is not None
            or self._load_targets
            or self._evict_targets
            or self._pending_reserved_blocks
        )

    def _activate_pending_loads(self, layer_idx: int) -> None:
        pending_blocks = self._pending_reserved_blocks.pop(layer_idx, None)
        if not pending_blocks:
            self._load_targets.pop(layer_idx, None)
            self._maybe_complete_active_cache_delta()
            return

        for expert_id, (w1_block_id, w2_block_id, w3_block_id) in pending_blocks.items():
            self._set_cpu_block_row(
                layer_idx,
                expert_id,
                (w1_block_id, w2_block_id, w3_block_id),
            )
            self.block_table.commit_expert_row(layer_idx, expert_id)
        self._load_targets.pop(layer_idx, None)
        self._maybe_complete_active_cache_delta()

    def commit_post_execute_evictions(self) -> None:
        if not self._evict_targets:
            self._maybe_complete_active_cache_delta()
            return

        affected_layers = sorted(self._evict_targets)
        for layer_idx in affected_layers:
            for expert_id in self._evict_targets[layer_idx]:
                self._clear_cpu_block_row(layer_idx, expert_id)
                if self._evict_commit_mode == "row":
                    self.block_table.commit_expert_row(layer_idx, expert_id)

        if self._evict_commit_mode == "table":
            self.block_table.commit_all_experts()

        self._evict_targets.clear()
        self._evict_commit_mode = "row"
        self._maybe_complete_active_cache_delta()

    def _maybe_complete_active_cache_delta(self) -> None:
        if self._active_cache_delta_id is None:
            return
        if self._load_targets or self._evict_targets or self._pending_reserved_blocks:
            return
        self._active_cache_delta_id = None

    def assert_no_active_cache_delta(self) -> None:
        if not self._has_pending_cache_delta_state():
            return
        raise RuntimeError(
            "Backend expert cache delta state was not fully drained within "
            "the current step: "
            f"active_delta_id={self._active_cache_delta_id}, "
            f"load_targets={list(self._load_targets)}, "
            f"evict_targets={list(self._evict_targets)}, "
            f"pending_reserved_layers={list(self._pending_reserved_blocks)}."
        )

    def apply_cache_delta(self, delta) -> None:
        """Receive an ExpertCacheDelta from the frontend and decompose it
        into staged load / evict targets for the current execution round."""
        if self._has_pending_cache_delta_state():
            raise RuntimeError(
                "Attempted to apply a new expert-cache delta before the "
                "previous delta was fully applied."
            )
        self._active_cache_delta_id = delta.delta_id
        self._evict_commit_mode = delta.evict_commit_mode

        for layer_id, expert_id in delta.experts_to_evict:
            self._evict_targets.setdefault(layer_id, set()).add(expert_id)

        for layer_id, expert_id in delta.experts_to_load:
            self._load_targets.setdefault(layer_id, set()).add(expert_id)

        if delta.new_expert_to_block:
            for (layer_id, expert_id, _), _block_id in delta.new_expert_to_block.items():
                if expert_id not in self._load_targets.get(layer_id, set()):
                    continue
                layer_blocks = self._pending_reserved_blocks.setdefault(layer_id, {})
                if expert_id not in layer_blocks:
                    layer_blocks[expert_id] = (
                        delta.new_expert_to_block[(layer_id, expert_id, "w1")],
                        delta.new_expert_to_block[(layer_id, expert_id, "w2")],
                        delta.new_expert_to_block[(layer_id, expert_id, "w3")],
                    )

        self._maybe_complete_active_cache_delta()

    def get_layer_prefix(self, layer_id: int):
        layer_prefix = self.layer_idx_to_prefix.get(layer_id)
        if layer_prefix is None:
            raise KeyError(f"No layer prefix registered for layer_id={layer_id}")
        return layer_prefix
    
    def get_layer_index(self, layer_prefix: str):
        from vllm.model_executor.models.utils import extract_layer_index

        layer_idx = self.layer_prefix_to_idx.get(layer_prefix)
        if layer_idx is not None:
            return layer_idx
        layer_idx = extract_layer_index(layer_prefix)
        self.layer_prefix_to_idx[layer_prefix] = layer_idx
        existing_prefix = self.layer_idx_to_prefix.get(layer_idx)
        if existing_prefix is not None and existing_prefix != layer_prefix:
            raise KeyError(
                "Conflicting layer prefix registration for "
                f"layer_id={layer_idx}: {existing_prefix} vs {layer_prefix}"
            )
        self.layer_idx_to_prefix[layer_idx] = layer_prefix
        return layer_idx
    
    def initialize_experts(self, expert_to_block: dict[(int, int, str), int]):
        self._evict_commit_mode = "row"
        if self.w13_blocks is None or self.w2_blocks is None:
            assert False, "Failed to initialize experts, w13_blocks or w2_blocks is None"
        
        for (layer_id, expert_id, w123) in expert_to_block:
            block_id = expert_to_block[(layer_id, expert_id, w123)]
            layer_prefix = self.get_layer_prefix(layer_id)
            layer_weights = self.expert_params.get(layer_prefix)
            if layer_weights is None:
                raise KeyError(f"No expert params found for {layer_prefix}")
            if w123 == "w1":
                param = layer_weights.w13_cpu[expert_id,
                                              :layer_weights.intermediate_size]
                blocks = self.w13_blocks
            elif w123 == "w3":
                param = layer_weights.w13_cpu[expert_id,
                                              layer_weights.intermediate_size:]
                blocks = self.w13_blocks
            elif w123 == "w2":
                param = layer_weights.w2_cpu[expert_id]
                blocks = self.w2_blocks
            else:
                assert False, "Invalid w123"

            assert blocks[block_id].shape == param.shape, "Shape mismatch"
            blocks[block_id].copy_(param, non_blocking=True)

    def init_w13_w2_weight(self, w13_weight, w2_weight):
        device = torch.cuda.current_device()
        if self.w13_weight_1 is None and self.w13_weight_2 is None:
            capacity = self.comp_buffer_capacity()
            w13_shape = (capacity, w13_weight.size(1), w13_weight.size(2))
            w2_shape = (capacity, w2_weight.size(1), w2_weight.size(2))
            self.w13_weight_1 = torch.empty(
                w13_shape,
                dtype=w13_weight.dtype,
                device=device,
            )
            self.w2_weight_1 = torch.empty(
                w2_shape,
                dtype=w2_weight.dtype,
                device=device,
            )
            self.w13_weight_2 = torch.empty_like(self.w13_weight_1)
            self.w2_weight_2 = torch.empty_like(self.w2_weight_1)
            # print(f"[debug] init w13_weight and w2_weight on {device=}, {self.w13_weight_1.shape=}, {self.w2_weight_1.shape=}")
       
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

    def get_experts_with_topk_ids(self, layer_prefix, topk_ids, topk_ids_pred):
        if self.compact_comp_buffer_enabled():
            raise RuntimeError(
                "Copy-based expert gathering is disabled when compact no-copy "
                "expert buffers are enabled."
            )

        layer_idx = self.get_layer_index(layer_prefix)
        unique_ids = torch.unique(topk_ids).tolist()

        if self.w13_weight_1 is None or self.w2_weight_1 is None or self.w13_weight_2 is None or self.w2_weight_2 is None:
            logger.error(f"w13_weight or w2_weight not initialized for {layer_prefix}")
            return None, None

        if layer_prefix not in self.expert_params:
            logger.error(f"No expert params found for {layer_prefix}")
            return None, None
        
        layer_weights = self.expert_params[layer_prefix]
        stream = StreamContext.memory_stream

        prefetched_experts = self._legacy_daemon_or_raise().notify_layer_loading(
            layer_idx)
        if prefetched_experts is None:
            prefetched_experts = []

        w13_weight_comp = self.w13_weight_1 if self.comp_flag == 1 else self.w13_weight_2
        w2_weight_comp = self.w2_weight_1 if self.comp_flag == 1 else self.w2_weight_2
        self.comp_flag = 1 if self.comp_flag == 2 else 2

        intermediate_size = layer_weights.intermediate_size
        with torch.cuda.stream(stream):
            for expert_id in unique_ids:
                if expert_id in prefetched_experts:
                    continue

                w1_block_id, w2_block_id, w3_block_id = self.block_table.get_device_block_id(
                    layer_idx, expert_id)
                if w1_block_id != -1:
                    w1_param = self.w13_blocks[w1_block_id]
                    w2_param = self.w2_blocks[w2_block_id]
                    w3_param = self.w13_blocks[w3_block_id]
                else:
                    w13_param = layer_weights.w13_cpu[expert_id]
                    w2_cpu = layer_weights.w2_cpu[expert_id]
                    w1_param = self.load_param(w13_param[:intermediate_size])
                    w3_param = self.load_param(w13_param[intermediate_size:])
                    w2_param = self.load_param(w2_cpu)

                w13_weight_comp[expert_id][:intermediate_size].copy_(w1_param.data, non_blocking=True)
                w13_weight_comp[expert_id][intermediate_size:].copy_(w3_param.data, non_blocking=True)
                w2_weight_comp[expert_id].copy_(w2_param.data, non_blocking=True)

        stream.synchronize()

        self._schedule_prefetch_legacy_copy_path(layer_idx, topk_ids_pred)
        return w13_weight_comp, w2_weight_comp

    def get_offloaded_compute_inputs(self, layer_prefix, topk_ids, topk_ids_pred):

        # === PIECEWISE CUDA Graph 模式处理 ===
        if self.is_capturing_cuda_graph():
            return self._get_offloaded_compute_inputs_cudagraph(
                layer_prefix, topk_ids, topk_ids_pred
            )
        # ======================================

        layer_idx = self.get_layer_index(layer_prefix)

        if (
            self.w13_weight_1 is None
            or self.w2_weight_1 is None
            or self.w13_weight_2 is None
            or self.w2_weight_2 is None
        ):
            logger.error(f"w13_weight or w2_weight not initialized for {layer_prefix}")
            return None

        if layer_prefix not in self.expert_params:
            logger.error(f"No expert params found for {layer_prefix}")
            return None

        layer_weights = self.expert_params[layer_prefix]
        stream = StreamContext.memory_stream

        next_buffer_idx = self.comp_flag
        prefetched_ready_mask, prefetch_handoff_event, pending_prefetch_mask, _ = self._prepare_layer_prefetch(
            layer_idx,
            buffer_idx=next_buffer_idx,
            num_experts=self.num_experts,
            device=self.w13_weight_1.device,
        )

        buffer_idx, w13_weight_comp, w2_weight_comp = self._acquire_comp_buffers()
        compute_inputs = self._get_offloaded_compute_inputs(
            buffer_idx,
            w13_weight_comp,
            w2_weight_comp,
        )
        current_stream = torch.cuda.current_stream()

        with torch.cuda.stream(stream):
            stream.wait_stream(current_stream)
            if self.compact_comp_buffer_enabled():
                ready_mask = prefetched_ready_mask
                pending_mask = pending_prefetch_mask
                device = compute_inputs.expert_source.device
                if ready_mask is None:
                    ready_mask = torch.empty((0, ),
                                             dtype=torch.bool,
                                             device=device)
                if pending_mask is None:
                    pending_mask = torch.empty((0, ),
                                               dtype=torch.bool,
                                               device=device)
                if prefetch_handoff_event is not None:
                    stream.wait_event(prefetch_handoff_event)
                    if pending_mask.numel() > 0:
                        ready_mask = torch.logical_or(ready_mask, pending_mask)
                        pending_mask = torch.empty((0, ),
                                                   dtype=torch.bool,
                                                   device=ready_mask.device)
                self._prepare_and_copy_compact_offloaded_compute_inputs(
                    buffer_idx,
                    compute_inputs,
                    layer_weights,
                    layer_idx,
                    topk_ids,
                    ready_mask,
                    pending_mask,
                )
            else:
                if prefetch_handoff_event is not None:
                    stream.wait_event(prefetch_handoff_event)
                self._prepare_and_copy_offloaded_compute_inputs(
                    buffer_idx,
                    compute_inputs,
                    layer_weights,
                    layer_idx,
                    topk_ids,
                    prefetched_ready_mask,
                )

        current_stream.wait_stream(stream)

        self._schedule_prefetch(layer_idx, topk_ids_pred)

        return compute_inputs

    def _get_offloaded_compute_inputs_cudagraph(self, layer_prefix, topk_ids, topk_ids_pred):
        """PIECEWISE 模式下同步获取计算输入的 CUDA Graph 安全实现。"""
        layer_idx = self.get_layer_index(layer_prefix)

        if (self.w13_weight_1 is None
                or self.w2_weight_1 is None
                or self.w13_weight_2 is None
                or self.w2_weight_2 is None):
            logger.error(f"w13_weight or w2_weight not initialized for {layer_prefix}")
            return None

        if layer_prefix not in self.expert_params:
            logger.error(f"No expert params found for {layer_prefix}")
            return None

        layer_weights = self.expert_params[layer_prefix]

        # 使用固定缓冲区 1（地址稳定）
        buffer_idx = 1
        w13_weight_comp = self.w13_weight_1
        w2_weight_comp = self.w2_weight_1

        compute_inputs = self._get_offloaded_compute_inputs(
            buffer_idx,
            w13_weight_comp,
            w2_weight_comp,
        )

        # 同步复制路径（不使用预取）
        if self.compact_comp_buffer_enabled():
            self._prepare_and_copy_compact_offloaded_compute_inputs(
                buffer_idx,
                compute_inputs,
                layer_weights,
                layer_idx,
                topk_ids,
                torch.empty((0,), dtype=torch.bool, device=w13_weight_comp.device),
                torch.empty((0,), dtype=torch.bool, device=w13_weight_comp.device),
            )
        else:
            self._prepare_and_copy_offloaded_compute_inputs(
                buffer_idx,
                compute_inputs,
                layer_weights,
                layer_idx,
                topk_ids,
                None,
            )

        # 不调度预取（捕获期间跳过）
        return compute_inputs

    def register_moe_module(self, layer_prefix, module):
        from vllm.model_executor.models.utils import extract_layer_index

        if layer_prefix in self.moe_modules:
            logger.error(f"Module with layer_prefix {layer_prefix} already registered.")
            return
        layer_idx = extract_layer_index(layer_prefix)
        existing_prefix = self.layer_idx_to_prefix.get(layer_idx)
        if existing_prefix is not None and existing_prefix != layer_prefix:
            raise KeyError(
                "Duplicate MoE layer registration for "
                f"layer_id={layer_idx}: {existing_prefix} vs {layer_prefix}"
            )
        self.moe_modules[layer_prefix] = weakref.ref(module)
        self.layer_prefixes.append(layer_prefix)
        self.layer_prefix_to_idx[layer_prefix] = layer_idx
        self.layer_idx_to_prefix[layer_idx] = layer_prefix
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
            intermediate_size = module.w13_weight.size(1) // 2
            self.expert_params[layer_prefix] = OffloadedLayerWeights(
                w13_cpu=module.w13_weight.detach(),
                w2_cpu=module.w2_weight.detach(),
                intermediate_size=intermediate_size,
            )

            self.init_w13_w2_weight(w13_weight=module.w13_weight, w2_weight=module.w2_weight)

            # 从GPU卸载权重
            module.w13_weight = None
            module.w2_weight = None

        else:
            # print(f"[debug] weights for {layer_prefix} are not on CPU!!")
            pass

    def is_capturing_cuda_graph(self) -> bool:
        """检查当前流是否正在捕获 CUDA Graph。"""
        return torch.cuda.is_current_stream_capturing()

    def sync_before_graph_capture(self) -> None:
        """在 CUDA Graph 捕获前同步所有待处理的卸载操作。

        由 GPUModelRunner 在开始 CUDA Graph 捕获前调用。
        确保 prefetch_stream 上所有进行中的 H2D 复制完成。
        """
        if not hasattr(self, 'prefetch_daemon'):
            return
        current_stream = torch.cuda.current_stream()
        prefetch_stream = StreamContext.prefetch_stream
        if prefetch_stream is not None:
            current_stream.wait_stream(prefetch_stream)

    def sync_prev_onload(self) -> None:
        """同步之前的卸载操作 - 与 BaseOffloader 接口兼容。"""
        self.sync_before_graph_capture()

    def join_after_forward(self) -> None:
        """在模型前向传播后在图捕获期间加入复制流。

        BackendExpertManager 的无操作实现，因为它不使用分叉复制流。
        """
        pass

    def get_capture_safe_comp_buffers(self) -> tuple[int, torch.Tensor, torch.Tensor]:
        """获取具有稳定内存地址的计算缓冲区以支持图捕获。

        捕获期间，返回固定缓冲区 1，而不是在缓冲区 1 和 2 之间切换。
        这确保了回放期间内存地址的稳定性。
        """
        if self.is_capturing_cuda_graph():
            return 1, self.w13_weight_1, self.w2_weight_1
        return self._acquire_comp_buffers()

    def load_param(self, param):
        if param is None:
            return None
        if param.device.type == "cuda":
            return param
        return param.cuda(non_blocking=True)

    def __del__(self):
        daemon = getattr(self, "prefetch_daemon", None)
        if daemon is not None:
            daemon.shutdown()
        legacy_daemon = getattr(self, "legacy_prefetch_daemon", None)
        if legacy_daemon is not None:
            legacy_daemon.shutdown()


class LegacyPrefetchDaemon:
    """Legacy copy-path prefetch daemon for ablation alignment."""

    def __init__(self, manager: BackendExpertManager):
        self.manager = manager
        self.prefetch_queue: list[int] = []
        self.loaded_queue: list[int] = []
        self.next_layer_idx: int | None = None
        self.shutdown_flag = threading.Event()
        self.lock = threading.Lock()

        self.daemon_thread = threading.Thread(
            target=self._prefetch_worker,
            daemon=True,
        )
        self.daemon_thread.start()

    def schedule_prefetch(self, layer_idx: int, prefetch_expert_ids_pred) -> None:
        with self.lock:
            self.prefetch_queue = []
            self.loaded_queue = []
            self.next_layer_idx = layer_idx
            self.prefetch_queue = list(prefetch_expert_ids_pred)

    def notify_layer_loading(self, layer_idx: int):
        with self.lock:
            if layer_idx == self.next_layer_idx:
                self.prefetch_queue = []
                return self.loaded_queue
        return []

    def shutdown(self) -> None:
        self.shutdown_flag.set()
        if self.daemon_thread.is_alive():
            self.daemon_thread.join(timeout=1)

    def _prefetch_worker(self):
        with torch.cuda.stream(StreamContext.prefetch_stream):
            while not self.shutdown_flag.is_set():
                expert_id = None
                layer_idx = None

                with self.lock:
                    if self.prefetch_queue and self.next_layer_idx is not None:
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

            layer_prefix = self.manager.get_layer_prefix(layer_idx)
            layer_weights = self.manager.expert_params.get(layer_prefix)
            if layer_weights is None:
                return False
            intermediate_size = layer_weights.intermediate_size

            w13_weight_comp = (
                self.manager.w13_weight_1
                if self.manager.comp_flag == 1 else self.manager.w13_weight_2)
            w2_weight_comp = (
                self.manager.w2_weight_1
                if self.manager.comp_flag == 1 else self.manager.w2_weight_2)
            if w13_weight_comp is None or w2_weight_comp is None:
                return False

            w1_block_id, w2_block_id, w3_block_id = (
                self.manager.block_table.get_device_block_id(layer_idx, expert_id))
            if w1_block_id != -1:
                w1_param = self.manager.w13_blocks[w1_block_id]
                w2_param = self.manager.w2_blocks[w2_block_id]
                w3_param = self.manager.w13_blocks[w3_block_id]
            else:
                w13_param = layer_weights.w13_cpu[expert_id]
                w2_cpu = layer_weights.w2_cpu[expert_id]
                w1_param = self.manager.load_param(w13_param[:intermediate_size])
                w3_param = self.manager.load_param(w13_param[intermediate_size:])
                w2_param = self.manager.load_param(w2_cpu)

            w13_weight_comp[expert_id][:intermediate_size].copy_(
                w1_param.data, non_blocking=True)
            w13_weight_comp[expert_id][intermediate_size:].copy_(
                w3_param.data, non_blocking=True)
            w2_weight_comp[expert_id].copy_(w2_param.data, non_blocking=True)

            StreamContext.prefetch_stream.synchronize()
            with self.lock:
                self.loaded_queue.append(expert_id)
            return True
        except Exception as e:
            logger.warning(
                "Legacy prefetch failed for layer=%s expert=%s: %s",
                layer_idx,
                expert_id,
                e,
            )
            return False


class PrefetchDaemon:
    """运行时预取专家权重"""

    def __init__(self, manager: BackendExpertManager):
        self.manager = manager
        self.contexts: dict[int, PrefetchContext] = {}
        self.foreground_context_id: int | None = None
        self._context_rr: deque[int] = deque()
        self._ready_masks: dict[int, torch.Tensor] = {}
        self._pending_masks: dict[int, torch.Tensor] = {}

        self.shutdown_flag = threading.Event()
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)

        self.daemon_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.daemon_thread.start()

    def _topk_ids_to_cpu_tensor(self, topk_ids) -> torch.Tensor:
        if topk_ids is None:
            return torch.empty((0, ), dtype=torch.long)
        if isinstance(topk_ids, torch.Tensor):
            if topk_ids.numel() == 0:
                return torch.empty((0, ), dtype=torch.long)
            return topk_ids.detach().to(device="cpu",
                                        dtype=torch.long).reshape(-1)
        if isinstance(topk_ids, np.ndarray):
            if topk_ids.size == 0:
                return torch.empty((0, ), dtype=torch.long)
            return torch.from_numpy(
                np.asarray(topk_ids, dtype=np.int64).reshape(-1))

        flat_ids = self.manager._flatten_topk_ids(topk_ids)
        if not flat_ids:
            return torch.empty((0, ), dtype=torch.long)
        return torch.tensor(flat_ids, dtype=torch.long)

    def _filter_uncached_expert_ids_cpu(
        self,
        layer_idx: int,
        expert_ids_cpu: torch.Tensor,
    ) -> torch.Tensor:
        if expert_ids_cpu.numel() == 0:
            return expert_ids_cpu
        block_rows = self.manager.block_table.block_table.np[layer_idx,
                                                             expert_ids_cpu.numpy()]
        uncached_mask = torch.from_numpy(block_rows[:, 0] == -1)
        uncached_ids = expert_ids_cpu[uncached_mask]
        pending_delta_ids = self.manager._load_targets.get(layer_idx, set())
        if not pending_delta_ids:
            return uncached_ids
        keep_mask = torch.tensor(
            [int(expert_id) not in pending_delta_ids for expert_id in uncached_ids.tolist()],
            dtype=torch.bool,
        )
        return uncached_ids[keep_mask]

    def _build_block_load_ids(
        self,
        layer_idx: int,
        next_load_ids: set[int],
    ) -> list[int]:
        return [
            expert_id
            for expert_id in sorted(next_load_ids)
            if self.manager._reserved_block_row(layer_idx, expert_id) is not None
        ]

    def _build_comp_prefetch_ids(
        self,
        layer_idx: int,
        topk_ids_pred,
        block_load_ids: set[int],
    ) -> list[int]:
        predicted_ids_cpu = self._topk_ids_to_cpu_tensor(topk_ids_pred)
        if predicted_ids_cpu.numel() > 0:
            predicted_ids_cpu = torch.unique(predicted_ids_cpu, sorted=True)
            predicted_ids_cpu = self._filter_uncached_expert_ids_cpu(
                layer_idx,
                predicted_ids_cpu,
            )
            if block_load_ids:
                keep_mask = torch.tensor(
                    [
                        int(expert_id) not in block_load_ids
                        for expert_id in predicted_ids_cpu.tolist()
                    ],
                    dtype=torch.bool,
                )
                predicted_ids_cpu = predicted_ids_cpu[keep_mask]
        return predicted_ids_cpu.tolist()

    def _get_ready_mask_buffer(
        self,
        buffer_idx: int,
        num_experts: int,
        device: torch.device,
    ) -> torch.Tensor:
        ready_mask = self._ready_masks.get(buffer_idx)
        if (ready_mask is None or ready_mask.numel() != num_experts
                or ready_mask.device != device):
            ready_mask = torch.empty((num_experts, ),
                                     dtype=torch.bool,
                                     device=device)
            self._ready_masks[buffer_idx] = ready_mask
        ready_mask.zero_()
        return ready_mask

    def _get_pending_mask_buffer(
        self,
        buffer_idx: int,
        num_experts: int,
        device: torch.device,
    ) -> torch.Tensor:
        pending_mask = self._pending_masks.get(buffer_idx)
        if (pending_mask is None or pending_mask.numel() != num_experts
                or pending_mask.device != device):
            pending_mask = torch.empty((num_experts, ),
                                       dtype=torch.bool,
                                       device=device)
            self._pending_masks[buffer_idx] = pending_mask
        pending_mask.zero_()
        return pending_mask

    def _get_context_locked(self, context_id: int) -> PrefetchContext | None:
        return self.contexts.get(context_id)

    def _remove_context_id_locked(self, context_id: int) -> None:
        self._context_rr = deque(
            cid for cid in self._context_rr if cid != context_id)

    def _clear_block_load_state_locked(self, context: PrefetchContext) -> None:
        block_load = context.block_load
        block_load.layer_idx = None
        block_load.buffer_idx = None
        block_load.target_ids.clear()
        block_load.queue.clear()
        block_load.inflight.clear()
        block_load.ready_events.clear()
        block_load.error = None

    def _clear_comp_prefetch_state_locked(
        self,
        context: PrefetchContext,
        *,
        invalidate: bool = False,
    ) -> None:
        comp_prefetch = context.comp_prefetch
        if invalidate:
            comp_prefetch.version += 1
        comp_prefetch.layer_idx = None
        comp_prefetch.buffer_idx = None
        comp_prefetch.pending_request = None
        comp_prefetch.queue.clear()
        comp_prefetch.loaded_queue.clear()
        comp_prefetch.error = None

    def _clear_layer_state_locked(
        self,
        context: PrefetchContext,
        *,
        invalidate_comp: bool = False,
    ) -> None:
        self._clear_block_load_state_locked(context)
        self._clear_comp_prefetch_state_locked(
            context,
            invalidate=invalidate_comp,
        )

    def create_context(self, context_id: int) -> None:
        with self.cv:
            if context_id in self.contexts:
                raise ValueError(
                    f"Prefetch context {context_id} already exists.")
            self.contexts[context_id] = PrefetchContext(context_id=context_id)
            self._context_rr.append(context_id)
            self.cv.notify_all()

    def set_foreground_context(self, context_id: int) -> None:
        with self.cv:
            if context_id not in self.contexts:
                raise ValueError(
                    f"Missing prefetch context {context_id}.")
            self.foreground_context_id = context_id
            self.cv.notify_all()

    def clear_context(self, context_id: int) -> None:
        with self.cv:
            context = self.contexts.pop(context_id, None)
            if context is None:
                return
            self._remove_context_id_locked(context_id)
            if self.foreground_context_id == context_id:
                self.foreground_context_id = None
            self.cv.notify_all()

    def schedule_prefetch(
        self,
        context_id: int,
        layer_idx: int,
        topk_ids_pred,
        next_load_ids: set[int] | None = None,
    ) -> None:
        producer_event = None
        request_topk_ids = topk_ids_pred
        if isinstance(topk_ids_pred, torch.Tensor):
            request_topk_ids = topk_ids_pred.detach()
            if (request_topk_ids.device.type == "cuda"
                    and request_topk_ids.numel() > 0):
                producer_event = torch.cuda.Event()
                producer_event.record(
                    torch.cuda.current_stream(device=request_topk_ids.device))

        with self.cv:
            context = self._get_context_locked(context_id)
            if context is None:
                raise ValueError(
                    f"Missing prefetch context {context_id}.")
            self._clear_layer_state_locked(context, invalidate_comp=True)

            buffer_idx = self.manager.comp_flag
            block_load_ids = self._build_block_load_ids(
                layer_idx,
                set(next_load_ids or ()),
            )
            if block_load_ids:
                block_load = context.block_load
                block_load.layer_idx = layer_idx
                block_load.buffer_idx = buffer_idx
                block_load.target_ids = set(block_load_ids)
                block_load.queue = list(block_load_ids)

            comp_prefetch = context.comp_prefetch
            comp_prefetch.layer_idx = layer_idx
            comp_prefetch.buffer_idx = buffer_idx
            if self.manager.compact_comp_buffer_enabled():
                w13_weight = (
                    self.manager.w13_weight_1
                    if buffer_idx == 1 else self.manager.w13_weight_2
                )
                if w13_weight is None:
                    raise RuntimeError(
                        "Compact comp-buffer prefetch scheduled before "
                        "expert weights were initialized."
                    )
                buffer_state = self.manager._reset_comp_buffer_state(
                    buffer_idx,
                    layer_idx,
                    w13_weight.device,
                )
                comp_prefetch_ids = self._build_comp_prefetch_ids(
                    layer_idx,
                    request_topk_ids,
                    set(block_load_ids),
                )
                if len(comp_prefetch_ids) > buffer_state.slot_to_expert.numel():
                    raise RuntimeError(
                        "Predicted compact-prefetch experts exceeded comp buffer "
                        f"capacity for layer={layer_idx}, buffer={buffer_idx}: "
                        f"predicted={len(comp_prefetch_ids)}, "
                        f"capacity={buffer_state.slot_to_expert.numel()}."
                    )
                comp_prefetch.queue = []
                for slot_id, expert_id in enumerate(comp_prefetch_ids):
                    buffer_state.expert_to_slot[expert_id] = slot_id
                    buffer_state.slot_to_expert[slot_id] = expert_id
                    comp_prefetch.queue.append((expert_id, slot_id))
                buffer_state.assigned_slot_count = len(comp_prefetch_ids)
            comp_prefetch.pending_request = CompPrefetchRequest(
                context_id=context_id,
                version=comp_prefetch.version,
                layer_idx=layer_idx,
                buffer_idx=buffer_idx,
                topk_ids_pred=request_topk_ids,
                producer_event=producer_event,
            )
            self.cv.notify_all()

    def notify_layer_loading(
        self,
        context_id: int,
        layer_idx: int,
        buffer_idx: int | None = None,
        num_experts: int | None = None,
        device: torch.device | None = None,
    ):
        with self.cv:
            context = self._get_context_locked(context_id)
            if context is None:
                if (
                    buffer_idx is not None
                    and num_experts is not None
                    and device is not None
                ):
                    return self._get_ready_mask_buffer(
                        buffer_idx,
                        num_experts,
                        device,
                    ), None, self._get_pending_mask_buffer(
                        buffer_idx,
                        num_experts,
                        device,
                    ), False
                return [], None, [], False

            block_load = context.block_load
            comp_prefetch = context.comp_prefetch
            has_block_state = block_load.layer_idx == layer_idx
            has_comp_state = comp_prefetch.layer_idx == layer_idx

            if not has_block_state and not has_comp_state:
                if (
                    buffer_idx is not None
                    and num_experts is not None
                    and device is not None
                ):
                    return (
                        self._get_ready_mask_buffer(
                            buffer_idx,
                            num_experts,
                            device,
                        ),
                        None,
                        self._get_pending_mask_buffer(
                            buffer_idx,
                            num_experts,
                            device,
                        ),
                        False,
                    )
                return [], None, [], False

            pending_block_ids = sorted(block_load.target_ids) if has_block_state else []
            while pending_block_ids:
                if block_load.error is not None:
                    raise RuntimeError(block_load.error)
                all_ready = True
                for expert_id in pending_block_ids:
                    completion_event = block_load.ready_events.get(expert_id)
                    if completion_event is None:
                        if (expert_id in block_load.queue
                                or expert_id in block_load.inflight):
                            all_ready = False
                            break
                        raise RuntimeError(
                            "Missing block-load completion for "
                            f"context={context_id}, layer={layer_idx}, "
                            f"expert={expert_id}.")
                    if not completion_event.query():
                        all_ready = False
                        break
                if all_ready:
                    break
                self.cv.wait(timeout=0.0005)

            if block_load.error is not None:
                raise RuntimeError(block_load.error)
            if has_comp_state and comp_prefetch.error is not None:
                raise RuntimeError(comp_prefetch.error)

            scheduled_buffer_idx = None
            if has_block_state and block_load.buffer_idx is not None:
                scheduled_buffer_idx = block_load.buffer_idx
            elif has_comp_state:
                scheduled_buffer_idx = comp_prefetch.buffer_idx
            ready_experts: list[int] = []
            inflight_experts: list[int] = []
            handoff_event: torch.cuda.Event | None = None
            if has_comp_state:
                for expert_id, completion_event in comp_prefetch.loaded_queue:
                    if completion_event is None:
                        ready_experts.append(expert_id)
                        continue
                    handoff_event = completion_event
                    if completion_event.query():
                        ready_experts.append(expert_id)
                    else:
                        inflight_experts.append(expert_id)

            has_block_loads = has_block_state and bool(block_load.target_ids)
            if has_block_state:
                self._clear_block_load_state_locked(context)
            if has_comp_state:
                self._clear_comp_prefetch_state_locked(
                    context,
                    invalidate=True,
                )
            self.cv.notify_all()

            if buffer_idx is None or num_experts is None or device is None:
                return ready_experts

            if scheduled_buffer_idx is not None and buffer_idx != scheduled_buffer_idx:
                return (
                    self._get_ready_mask_buffer(buffer_idx, num_experts, device),
                    None,
                    self._get_pending_mask_buffer(buffer_idx, num_experts, device),
                    has_block_loads,
                )

            ready_mask = self._get_ready_mask_buffer(
                buffer_idx,
                num_experts,
                device,
            )
            if ready_experts:
                ready_ids = torch.tensor(
                    ready_experts,
                    dtype=torch.long,
                    device=device,
                )
                ready_mask.index_fill_(0, ready_ids, True)
            pending_mask = self._get_pending_mask_buffer(
                buffer_idx,
                num_experts,
                device,
            )
            if inflight_experts:
                pending_ids = torch.tensor(
                    inflight_experts,
                    dtype=torch.long,
                    device=device,
                )
                pending_mask.index_fill_(0, pending_ids, True)
            return ready_mask, handoff_event, pending_mask, has_block_loads
    
    def shutdown(self):
        shutdown_flag = getattr(self, "shutdown_flag", None)
        if shutdown_flag is not None:
            shutdown_flag.set()
        cv = getattr(self, "cv", None)
        if cv is not None:
            with cv:
                cv.notify_all()
        daemon_thread = getattr(self, "daemon_thread", None)
        if daemon_thread is not None and daemon_thread.is_alive():
            daemon_thread.join(timeout=1)

    def _activate_prefetch_request(self, request: CompPrefetchRequest) -> None:
        block_load_ids: set[int] = set()
        with self.cv:
            context = self._get_context_locked(request.context_id)
            if context is None:
                return
            comp_prefetch = context.comp_prefetch
            if (request.version != comp_prefetch.version
                    or request.layer_idx != comp_prefetch.layer_idx):
                return
            if context.block_load.layer_idx == request.layer_idx:
                block_load_ids = set(context.block_load.target_ids)
            if self.manager.compact_comp_buffer_enabled():
                if not comp_prefetch.queue and not context.block_load.target_ids:
                    self._clear_comp_prefetch_state_locked(context)
                    self.cv.notify_all()
                return
        try:
            comp_prefetch_ids = self._build_comp_prefetch_ids(
                request.layer_idx,
                request.topk_ids_pred,
                block_load_ids,
            )
        except Exception as e:
            with self.cv:
                context = self._get_context_locked(request.context_id)
                if context is not None:
                    comp_prefetch = context.comp_prefetch
                    if (request.version != comp_prefetch.version
                            or request.layer_idx != comp_prefetch.layer_idx):
                        return
                    comp_prefetch.error = (
                        f"fail to build prefetch request for context "
                        f"{request.context_id}, layer {request.layer_idx}: {e}"
                    )
                    self.cv.notify_all()
            return

        with self.cv:
            context = self._get_context_locked(request.context_id)
            if context is None:
                return
            comp_prefetch = context.comp_prefetch
            if (request.version != comp_prefetch.version
                    or request.layer_idx != comp_prefetch.layer_idx):
                return
            comp_prefetch.queue = [(expert_id, expert_id)
                                   for expert_id in comp_prefetch_ids]
            comp_prefetch.loaded_queue.clear()
            comp_prefetch.error = None
            if not comp_prefetch_ids and not context.block_load.target_ids:
                self._clear_comp_prefetch_state_locked(context)
            self.cv.notify_all()

    def _pop_context_work_locked(
        self,
        context: PrefetchContext,
    ):
        block_load = context.block_load
        comp_prefetch = context.comp_prefetch
        if block_load.queue and block_load.layer_idx is not None:
            expert_id = block_load.queue.pop(0)
            block_load.inflight.add(expert_id)
            return ("block", context.context_id, block_load.layer_idx, expert_id)
        if comp_prefetch.pending_request is not None:
            producer_event = comp_prefetch.pending_request.producer_event
            if producer_event is None or producer_event.query():
                request = comp_prefetch.pending_request
                comp_prefetch.pending_request = None
                return ("request", request)
        if comp_prefetch.queue and comp_prefetch.layer_idx is not None:
            expert_id, target_row_idx = comp_prefetch.queue.pop(0)
            return (
                "comp",
                context.context_id,
                comp_prefetch.layer_idx,
                expert_id,
                target_row_idx,
                comp_prefetch.version,
                comp_prefetch.buffer_idx,
            )
        return None

    def _pop_next_work_locked(self):
        foreground_context = None
        if self.foreground_context_id is not None:
            foreground_context = self.contexts.get(self.foreground_context_id)
        if foreground_context is not None:
            work_item = self._pop_context_work_locked(foreground_context)
            if work_item is not None:
                return work_item

        for _ in range(len(self._context_rr)):
            context_id = self._context_rr[0]
            self._context_rr.rotate(-1)
            if context_id == self.foreground_context_id:
                continue
            context = self.contexts.get(context_id)
            if context is None:
                continue
            work_item = self._pop_context_work_locked(context)
            if work_item is not None:
                return work_item
        return None

    def _prefetch_worker(self):
        if torch.cuda.is_available():
            maybe_bind_prefetch_thread_expert_numa(
                self.manager.vllm_config,
                local_cuda_index=int(torch.cuda.current_device()),
                local_rank=int(torch.cuda.current_device()),
            )
        with torch.cuda.stream(StreamContext.prefetch_stream):
            while not self.shutdown_flag.is_set():
                with self.cv:
                    work_item = None
                    while not self.shutdown_flag.is_set():
                        work_item = self._pop_next_work_locked()
                        if work_item is not None:
                            break
                        self.cv.wait(timeout=0.0005)

                if self.shutdown_flag.is_set():
                    return

                if work_item is None:
                    time.sleep(0.0005)
                    continue

                work_kind = work_item[0]
                if work_kind == "request":
                    self._activate_prefetch_request(work_item[1])
                    continue

                if work_kind == "block":
                    _kind, context_id, layer_idx, expert_id = work_item
                    self._load_expert_to_blocks(context_id, layer_idx, expert_id)
                    continue

                (_kind, context_id, layer_idx, expert_id, target_row_idx,
                 request_version, target_buffer_idx) = work_item
                self._load_expert_to_comp(
                    context_id,
                    layer_idx,
                    expert_id,
                    target_row_idx=target_row_idx,
                    request_version=request_version,
                    target_buffer_idx=target_buffer_idx,
                )

    def _load_expert_to_blocks(
        self,
        context_id: int,
        layer_idx: int,
        expert_id: int,
    ) -> bool:
        try:
            with self.cv:
                context = self._get_context_locked(context_id)
                if context is None:
                    return False
                block_load = context.block_load
                if (layer_idx != block_load.layer_idx
                        or expert_id not in block_load.inflight):
                    return False
                block_row = self.manager._reserved_block_row(layer_idx, expert_id)
                if block_row is None:
                    raise RuntimeError(
                        f"missing reserved block row for layer={layer_idx}, expert={expert_id}"
                    )

            layer_prefix = self.manager.get_layer_prefix(layer_idx)
            layer_weights = self.manager.expert_params.get(layer_prefix)
            if layer_weights is None:
                raise RuntimeError(f"missing layer weights for {layer_prefix}")

            with torch.cuda.stream(StreamContext.prefetch_stream):
                self.manager._copy_expert_to_blocks(
                    layer_weights,
                    expert_id,
                    block_row,
                )
                completion_event = torch.cuda.Event()
                completion_event.record(StreamContext.prefetch_stream)
            with self.cv:
                context = self._get_context_locked(context_id)
                if context is None:
                    return True
                block_load = context.block_load
                block_load.inflight.discard(expert_id)
                if (layer_idx == block_load.layer_idx
                        and expert_id in block_load.target_ids):
                    block_load.ready_events[expert_id] = completion_event
                self.cv.notify_all()
            return True

        except Exception as e:
            with self.cv:
                context = self._get_context_locked(context_id)
                if context is not None:
                    block_load = context.block_load
                    block_load.inflight.discard(expert_id)
                    block_load.error = (
                        f"fail to materialize block-load expert {expert_id} "
                        f"from context {context_id}, layer {layer_idx}: {e}"
                    )
                self.cv.notify_all()
            return False

    def _load_expert_to_comp(
        self,
        context_id: int,
        layer_idx: int,
        expert_id: int,
        target_row_idx: int,
        request_version: int | None = None,
        target_buffer_idx: int | None = None,
    ) -> bool:
        try:
            with self.cv:
                context = self._get_context_locked(context_id)
                if context is None:
                    return False
                comp_prefetch = context.comp_prefetch
                if layer_idx != comp_prefetch.layer_idx:
                    return False
                if (request_version is not None
                        and request_version != comp_prefetch.version):
                    return False
                if target_buffer_idx is None:
                    target_buffer_idx = comp_prefetch.buffer_idx
            layer_prefix = self.manager.get_layer_prefix(layer_idx)
            if target_buffer_idx is None:
                target_buffer_idx = self.manager.comp_flag
            w13_weight_comm = (
                self.manager.w13_weight_1
                if target_buffer_idx == 1 else self.manager.w13_weight_2
            )
            w2_weight_comm = (
                self.manager.w2_weight_1
                if target_buffer_idx == 1 else self.manager.w2_weight_2
            )

            block_row = self.manager.block_table.block_table.np[layer_idx,
                                                                expert_id]
            w1_block_id, w2_block_id, w3_block_id = (
                int(block_row[0]),
                int(block_row[1]),
                int(block_row[2]),
            )
            layer_weights = None
            if w1_block_id == -1:
                layer_weights = self.manager.expert_params.get(layer_prefix)
                if layer_weights is None:
                    return False

            with self.cv:
                context = self._get_context_locked(context_id)
                if context is None:
                    return False
                if (
                    layer_idx != context.comp_prefetch.layer_idx
                    or (
                        request_version is not None
                        and request_version != context.comp_prefetch.version
                    )
                ):
                    return False

                with torch.cuda.stream(StreamContext.prefetch_stream):
                    if w1_block_id != -1:
                        if self.manager.no_copy_compute_enabled():
                            return True
                        w1_param = self.manager.w13_blocks[w1_block_id]
                        w2_param = self.manager.w2_blocks[w2_block_id]
                        w3_param = self.manager.w13_blocks[w3_block_id]
                        intermediate_size = w1_param.shape[0]
                        w13_weight_comm[target_row_idx][:intermediate_size].copy_(
                            w1_param.data,
                            non_blocking=True,
                        )
                        w13_weight_comm[target_row_idx][intermediate_size:].copy_(
                            w3_param.data,
                            non_blocking=True,
                        )
                        w2_weight_comm[target_row_idx].copy_(w2_param.data,
                                                             non_blocking=True)
                    else:
                        assert layer_weights is not None
                        self.manager._copy_expert_to_comp_buffer(
                            layer_weights,
                            expert_id,
                            target_row_idx,
                            w13_weight_comm,
                            w2_weight_comm,
                        )

                    completion_event = torch.cuda.Event()
                    completion_event.record(StreamContext.prefetch_stream)

                context.comp_prefetch.loaded_queue.append(
                    (expert_id, completion_event))
                self.cv.notify_all()
            return True

        except Exception as e:
            with self.cv:
                context = self._get_context_locked(context_id)
                if (
                    context is not None
                    and layer_idx == context.comp_prefetch.layer_idx
                    and (
                        request_version is None
                        or request_version == context.comp_prefetch.version
                    )
                ):
                    context.comp_prefetch.error = (
                        f"fail to prefetch expert {expert_id} from "
                        f"context {context_id}, layer {layer_idx}: {e}"
                    )
                self.cv.notify_all()
            return False
