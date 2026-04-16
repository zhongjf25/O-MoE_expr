import threading
import types
import weakref

import pytest
import torch

from vllm import _custom_ops as _custom_ops  # noqa: F401
from vllm.model_executor.backend_expert_manager import (
    BackendExpertManager,
    OffloadedExpertComputeInputs,
    OffloadedExpertSelection,
    OffloadedLayerWeights,
    StreamContext,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA",
)


class _DummyModule:
    pass


class _DummyPrefetchDaemon:
    def __init__(self, loaded_experts: list[int] | None = None):
        self.loaded_experts = loaded_experts or []
        self.scheduled: list[tuple[int, int, object, object]] = []
        self.clear_calls = 0

    def create_context(self, context_id: int) -> None:
        return None

    def set_foreground_context(self, context_id: int) -> None:
        return None

    def clear_context(self, context_id: int) -> None:
        return None

    def notify_layer_loading(self,
                             context_id: int,
                             layer_idx: int,
                             buffer_idx: int | None = None,
                             num_experts: int | None = None,
                             device: torch.device | None = None):
        if buffer_idx is not None and num_experts is not None and device is not None:
            ready_mask = torch.zeros((num_experts, ),
                                     dtype=torch.bool,
                                     device=device)
            pending_mask = torch.zeros((num_experts, ),
                                       dtype=torch.bool,
                                       device=device)
            if self.loaded_experts:
                ready_ids = torch.tensor(self.loaded_experts,
                                         dtype=torch.long,
                                         device=device)
                ready_mask.index_fill_(0, ready_ids, True)
            return ready_mask, None, pending_mask, False
        return self.loaded_experts

    def schedule_prefetch(self,
                          context_id: int,
                          layer_idx: int,
                          topk_ids_pred,
                          next_load_ids=None):
        self.scheduled.append((context_id, layer_idx, topk_ids_pred, next_load_ids))

    def shutdown(self):
        return None


class _DelayedWritePrefetchDaemon(_DummyPrefetchDaemon):
    def __init__(
        self,
        manager: BackendExpertManager,
        *,
        expert_id: int,
        w13_fill: float,
        w2_fill: float,
        sleep_cycles: int = 20_000_000,
    ):
        super().__init__([])
        self.manager = manager
        self.expert_id = expert_id
        self.w13_fill = w13_fill
        self.w2_fill = w2_fill
        self.sleep_cycles = sleep_cycles

    def notify_layer_loading(self,
                             context_id: int,
                             layer_idx: int,
                             buffer_idx: int | None = None,
                             num_experts: int | None = None,
                             device: torch.device | None = None):
        if buffer_idx is None or num_experts is None or device is None:
            return []

        ready_mask = torch.zeros((num_experts, ),
                                 dtype=torch.bool,
                                 device=device)
        pending_mask = torch.zeros((num_experts, ),
                                   dtype=torch.bool,
                                   device=device)
        w13_weight_comp = (
            self.manager.w13_weight_1
            if buffer_idx == 1 else self.manager.w13_weight_2
        )
        w2_weight_comp = (
            self.manager.w2_weight_1
            if buffer_idx == 1 else self.manager.w2_weight_2
        )
        with torch.cuda.stream(StreamContext.prefetch_stream):
            torch.cuda._sleep(self.sleep_cycles)
            w13_weight_comp[self.expert_id].fill_(self.w13_fill)
            w2_weight_comp[self.expert_id].fill_(self.w2_fill)
            handoff_event = torch.cuda.Event()
            handoff_event.record(StreamContext.prefetch_stream)
        return ready_mask, handoff_event, pending_mask, False


class _DummyBlockTable:
    def __init__(self, gpu_table: torch.Tensor):
        self.block_table = types.SimpleNamespace(
            gpu=gpu_table,
            cpu=gpu_table.cpu().clone(),
            np=gpu_table.cpu().numpy().copy(),
            copy_to_gpu=lambda *args, **kwargs: None,
        )

    def commit_expert_row(self,
                          layer_id: int,
                          expert_id: int,
                          non_blocking: bool = True):
        self.block_table.gpu[layer_id, expert_id].copy_(
            self.block_table.cpu[layer_id, expert_id],
            non_blocking=non_blocking,
        )

    def commit_all_experts(self, non_blocking: bool = True):
        self.block_table.gpu.copy_(self.block_table.cpu, non_blocking=non_blocking)


def _make_layer_weights(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    *,
    dtype: torch.dtype,
) -> OffloadedLayerWeights:
    w13_cpu = torch.randn(
        num_experts,
        2 * intermediate_size,
        hidden_size,
        dtype=dtype,
    )
    w2_cpu = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        dtype=dtype,
    )
    return OffloadedLayerWeights(
        w13_cpu=w13_cpu,
        w2_cpu=w2_cpu,
        intermediate_size=intermediate_size,
    )


def _packed_w13(layer_weights: OffloadedLayerWeights, expert_id: int) -> torch.Tensor:
    return layer_weights.w13_cpu[expert_id]


def _preload_comp_buffer(
    w13_weight_comp: torch.Tensor,
    w2_weight_comp: torch.Tensor,
    layer_weights: OffloadedLayerWeights,
    expert_id: int,
    *,
    row_idx: int | None = None,
) -> None:
    if row_idx is None:
        row_idx = expert_id
    w13_weight_comp[row_idx].copy_(_packed_w13(layer_weights, expert_id).cuda())
    w2_weight_comp[row_idx].copy_(layer_weights.w2_cpu[expert_id].cuda())


def _populate_cached_expert(
    manager: BackendExpertManager,
    gpu_table: torch.Tensor,
    layer_weights: OffloadedLayerWeights,
    expert_id: int,
    *,
    w1_block_id: int,
    w2_block_id: int,
    w3_block_id: int,
) -> None:
    manager.w13_blocks[w1_block_id].copy_(
        layer_weights.w13_cpu[expert_id, :layer_weights.intermediate_size].cuda())
    manager.w2_blocks[w2_block_id].copy_(layer_weights.w2_cpu[expert_id].cuda())
    manager.w13_blocks[w3_block_id].copy_(
        layer_weights.w13_cpu[expert_id, layer_weights.intermediate_size:].cuda())
    gpu_table[0, expert_id] = torch.tensor(
        [w1_block_id, w2_block_id, w3_block_id],
        dtype=torch.int32,
        device="cuda",
    )


def _make_manager(
    *,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    sentinel: float,
    dtype: torch.dtype = torch.bfloat16,
    offload_expert_limit: int = 0,
):
    StreamContext.init()
    layer_prefix = "model.layers.0.mlp.experts"
    compact_capacity = (
        offload_expert_limit
        if 0 < offload_expert_limit < num_experts
        else num_experts
    )

    manager = object.__new__(BackendExpertManager)
    manager.first_k_dense_replace = 0
    manager.num_hidden_layers = 1
    manager.num_experts = num_experts
    manager.layer_prefixes = [layer_prefix]
    manager.layer_prefix_to_idx = {layer_prefix: 0}
    manager.layer_idx_to_prefix = {0: layer_prefix}
    manager.pending_topk_updates = {}
    manager.pending_topk_lock = threading.Lock()
    manager._load_targets = {}
    manager._evict_targets = {}
    manager._pending_reserved_blocks = {}
    manager._active_cache_delta_id = None
    manager._active_prefetch_context_id = 1
    manager.prefetch_daemon = _DummyPrefetchDaemon([])
    manager.expert_offload_config = types.SimpleNamespace(
        expert_no_copy_compute=True,
        offload_expert_limit=offload_expert_limit,
    )
    manager.comp_flag = 1
    manager._selection_masks = {}
    manager._offloaded_prepare_states = {}
    manager._comp_buffer_states = {}
    manager.w13_weight_1 = torch.full(
        (compact_capacity, 2 * intermediate_size, hidden_size),
        sentinel,
        dtype=dtype,
        device="cuda",
    )
    manager.w2_weight_1 = torch.full(
        (compact_capacity, hidden_size, intermediate_size),
        sentinel,
        dtype=dtype,
        device="cuda",
    )
    manager.w13_weight_2 = torch.full_like(manager.w13_weight_1, sentinel)
    manager.w2_weight_2 = torch.full_like(manager.w2_weight_1, sentinel)
    manager.w13_blocks = torch.full(
        (max(1, num_experts * 2), intermediate_size, hidden_size),
        sentinel,
        dtype=dtype,
        device="cuda",
    )
    manager.w2_blocks = torch.full(
        (max(1, num_experts), hidden_size, intermediate_size),
        sentinel,
        dtype=dtype,
        device="cuda",
    )

    layer_weights = _make_layer_weights(
        num_experts,
        hidden_size,
        intermediate_size,
        dtype=dtype,
    )
    manager.expert_params = {layer_prefix: layer_weights}

    gpu_table = torch.full(
        (1, num_experts, 3),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    manager.block_table = _DummyBlockTable(gpu_table)

    module = _DummyModule()
    manager.moe_modules = {layer_prefix: weakref.ref(module)}
    manager._test_module = module
    return manager, layer_prefix, layer_weights, gpu_table


@torch.inference_mode()
def test_get_offloaded_compute_inputs_mixed_sources_without_cpu_roundtrip():
    num_experts = 4
    hidden_size = 8
    intermediate_size = 8
    sentinel = -123.0

    manager, layer_prefix, layer_weights, gpu_table = _make_manager(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        sentinel=sentinel,
    )

    manager.prefetch_daemon.loaded_experts = [0]
    _preload_comp_buffer(
        manager.w13_weight_1,
        manager.w2_weight_1,
        layer_weights,
        0,
    )
    _populate_cached_expert(
        manager,
        gpu_table,
        layer_weights,
        1,
        w1_block_id=0,
        w2_block_id=0,
        w3_block_id=1,
    )

    topk_ids = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32, device="cuda")
    inputs = manager.get_offloaded_compute_inputs(layer_prefix, topk_ids, [])

    assert inputs is not None
    assert inputs.expert_source[0].item() == OffloadedExpertComputeInputs.SOURCE_COMP
    assert inputs.expert_source[1].item() == OffloadedExpertComputeInputs.SOURCE_CACHE
    assert inputs.expert_source[2].item() == OffloadedExpertComputeInputs.SOURCE_COMP
    assert inputs.cache_w1_block_ids[1].item() == 0
    assert inputs.cache_w2_block_ids[1].item() == 0
    assert inputs.cache_w3_block_ids[1].item() == 1

    torch.testing.assert_close(inputs.w13_weight_comp[0],
                               _packed_w13(layer_weights, 0).cuda())
    torch.testing.assert_close(inputs.w2_weight_comp[0],
                               layer_weights.w2_cpu[0].cuda())
    torch.testing.assert_close(
        inputs.w13_weight_comp[1],
        torch.full_like(inputs.w13_weight_comp[1], sentinel),
    )
    torch.testing.assert_close(
        inputs.w2_weight_comp[1],
        torch.full_like(inputs.w2_weight_comp[1], sentinel),
    )
    torch.testing.assert_close(inputs.w13_weight_comp[2],
                               _packed_w13(layer_weights, 2).cuda())
    torch.testing.assert_close(inputs.w2_weight_comp[2],
                               layer_weights.w2_cpu[2].cuda())


@torch.inference_mode()
def test_get_offloaded_compute_inputs_compact_buffer_uses_slot_mapping():
    sentinel = -123.0
    manager, layer_prefix, layer_weights, _gpu_table = _make_manager(
        num_experts=4,
        hidden_size=8,
        intermediate_size=8,
        sentinel=sentinel,
        offload_expert_limit=2,
    )

    manager.prefetch_daemon.loaded_experts = [3]
    buffer_state = manager._get_comp_buffer_state(1, manager.w13_weight_1.device)
    buffer_state.expert_to_slot[3] = 0
    buffer_state.slot_to_expert[0] = 3
    buffer_state.assigned_slot_count = 1
    _preload_comp_buffer(
        manager.w13_weight_1,
        manager.w2_weight_1,
        layer_weights,
        3,
        row_idx=0,
    )

    topk_ids = torch.tensor([[1, 3]], dtype=torch.int32, device="cuda")
    inputs = manager.get_offloaded_compute_inputs(layer_prefix, topk_ids, [])

    assert inputs is not None
    assert inputs.w13_weight_comp.size(0) == 2
    assert inputs.expert_source[3].item() == OffloadedExpertComputeInputs.SOURCE_COMP
    assert inputs.expert_source[1].item() == OffloadedExpertComputeInputs.SOURCE_COMP
    assert inputs.comp_expert_to_slot[3].item() == 0
    assert inputs.comp_expert_to_slot[1].item() == 1
    torch.testing.assert_close(inputs.w13_weight_comp[0],
                               _packed_w13(layer_weights, 3).cuda())
    torch.testing.assert_close(inputs.w2_weight_comp[0],
                               layer_weights.w2_cpu[3].cuda())
    torch.testing.assert_close(inputs.w13_weight_comp[1],
                               _packed_w13(layer_weights, 1).cuda())
    torch.testing.assert_close(inputs.w2_weight_comp[1],
                               layer_weights.w2_cpu[1].cuda())


@torch.inference_mode()
def test_get_offloaded_compute_inputs_waits_for_handoff_event_before_reusing_comp_buffer():
    sentinel = -123.0
    manager, layer_prefix, layer_weights, _gpu_table = _make_manager(
        num_experts=4,
        hidden_size=8,
        intermediate_size=8,
        sentinel=sentinel,
    )
    manager.prefetch_daemon = _DelayedWritePrefetchDaemon(
        manager,
        expert_id=2,
        w13_fill=77.0,
        w2_fill=55.0,
    )

    inputs = manager.get_offloaded_compute_inputs(
        layer_prefix,
        torch.tensor([[2]], dtype=torch.int32, device="cuda"),
        [],
    )
    torch.cuda.synchronize()

    assert inputs is not None
    assert inputs.expert_source[2].item() == OffloadedExpertComputeInputs.SOURCE_COMP
    torch.testing.assert_close(inputs.w13_weight_comp[2],
                               _packed_w13(layer_weights, 2).cuda())
    torch.testing.assert_close(inputs.w2_weight_comp[2],
                               layer_weights.w2_cpu[2].cuda())


@torch.inference_mode()
def test_get_offloaded_compute_inputs_reuses_metadata_buffers_without_fill():
    sentinel = -123.0
    manager, layer_prefix, layer_weights, gpu_table = _make_manager(
        num_experts=5,
        hidden_size=8,
        intermediate_size=8,
        sentinel=sentinel,
    )

    _populate_cached_expert(
        manager,
        gpu_table,
        layer_weights,
        0,
        w1_block_id=0,
        w2_block_id=0,
        w3_block_id=1,
    )
    _populate_cached_expert(
        manager,
        gpu_table,
        layer_weights,
        3,
        w1_block_id=2,
        w2_block_id=1,
        w3_block_id=3,
    )

    first = manager.get_offloaded_compute_inputs(
        layer_prefix,
        torch.tensor([[0, 1]], dtype=torch.int32, device="cuda"),
        [],
    )
    assert first is not None

    second = manager.get_offloaded_compute_inputs(
        layer_prefix,
        torch.tensor([[1, 4]], dtype=torch.int32, device="cuda"),
        [],
    )
    assert second is not None

    manager.prefetch_daemon.loaded_experts = [2]
    _preload_comp_buffer(
        manager.w13_weight_1,
        manager.w2_weight_1,
        layer_weights,
        2,
    )
    third = manager.get_offloaded_compute_inputs(
        layer_prefix,
        torch.tensor([[2, 3]], dtype=torch.int32, device="cuda"),
        [],
    )

    assert third is not None
    assert first.expert_source.data_ptr() == third.expert_source.data_ptr()
    assert first.cache_w1_block_ids.data_ptr() == third.cache_w1_block_ids.data_ptr()
    assert third.expert_source[2].item() == OffloadedExpertComputeInputs.SOURCE_COMP
    assert third.expert_source[3].item() == OffloadedExpertComputeInputs.SOURCE_CACHE
    assert third.cache_w1_block_ids[3].item() == 2
    assert third.cache_w2_block_ids[3].item() == 1
    assert third.cache_w3_block_ids[3].item() == 3
    torch.testing.assert_close(third.w13_weight_comp[2],
                               _packed_w13(layer_weights, 2).cuda())
    torch.testing.assert_close(third.w2_weight_comp[2],
                               layer_weights.w2_cpu[2].cuda())
    torch.testing.assert_close(
        third.w13_weight_comp[3],
        torch.full_like(third.w13_weight_comp[3], sentinel),
    )
    torch.testing.assert_close(
        third.w2_weight_comp[3],
        torch.full_like(third.w2_weight_comp[3], sentinel),
    )


@torch.inference_mode()
def test_get_offloaded_compute_inputs_overwrites_cached_metadata_for_prefetched_expert():
    sentinel = -123.0
    manager, layer_prefix, layer_weights, gpu_table = _make_manager(
        num_experts=4,
        hidden_size=8,
        intermediate_size=8,
        sentinel=sentinel,
    )

    _populate_cached_expert(
        manager,
        gpu_table,
        layer_weights,
        1,
        w1_block_id=0,
        w2_block_id=0,
        w3_block_id=1,
    )

    first = manager.get_offloaded_compute_inputs(
        layer_prefix,
        torch.tensor([[1]], dtype=torch.int32, device="cuda"),
        [],
    )
    assert first is not None
    assert first.expert_source[1].item() == OffloadedExpertComputeInputs.SOURCE_CACHE
    assert first.cache_w1_block_ids[1].item() == 0
    assert first.cache_w2_block_ids[1].item() == 0
    assert first.cache_w3_block_ids[1].item() == 1

    manager.prefetch_daemon.loaded_experts = [1]
    _preload_comp_buffer(
        manager.w13_weight_2,
        manager.w2_weight_2,
        layer_weights,
        1,
    )
    second = manager.get_offloaded_compute_inputs(
        layer_prefix,
        torch.tensor([[1]], dtype=torch.int32, device="cuda"),
        [],
    )

    assert second is not None
    assert second.expert_source[1].item() == OffloadedExpertComputeInputs.SOURCE_COMP
    torch.testing.assert_close(second.w13_weight_comp[1],
                               _packed_w13(layer_weights, 1).cuda())
    torch.testing.assert_close(second.w2_weight_comp[1],
                               layer_weights.w2_cpu[1].cuda())

    _preload_comp_buffer(
        manager.w13_weight_1,
        manager.w2_weight_1,
        layer_weights,
        1,
    )
    third = manager.get_offloaded_compute_inputs(
        layer_prefix,
        torch.tensor([[1]], dtype=torch.int32, device="cuda"),
        [],
    )

    assert third is not None
    assert first.expert_source.data_ptr() == third.expert_source.data_ptr()
    assert third.expert_source[1].item() == OffloadedExpertComputeInputs.SOURCE_COMP


@torch.inference_mode()
def test_copy_uncached_experts_to_comp_buffer_copies_only_selected_experts():
    sentinel = -123.0
    manager, _layer_prefix, layer_weights, _gpu_table = _make_manager(
        num_experts=4,
        hidden_size=8,
        intermediate_size=8,
        sentinel=sentinel,
    )

    selection = OffloadedExpertSelection(
        routed_expert_ids=torch.tensor([2, 0], dtype=torch.long, device="cuda"),
        cached_expert_ids=torch.empty((0, ), dtype=torch.long, device="cuda"),
        uncached_expert_ids=torch.tensor([2, 0], dtype=torch.long, device="cuda"),
        cached_block_rows=torch.empty((0, 3), dtype=torch.int32, device="cuda"),
    )

    manager._copy_uncached_experts_to_comp_buffer(
        layer_weights,
        selection,
        manager.w13_weight_1,
        manager.w2_weight_1,
    )

    torch.testing.assert_close(manager.w13_weight_1[0],
                               _packed_w13(layer_weights, 0).cuda())
    torch.testing.assert_close(manager.w2_weight_1[0],
                               layer_weights.w2_cpu[0].cuda())
    torch.testing.assert_close(manager.w13_weight_1[2],
                               _packed_w13(layer_weights, 2).cuda())
    torch.testing.assert_close(manager.w2_weight_1[2],
                               layer_weights.w2_cpu[2].cuda())
    torch.testing.assert_close(
        manager.w13_weight_1[1],
        torch.full_like(manager.w13_weight_1[1], sentinel),
    )
    torch.testing.assert_close(
        manager.w2_weight_1[1],
        torch.full_like(manager.w2_weight_1[1], sentinel),
    )
    torch.testing.assert_close(
        manager.w13_weight_1[3],
        torch.full_like(manager.w13_weight_1[3], sentinel),
    )
    torch.testing.assert_close(
        manager.w2_weight_1[3],
        torch.full_like(manager.w2_weight_1[3], sentinel),
    )


@torch.inference_mode()
def test_copy_uncached_experts_to_comp_op_handles_unsorted_expert_ids():
    num_experts = 4
    hidden_size = 8
    intermediate_size = 8
    sentinel = -123.0
    layer_weights = _make_layer_weights(
        num_experts,
        hidden_size,
        intermediate_size,
        dtype=torch.bfloat16,
    )

    w13_weight_comp = torch.full(
        (num_experts, 2 * intermediate_size, hidden_size),
        sentinel,
        dtype=torch.bfloat16,
        device="cuda",
    )
    w2_weight_comp = torch.full(
        (num_experts, hidden_size, intermediate_size),
        sentinel,
        dtype=torch.bfloat16,
        device="cuda",
    )
    expert_ids = torch.tensor([3, 1], dtype=torch.long, device="cuda")

    assert layer_weights.w13_uva is not None
    assert layer_weights.w2_uva is not None
    torch.ops._C.copy_uncached_experts_to_comp(
        w13_weight_comp,
        w2_weight_comp,
        layer_weights.w13_uva,
        layer_weights.w2_uva,
        expert_ids,
    )

    torch.testing.assert_close(w13_weight_comp[1],
                               _packed_w13(layer_weights, 1).cuda())
    torch.testing.assert_close(w2_weight_comp[1],
                               layer_weights.w2_cpu[1].cuda())
    torch.testing.assert_close(w13_weight_comp[3],
                               _packed_w13(layer_weights, 3).cuda())
    torch.testing.assert_close(w2_weight_comp[3],
                               layer_weights.w2_cpu[3].cuda())
    torch.testing.assert_close(
        w13_weight_comp[0],
        torch.full_like(w13_weight_comp[0], sentinel),
    )
    torch.testing.assert_close(
        w2_weight_comp[0],
        torch.full_like(w2_weight_comp[0], sentinel),
    )
    torch.testing.assert_close(
        w13_weight_comp[2],
        torch.full_like(w13_weight_comp[2], sentinel),
    )
    torch.testing.assert_close(
        w2_weight_comp[2],
        torch.full_like(w2_weight_comp[2], sentinel),
    )


@torch.inference_mode()
def test_copy_uncached_experts_to_comp_op_skips_negative_sentinel_ids():
    num_experts = 4
    hidden_size = 8
    intermediate_size = 8
    sentinel = -123.0
    layer_weights = _make_layer_weights(
        num_experts,
        hidden_size,
        intermediate_size,
        dtype=torch.bfloat16,
    )

    w13_weight_comp = torch.full(
        (num_experts, 2 * intermediate_size, hidden_size),
        sentinel,
        dtype=torch.bfloat16,
        device="cuda",
    )
    w2_weight_comp = torch.full(
        (num_experts, hidden_size, intermediate_size),
        sentinel,
        dtype=torch.bfloat16,
        device="cuda",
    )
    expert_ids = torch.tensor([3, -1, 1, -1], dtype=torch.long, device="cuda")

    assert layer_weights.w13_uva is not None
    assert layer_weights.w2_uva is not None
    torch.ops._C.copy_uncached_experts_to_comp(
        w13_weight_comp,
        w2_weight_comp,
        layer_weights.w13_uva,
        layer_weights.w2_uva,
        expert_ids,
    )

    torch.testing.assert_close(w13_weight_comp[1],
                               _packed_w13(layer_weights, 1).cuda())
    torch.testing.assert_close(w2_weight_comp[1],
                               layer_weights.w2_cpu[1].cuda())
    torch.testing.assert_close(w13_weight_comp[3],
                               _packed_w13(layer_weights, 3).cuda())
    torch.testing.assert_close(w2_weight_comp[3],
                               layer_weights.w2_cpu[3].cuda())
    torch.testing.assert_close(
        w13_weight_comp[0],
        torch.full_like(w13_weight_comp[0], sentinel),
    )
    torch.testing.assert_close(
        w2_weight_comp[0],
        torch.full_like(w2_weight_comp[0], sentinel),
    )
    torch.testing.assert_close(
        w13_weight_comp[2],
        torch.full_like(w13_weight_comp[2], sentinel),
    )
    torch.testing.assert_close(
        w2_weight_comp[2],
        torch.full_like(w2_weight_comp[2], sentinel),
    )


@torch.inference_mode()
def test_prepare_offloaded_compute_inputs_builds_metadata_and_miss_ids():
    sentinel = -123.0
    manager, _layer_prefix, layer_weights, gpu_table = _make_manager(
        num_experts=5,
        hidden_size=8,
        intermediate_size=8,
        sentinel=sentinel,
    )

    _populate_cached_expert(
        manager,
        gpu_table,
        layer_weights,
        1,
        w1_block_id=0,
        w2_block_id=0,
        w3_block_id=1,
    )

    compute_inputs = manager._build_offloaded_compute_inputs(
        1,
        manager.w13_weight_1,
        manager.w2_weight_1,
    )
    ready_mask = torch.zeros((5, ), dtype=torch.bool, device="cuda")
    ready_mask[0] = True
    miss_ids = manager._prepare_offloaded_compute_inputs(
        1,
        compute_inputs,
        0,
        torch.tensor([[0, 1], [2, 1]], dtype=torch.int32, device="cuda"),
        ready_mask,
    )
    miss_count = manager._offloaded_prepare_states[1].miss_count.cpu().item()

    assert compute_inputs.expert_source[0].item() == OffloadedExpertComputeInputs.SOURCE_COMP
    assert compute_inputs.expert_source[1].item() == OffloadedExpertComputeInputs.SOURCE_CACHE
    assert compute_inputs.expert_source[2].item() == OffloadedExpertComputeInputs.SOURCE_COMP
    assert compute_inputs.cache_w1_block_ids[1].item() == 0
    assert compute_inputs.cache_w2_block_ids[1].item() == 0
    assert compute_inputs.cache_w3_block_ids[1].item() == 1
    assert miss_count == 1
    assert torch.equal(
        torch.sort(miss_ids[:miss_count]).values.cpu(),
        torch.tensor([2], dtype=torch.long),
    )


@torch.inference_mode()
def test_prepare_and_copy_offloaded_compute_inputs_handles_mixed_sources():
    sentinel = -123.0
    manager, _layer_prefix, layer_weights, gpu_table = _make_manager(
        num_experts=5,
        hidden_size=8,
        intermediate_size=8,
        sentinel=sentinel,
    )
    _preload_comp_buffer(
        manager.w13_weight_1,
        manager.w2_weight_1,
        layer_weights,
        0,
    )
    _populate_cached_expert(
        manager,
        gpu_table,
        layer_weights,
        1,
        w1_block_id=0,
        w2_block_id=0,
        w3_block_id=1,
    )

    compute_inputs = manager._build_offloaded_compute_inputs(
        1,
        manager.w13_weight_1,
        manager.w2_weight_1,
    )
    topk_ids = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32, device="cuda")
    ready_mask = torch.zeros((5, ), dtype=torch.bool, device="cuda")
    ready_mask[0] = True
    miss_ids = manager._prepare_and_copy_offloaded_compute_inputs(
        1,
        compute_inputs,
        layer_weights,
        0,
        topk_ids,
        ready_mask,
    )
    miss_count = manager._offloaded_prepare_states[1].miss_count.cpu().item()

    active_miss_ids = miss_ids[:miss_count]
    assert compute_inputs.expert_source[0].item() == OffloadedExpertComputeInputs.SOURCE_COMP
    assert compute_inputs.expert_source[1].item() == OffloadedExpertComputeInputs.SOURCE_CACHE
    assert compute_inputs.expert_source[2].item() == OffloadedExpertComputeInputs.SOURCE_COMP
    assert compute_inputs.cache_w1_block_ids[1].item() == 0
    assert compute_inputs.cache_w2_block_ids[1].item() == 0
    assert compute_inputs.cache_w3_block_ids[1].item() == 1
    assert miss_count == 1
    assert torch.equal(torch.sort(active_miss_ids).values.cpu(),
                       torch.tensor([2], dtype=torch.long))
    torch.testing.assert_close(compute_inputs.w13_weight_comp[0],
                               _packed_w13(layer_weights, 0).cuda())
    torch.testing.assert_close(compute_inputs.w2_weight_comp[0],
                               layer_weights.w2_cpu[0].cuda())
    torch.testing.assert_close(
        compute_inputs.w13_weight_comp[1],
        torch.full_like(compute_inputs.w13_weight_comp[1], sentinel),
    )
    torch.testing.assert_close(
        compute_inputs.w2_weight_comp[1],
        torch.full_like(compute_inputs.w2_weight_comp[1], sentinel),
    )
    torch.testing.assert_close(compute_inputs.w13_weight_comp[2],
                               _packed_w13(layer_weights, 2).cuda())
    torch.testing.assert_close(compute_inputs.w2_weight_comp[2],
                               layer_weights.w2_cpu[2].cuda())


@torch.inference_mode()
def test_prepare_and_copy_offloaded_compute_inputs_grows_and_reuses_miss_scratch():
    sentinel = -123.0
    manager, _layer_prefix, layer_weights, _gpu_table = _make_manager(
        num_experts=6,
        hidden_size=8,
        intermediate_size=8,
        sentinel=sentinel,
    )
    compute_inputs = manager._build_offloaded_compute_inputs(
        1,
        manager.w13_weight_1,
        manager.w2_weight_1,
    )
    ready_mask = torch.empty((0, ), dtype=torch.bool, device="cuda")

    manager._prepare_and_copy_offloaded_compute_inputs(
        1,
        compute_inputs,
        layer_weights,
        0,
        torch.tensor([[0, 1]], dtype=torch.int32, device="cuda"),
        ready_mask,
    )
    first_state = manager._offloaded_prepare_states[1]
    first_capacity = first_state.miss_expert_ids.numel()

    manager._prepare_and_copy_offloaded_compute_inputs(
        1,
        compute_inputs,
        layer_weights,
        0,
        torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32, device="cuda"),
        ready_mask,
    )
    second_state = manager._offloaded_prepare_states[1]
    second_capacity = second_state.miss_expert_ids.numel()
    second_ptr = second_state.miss_expert_ids.data_ptr()

    manager._prepare_and_copy_offloaded_compute_inputs(
        1,
        compute_inputs,
        layer_weights,
        0,
        torch.tensor([[1]], dtype=torch.int32, device="cuda"),
        ready_mask,
    )
    third_state = manager._offloaded_prepare_states[1]

    assert first_capacity == 2
    assert second_capacity == 6
    assert second_capacity > first_capacity
    assert third_state.miss_expert_ids.numel() == second_capacity
    assert third_state.miss_expert_ids.data_ptr() == second_ptr
