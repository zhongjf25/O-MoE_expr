import threading
import types
from collections import deque

import numpy as np
import pytest
import torch

from vllm.model_executor.backend_expert_manager import (
    BackendExpertManager,
    BlockLoadState,
    CompPrefetchState,
    OffloadedExpertComputeInputs,
    OffloadedLayerWeights,
    PrefetchContext,
    PrefetchDaemon,
    StreamContext,
)
from vllm.v1.core.sched.output import ExpertCacheDelta

class _DummyPrefetchDaemon:
    def __init__(self,
                 loaded_experts: list[int],
                 manager: BackendExpertManager | None = None):
        self.loaded_experts = loaded_experts
        self.manager = manager
        self.scheduled: list[tuple[int, int, list[int], list[int]]] = []
        self.clear_calls = 0
        self.created_contexts: set[int] = set()
        self.foreground_context_id: int | None = None

    def create_context(self, context_id: int) -> None:
        self.created_contexts.add(context_id)

    def set_foreground_context(self, context_id: int) -> None:
        self.foreground_context_id = context_id

    def clear_context(self, context_id: int) -> None:
        self.created_contexts.discard(context_id)
        if self.foreground_context_id == context_id:
            self.foreground_context_id = None

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
            if self.loaded_experts:
                ready_ids = torch.tensor(self.loaded_experts,
                                         dtype=torch.long,
                                         device=device)
                ready_mask.index_fill_(0, ready_ids, True)
            return ready_mask, None, False
        return self.loaded_experts

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
        return torch.tensor(list(topk_ids), dtype=torch.long)

    def _filter_uncached_expert_ids_cpu(self, layer_idx: int,
                                        expert_ids_cpu: torch.Tensor) -> torch.Tensor:
        if expert_ids_cpu.numel() == 0 or self.manager is None:
            return expert_ids_cpu
        block_rows = self.manager.block_table.block_table.np[layer_idx,
                                                             expert_ids_cpu.numpy()]
        uncached_ids = expert_ids_cpu[torch.from_numpy(block_rows[:, 0] == -1)]
        pending_delta_ids = self.manager._load_targets.get(layer_idx, set())
        if not pending_delta_ids:
            return uncached_ids
        keep_mask = torch.tensor(
            [int(expert_id) not in pending_delta_ids for expert_id in uncached_ids.tolist()],
            dtype=torch.bool,
        )
        return uncached_ids[keep_mask]

    def schedule_prefetch(self,
                          context_id: int,
                          layer_idx: int,
                          topk_ids_pred,
                          next_load_ids=None):
        predicted_ids_cpu = self._topk_ids_to_cpu_tensor(topk_ids_pred)
        if predicted_ids_cpu.numel() > 0:
            predicted_ids_cpu = torch.unique(predicted_ids_cpu, sorted=True)
            predicted_ids_cpu = self._filter_uncached_expert_ids_cpu(
                layer_idx,
                predicted_ids_cpu,
            )
        block_load_ids = sorted(next_load_ids or [])
        if block_load_ids:
            keep_mask = torch.tensor(
                [int(expert_id) not in set(block_load_ids)
                 for expert_id in predicted_ids_cpu.tolist()],
                dtype=torch.bool,
            )
            predicted_ids_cpu = predicted_ids_cpu[keep_mask]
        comp_prefetch_ids = predicted_ids_cpu.tolist()

        if not block_load_ids and not comp_prefetch_ids:
            self.clear_calls += 1
            return

        self.scheduled.append(
            (context_id, layer_idx, block_load_ids, comp_prefetch_ids))

    def shutdown(self):
        return None


class _DummyBlockTable:
    def __init__(self, gpu_table: torch.Tensor):
        cpu_table = gpu_table.cpu().clone()
        self.block_table = types.SimpleNamespace(
            gpu=gpu_table,
            cpu=cpu_table,
            np=cpu_table.numpy(),
            copy_to_gpu=lambda *args, **kwargs: None,
        )
        self.commit_all_calls = 0
        self.commit_row_calls: list[tuple[int, int]] = []

    def get_device_block_id(self, layer_id: int, expert_id: int):
        w1, w2, w3 = self.block_table.gpu[layer_id, expert_id]
        return int(w1.item()), int(w2.item()), int(w3.item())

    def commit_expert_layer(self,
                            layer_id: int,
                            layer_mapping: dict[tuple[int, int, str], int] | None = None,
                            non_blocking: bool = True):
        layer_cpu = self.block_table.cpu[layer_id]
        if layer_mapping is not None:
            layer_cpu.fill_(-1)
            for (mapped_layer_id, expert_id, w123), block_id in layer_mapping.items():
                if mapped_layer_id != layer_id:
                    continue
                col = {"w1": 0, "w2": 1, "w3": 2}[w123]
                layer_cpu[expert_id, col] = block_id
        self.block_table.gpu[layer_id].copy_(layer_cpu, non_blocking=non_blocking)

    def commit_expert_row(self,
                          layer_id: int,
                          expert_id: int,
                          non_blocking: bool = True):
        self.commit_row_calls.append((layer_id, expert_id))
        self.block_table.gpu[layer_id, expert_id].copy_(
            self.block_table.cpu[layer_id, expert_id],
            non_blocking=non_blocking,
        )

    def commit_all_experts(self, non_blocking: bool = True):
        self.commit_all_calls += 1
        self.block_table.gpu.copy_(self.block_table.cpu, non_blocking=non_blocking)


def _make_prefetch_manager(
    gpu_table: torch.Tensor,
    *,
    num_hidden_layers: int = 2,
    first_k_dense_replace: int = 0,
):
    manager = object.__new__(BackendExpertManager)
    manager.first_k_dense_replace = first_k_dense_replace
    manager.num_hidden_layers = num_hidden_layers
    manager._load_targets = {}
    manager._evict_targets = {}
    manager._pending_reserved_blocks = {}
    manager._active_cache_delta_id = None
    manager._active_prefetch_context_id = 17
    manager.comp_flag = 1
    manager.prefetch_daemon = _DummyPrefetchDaemon([], manager)
    manager.block_table = _DummyBlockTable(gpu_table)
    return manager


def _make_dynamic_backend_manager(
    *,
    num_experts: int = 3,
    hidden_size: int = 4,
    intermediate_size: int = 2,
    sentinel: float = -123.0,
):
    StreamContext.init()
    layer_prefix = "layer"
    manager = object.__new__(BackendExpertManager)
    manager.first_k_dense_replace = 0
    manager.num_hidden_layers = 1
    manager.layer_prefixes = [layer_prefix]
    manager.layer_prefix_to_idx = {layer_prefix: 0}
    manager.layer_idx_to_prefix = {0: layer_prefix}
    manager.pending_topk_updates = {}
    manager.pending_topk_lock = threading.Lock()
    manager._load_targets = {}
    manager._evict_targets = {}
    manager._pending_reserved_blocks = {}
    manager._active_cache_delta_id = None
    manager._active_prefetch_context_id = 11
    manager.comp_flag = 1
    manager.w13_weight_1 = torch.full(
        (num_experts, 2 * intermediate_size, hidden_size),
        sentinel,
        dtype=torch.bfloat16,
        device="cuda",
    )
    manager.w2_weight_1 = torch.full(
        (num_experts, hidden_size, intermediate_size),
        sentinel,
        dtype=torch.bfloat16,
        device="cuda",
    )
    manager.w13_weight_2 = torch.full_like(manager.w13_weight_1, sentinel)
    manager.w2_weight_2 = torch.full_like(manager.w2_weight_1, sentinel)
    manager.w13_blocks = torch.full(
        (8, intermediate_size, hidden_size),
        sentinel,
        dtype=torch.bfloat16,
        device="cuda",
    )
    manager.w2_blocks = torch.full(
        (4, hidden_size, intermediate_size),
        sentinel,
        dtype=torch.bfloat16,
        device="cuda",
    )
    w13_cpu = torch.randn(
        num_experts,
        2 * intermediate_size,
        hidden_size,
        dtype=torch.bfloat16,
    )
    w2_cpu = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        dtype=torch.bfloat16,
    )
    manager.expert_params = {
        layer_prefix: OffloadedLayerWeights(
            w13_cpu=w13_cpu,
            w2_cpu=w2_cpu,
            intermediate_size=intermediate_size,
        )
    }
    gpu_table = torch.full((1, num_experts, 3), -1, dtype=torch.int32, device="cuda")
    manager.block_table = _DummyBlockTable(gpu_table)
    manager.prefetch_daemon = _DummyPrefetchDaemon([], manager)
    return manager, layer_prefix, sentinel


def test_backend_expert_manager_layer_prefix_mapping_uses_actual_layer_index():
    class _Module:
        pass

    manager = object.__new__(BackendExpertManager)
    manager.first_k_dense_replace = 0
    manager.layer_prefixes = []
    manager.layer_prefix_to_idx = {}
    manager.layer_idx_to_prefix = {}
    manager.moe_modules = {}

    module_refs = [
        ("model.layers.5.mlp.experts", _Module()),
        ("model.layers.1.mlp.experts", _Module()),
        ("model.layers.3.mlp.experts", _Module()),
    ]
    for layer_prefix, module in module_refs:
        BackendExpertManager.register_moe_module(manager, layer_prefix, module)

    assert BackendExpertManager.get_layer_index(
        manager, "model.layers.1.mlp.experts") == 1
    assert BackendExpertManager.get_layer_index(
        manager, "model.layers.3.mlp.experts") == 3
    assert BackendExpertManager.get_layer_index(
        manager, "model.layers.5.mlp.experts") == 5
    assert BackendExpertManager.get_layer_prefix(
        manager, 1) == "model.layers.1.mlp.experts"
    assert BackendExpertManager.get_layer_prefix(
        manager, 3) == "model.layers.3.mlp.experts"
    assert BackendExpertManager.get_layer_prefix(
        manager, 5) == "model.layers.5.mlp.experts"


def _make_test_daemon(manager) -> PrefetchDaemon:
    daemon = object.__new__(PrefetchDaemon)
    daemon.manager = manager
    daemon.contexts = {}
    daemon.foreground_context_id = None
    daemon._context_rr = deque()
    daemon._ready_masks = {}
    daemon.shutdown_flag = threading.Event()
    daemon.lock = threading.Lock()
    daemon.cv = threading.Condition(daemon.lock)
    return daemon


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_get_offloaded_compute_inputs_mixed_sources():
    StreamContext.init()

    num_experts = 4
    hidden_size = 8
    intermediate_size = 8
    sentinel = -123.0
    layer_prefix = "model.layers.0.mlp.experts"

    manager = object.__new__(BackendExpertManager)
    manager.first_k_dense_replace = 0
    manager.num_hidden_layers = 1
    manager.layer_prefixes = [layer_prefix]
    manager.layer_prefix_to_idx = {layer_prefix: 0}
    manager.pending_topk_updates = {}
    manager.pending_topk_lock = threading.Lock()
    manager._load_targets = {}
    manager._evict_targets = {}
    manager._pending_reserved_blocks = {}
    manager._active_cache_delta_id = None
    manager._active_prefetch_context_id = 1
    manager.prefetch_daemon = _DummyPrefetchDaemon([0])
    manager.comp_flag = 1
    manager.w13_weight_1 = torch.full(
        (num_experts, 2 * intermediate_size, hidden_size),
        sentinel,
        dtype=torch.bfloat16,
        device="cuda",
    )
    manager.w2_weight_1 = torch.full(
        (num_experts, hidden_size, intermediate_size),
        sentinel,
        dtype=torch.bfloat16,
        device="cuda",
    )
    manager.w13_weight_2 = torch.empty_like(manager.w13_weight_1)
    manager.w2_weight_2 = torch.empty_like(manager.w2_weight_1)
    manager.w13_blocks = torch.full(
        (2, intermediate_size, hidden_size),
        sentinel,
        dtype=torch.bfloat16,
        device="cuda",
    )
    manager.w2_blocks = torch.full(
        (1, hidden_size, intermediate_size),
        sentinel,
        dtype=torch.bfloat16,
        device="cuda",
    )

    prefetched_w13 = torch.randn(
        2 * intermediate_size, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    prefetched_w2 = torch.randn(
        hidden_size, intermediate_size, dtype=torch.bfloat16, device="cuda"
    )
    manager.w13_weight_1[0].copy_(prefetched_w13)
    manager.w2_weight_1[0].copy_(prefetched_w2)

    cached_w13 = torch.randn(
        2 * intermediate_size, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    cached_w2 = torch.randn(
        hidden_size, intermediate_size, dtype=torch.bfloat16, device="cuda"
    )
    manager.w13_blocks[0].copy_(cached_w13[:intermediate_size])
    manager.w13_blocks[1].copy_(cached_w13[intermediate_size:])
    manager.w2_blocks[0].copy_(cached_w2)

    cpu_w13 = torch.randn(
        2 * intermediate_size, hidden_size, dtype=torch.bfloat16
    )
    cpu_w2 = torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16)
    manager.expert_params = {
        layer_prefix: OffloadedLayerWeights(
            w13_cpu=torch.stack(
                (
                    torch.zeros_like(cpu_w13),
                    torch.zeros_like(cpu_w13),
                    cpu_w13,
                    torch.zeros_like(cpu_w13),
                )
            ),
            w2_cpu=torch.stack(
                (
                    torch.zeros_like(cpu_w2),
                    torch.zeros_like(cpu_w2),
                    cpu_w2,
                    torch.zeros_like(cpu_w2),
                )
            ),
            intermediate_size=intermediate_size,
        )
    }

    gpu_table = torch.full((1, num_experts, 3), -1, dtype=torch.int32, device="cuda")
    gpu_table[0, 1] = torch.tensor([0, 0, 1], dtype=torch.int32, device="cuda")
    manager.block_table = _DummyBlockTable(gpu_table)

    topk_ids = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32, device="cuda")
    inputs = BackendExpertManager.get_offloaded_compute_inputs(
        manager,
        layer_prefix,
        topk_ids,
        [],
    )

    assert inputs is not None
    assert inputs.expert_source[0].item() == OffloadedExpertComputeInputs.SOURCE_COMP
    assert inputs.expert_source[1].item() == OffloadedExpertComputeInputs.SOURCE_CACHE
    assert inputs.expert_source[2].item() == OffloadedExpertComputeInputs.SOURCE_COMP
    assert inputs.cache_w1_block_ids[1].item() == 0
    assert inputs.cache_w2_block_ids[1].item() == 0
    assert inputs.cache_w3_block_ids[1].item() == 1

    torch.testing.assert_close(inputs.w13_weight_comp[0], prefetched_w13)
    torch.testing.assert_close(inputs.w2_weight_comp[0], prefetched_w2)
    torch.testing.assert_close(inputs.w13_weight_comp[1], torch.full_like(inputs.w13_weight_comp[1], sentinel))
    torch.testing.assert_close(inputs.w2_weight_comp[1], torch.full_like(inputs.w2_weight_comp[1], sentinel))
    torch.testing.assert_close(inputs.w13_weight_comp[2], cpu_w13.cuda())
    torch.testing.assert_close(inputs.w2_weight_comp[2], cpu_w2.cuda())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_build_offloaded_compute_inputs_reuses_metadata_buffers():
    manager = object.__new__(BackendExpertManager)
    manager.w13_blocks = torch.empty((0, 4, 4), dtype=torch.bfloat16, device="cuda")
    manager.w2_blocks = torch.empty((0, 4, 4), dtype=torch.bfloat16, device="cuda")

    w13_weight = torch.empty((4, 8, 4), dtype=torch.bfloat16, device="cuda")
    w2_weight = torch.empty((4, 4, 4), dtype=torch.bfloat16, device="cuda")

    first = BackendExpertManager._build_offloaded_compute_inputs(
        manager,
        1,
        w13_weight,
        w2_weight,
    )
    second = BackendExpertManager._build_offloaded_compute_inputs(
        manager,
        1,
        w13_weight,
        w2_weight,
    )

    assert first is second
    assert first.expert_source.data_ptr() == second.expert_source.data_ptr()
    assert first.cache_w1_block_ids.data_ptr() == second.cache_w1_block_ids.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_schedule_prefetch_skips_when_all_predicted_experts_are_cached():
    gpu_table = torch.full((2, 4, 3), -1, dtype=torch.int32, device="cuda")
    gpu_table[1, 1] = torch.tensor([0, 0, 1], dtype=torch.int32, device="cuda")
    gpu_table[1, 2] = torch.tensor([2, 1, 3], dtype=torch.int32, device="cuda")
    manager = _make_prefetch_manager(gpu_table)

    BackendExpertManager._schedule_prefetch(
        manager,
        0,
        torch.tensor([[1, 2], [2, 1]], dtype=torch.int32, device="cuda"),
    )

    assert manager.prefetch_daemon.scheduled == []
    assert manager.prefetch_daemon.clear_calls == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_schedule_prefetch_keeps_uncached_next_load_when_predictions_are_cached():
    gpu_table = torch.full((2, 4, 3), -1, dtype=torch.int32, device="cuda")
    gpu_table[1, 1] = torch.tensor([0, 0, 1], dtype=torch.int32, device="cuda")
    gpu_table[1, 2] = torch.tensor([2, 1, 3], dtype=torch.int32, device="cuda")
    manager = _make_prefetch_manager(gpu_table)
    manager._load_targets = {1: {3}}
    manager._pending_reserved_blocks = {1: {3: (4, 2, 5)}}

    BackendExpertManager._schedule_prefetch(
        manager,
        0,
        torch.tensor([[1, 2], [2, 1]], dtype=torch.int32, device="cuda"),
    )

    assert manager.prefetch_daemon.scheduled == [(17, 1, [3], [])]
    assert manager.prefetch_daemon.clear_calls == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_schedule_prefetch_prioritizes_next_load_over_comm_prefetch():
    gpu_table = torch.full((2, 6, 3), -1, dtype=torch.int32, device="cuda")
    gpu_table[1, 1] = torch.tensor([0, 0, 1], dtype=torch.int32, device="cuda")
    gpu_table[1, 4] = torch.tensor([2, 1, 3], dtype=torch.int32, device="cuda")
    manager = _make_prefetch_manager(gpu_table)
    manager._load_targets = {1: {2, 3}}
    manager._pending_reserved_blocks = {
        1: {
            2: (4, 2, 5),
            3: (6, 3, 7),
        }
    }

    BackendExpertManager._schedule_prefetch(
        manager,
        0,
        torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device="cuda"),
    )

    assert manager.prefetch_daemon.scheduled == [(17, 1, [2, 3], [0])]
    assert manager.prefetch_daemon.clear_calls == 0


def test_prefetch_daemon_schedule_prefetch_accepts_cpu_iterables():
    daemon = _make_test_daemon(types.SimpleNamespace(comp_flag=1))
    PrefetchDaemon.create_context(daemon, 1)
    context = daemon.contexts[1]
    context.comp_prefetch.queue = [98]
    context.comp_prefetch.loaded_queue = [(88, None)]
    context.comp_prefetch.layer_idx = 3

    PrefetchDaemon.schedule_prefetch(
        daemon,
        1,
        5,
        np.array([2, 4], dtype=np.int64),
    )

    assert context.comp_prefetch.queue == []
    assert context.comp_prefetch.loaded_queue == []
    assert context.comp_prefetch.layer_idx == 5
    assert context.comp_prefetch.pending_request is not None
    assert context.comp_prefetch.pending_request.context_id == 1
    assert context.comp_prefetch.pending_request.layer_idx == 5
    assert context.comp_prefetch.pending_request.buffer_idx == 1
    assert context.comp_prefetch.pending_request.topk_ids_pred.tolist() == [2, 4]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_schedule_prefetch_enqueues_block_loads_before_producer_event_is_ready():
    manager = _make_prefetch_manager(
        torch.full((1, 4, 3), -1, dtype=torch.int32, device="cuda"),
        num_hidden_layers=1,
    )
    manager._pending_reserved_blocks = {0: {1: (2, 1, 3)}}
    daemon = _make_test_daemon(manager)
    PrefetchDaemon.create_context(daemon, 1)

    topk_ids_pred = torch.tensor([2, 3], dtype=torch.long, device="cuda")
    delayed_stream = torch.cuda.Stream()
    with torch.cuda.stream(delayed_stream):
        torch.cuda._sleep(2_000_000)
        PrefetchDaemon.schedule_prefetch(
            daemon,
            1,
            0,
            topk_ids_pred,
            {1},
        )

    context = daemon.contexts[1]
    assert context.block_load.layer_idx == 0
    assert context.block_load.queue == [1]
    assert context.comp_prefetch.pending_request is not None

    work_item = PrefetchDaemon._pop_next_work_locked(daemon)
    assert work_item == ("block", 1, 0, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_notify_layer_loading_does_not_require_comp_request_activation():
    manager, _layer_prefix, _sentinel = _make_dynamic_backend_manager()
    manager.apply_cache_delta(
        ExpertCacheDelta(
            delta_id=41,
            experts_to_load=[(0, 1)],
            experts_to_evict=[],
            new_expert_to_block={
                (0, 1, "w1"): 2,
                (0, 1, "w2"): 1,
                (0, 1, "w3"): 3,
            },
            evict_commit_mode="row",
        ))

    daemon = _make_test_daemon(manager)
    manager.prefetch_daemon = daemon
    manager._active_prefetch_context_id = 1
    PrefetchDaemon.create_context(daemon, 1)

    topk_ids_pred = torch.tensor([2], dtype=torch.long, device="cuda")
    delayed_stream = torch.cuda.Stream()
    with torch.cuda.stream(delayed_stream):
        torch.cuda._sleep(2_000_000)
        PrefetchDaemon.schedule_prefetch(
            daemon,
            1,
            0,
            topk_ids_pred,
            {1},
        )

    work_item = PrefetchDaemon._pop_next_work_locked(daemon)
    assert work_item == ("block", 1, 0, 1)
    assert PrefetchDaemon._load_expert_to_blocks(daemon, 1, 0, 1)
    completion_event = daemon.contexts[1].block_load.ready_events[1]
    completion_event.synchronize()

    ready_mask, handoff_event, has_block_loads = PrefetchDaemon.notify_layer_loading(
        daemon,
        1,
        0,
        buffer_idx=1,
        num_experts=manager.w13_weight_1.size(0),
        device=manager.w13_weight_1.device,
    )

    assert has_block_loads
    assert not ready_mask.any()
    assert handoff_event is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_stale_comp_prefetch_results_are_dropped_after_layer_entry():
    StreamContext.init()
    cpu_w13 = torch.randn(1, 4, 4)
    cpu_w2 = torch.randn(1, 4, 2)

    manager = object.__new__(BackendExpertManager)
    manager.layer_prefixes = ["layer"]
    manager.layer_prefix_to_idx = {"layer": 0}
    manager.layer_idx_to_prefix = {0: "layer"}
    manager.first_k_dense_replace = 0
    manager.expert_params = {
        "layer": OffloadedLayerWeights(
            w13_cpu=cpu_w13,
            w2_cpu=cpu_w2,
            intermediate_size=2,
        )
    }
    manager.w13_weight_1 = torch.zeros_like(cpu_w13, device="cuda")
    manager.w2_weight_1 = torch.zeros_like(cpu_w2, device="cuda")
    manager.w13_weight_2 = torch.zeros_like(cpu_w13, device="cuda")
    manager.w2_weight_2 = torch.zeros_like(cpu_w2, device="cuda")
    manager.comp_flag = 1
    manager._pending_reserved_blocks = {}
    manager.block_table = types.SimpleNamespace(
        block_table=types.SimpleNamespace(
            np=np.array([[[-1, -1, -1]]], dtype=np.int32),
        )
    )
    manager.no_copy_compute_enabled = lambda: False

    daemon = _make_test_daemon(manager)
    PrefetchDaemon.create_context(daemon, 1)
    context = daemon.contexts[1]
    context.comp_prefetch.layer_idx = 0
    context.comp_prefetch.buffer_idx = 1
    context.comp_prefetch.version = 1
    context.comp_prefetch.queue = [0]

    work_item = PrefetchDaemon._pop_next_work_locked(daemon)
    assert work_item == ("comp", 1, 0, 0, 1, 1)

    ready_mask, handoff_event, has_block_loads = PrefetchDaemon.notify_layer_loading(
        daemon,
        1,
        0,
        buffer_idx=1,
        num_experts=1,
        device=manager.w13_weight_1.device,
    )

    assert not ready_mask.any()
    assert handoff_event is None
    assert not has_block_loads
    assert context.comp_prefetch.layer_idx is None
    assert context.comp_prefetch.version == 2
    assert not PrefetchDaemon._load_expert_to_comp(
        daemon,
        1,
        0,
        0,
        request_version=1,
        target_buffer_idx=1,
    )
    assert context.comp_prefetch.loaded_queue == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_prefetch_daemon_skips_cached_copy_in_no_copy_mode():
    StreamContext.init()
    manager = object.__new__(BackendExpertManager)
    manager.layer_prefixes = ["layer"]
    manager.layer_prefix_to_idx = {"layer": 0}
    manager.layer_idx_to_prefix = {0: "layer"}
    manager.first_k_dense_replace = 0
    manager.expert_params = {}
    manager.w13_weight_1 = torch.zeros(1, 4, 4, device="cuda")
    manager.w2_weight_1 = torch.zeros(1, 4, 2, device="cuda")
    manager.w13_weight_2 = torch.zeros(1, 4, 4, device="cuda")
    manager.w2_weight_2 = torch.zeros(1, 4, 2, device="cuda")
    manager.comp_flag = 1
    manager._pending_reserved_blocks = {}
    manager.block_table = types.SimpleNamespace(
        block_table=types.SimpleNamespace(
            np=np.array([[[0, 0, 1]]], dtype=np.int32),
        )
    )
    manager.no_copy_compute_enabled = lambda: True

    daemon = _make_test_daemon(manager)
    PrefetchDaemon.create_context(daemon, 1)
    context = daemon.contexts[1]
    context.comp_prefetch.queue = [0]
    context.comp_prefetch.layer_idx = 0
    context.comp_prefetch.version = 1

    assert PrefetchDaemon._load_expert_to_comp(
        daemon,
        1,
        0,
        0,
        request_version=1,
        target_buffer_idx=1,
    )
    assert context.comp_prefetch.loaded_queue == []
    assert torch.count_nonzero(manager.w13_weight_1) == 0
    assert torch.count_nonzero(manager.w2_weight_1) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_prefetch_daemon_loads_uncached_expert_from_packed_state():
    StreamContext.init()
    cpu_w13 = torch.randn(1, 4, 4)
    cpu_w2 = torch.randn(1, 4, 2)

    manager = object.__new__(BackendExpertManager)
    manager.layer_prefixes = ["layer"]
    manager.layer_prefix_to_idx = {"layer": 0}
    manager.layer_idx_to_prefix = {0: "layer"}
    manager.first_k_dense_replace = 0
    manager.expert_params = {
        "layer": OffloadedLayerWeights(
            w13_cpu=cpu_w13,
            w2_cpu=cpu_w2,
            intermediate_size=2,
        )
    }
    manager.w13_weight_1 = torch.zeros_like(cpu_w13, device="cuda")
    manager.w2_weight_1 = torch.zeros_like(cpu_w2, device="cuda")
    manager.w13_weight_2 = torch.zeros_like(cpu_w13, device="cuda")
    manager.w2_weight_2 = torch.zeros_like(cpu_w2, device="cuda")
    manager.comp_flag = 1
    manager._pending_reserved_blocks = {}
    manager.block_table = types.SimpleNamespace(
        block_table=types.SimpleNamespace(
            np=np.array([[[-1, -1, -1]]], dtype=np.int32),
        )
    )
    manager.no_copy_compute_enabled = lambda: False

    daemon = _make_test_daemon(manager)
    PrefetchDaemon.create_context(daemon, 1)
    context = daemon.contexts[1]
    context.comp_prefetch.queue = [0]
    context.comp_prefetch.layer_idx = 0
    context.comp_prefetch.version = 1

    assert PrefetchDaemon._load_expert_to_comp(
        daemon,
        1,
        0,
        0,
        request_version=1,
        target_buffer_idx=1,
    )
    assert len(context.comp_prefetch.loaded_queue) == 1
    expert_id, completion_event = context.comp_prefetch.loaded_queue[0]
    assert expert_id == 0
    completion_event.synchronize()
    assert PrefetchDaemon.notify_layer_loading(daemon, 1, 0) == [0]
    torch.testing.assert_close(manager.w13_weight_1[0], cpu_w13[0].cuda())
    torch.testing.assert_close(manager.w2_weight_1[0], cpu_w2[0].cuda())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_apply_cache_delta_defers_block_table_commit_until_context_notify():
    manager, _layer_prefix, _sentinel = _make_dynamic_backend_manager()
    delta = ExpertCacheDelta(
        delta_id=11,
        experts_to_load=[(0, 1)],
        experts_to_evict=[],
        new_expert_to_block={
            (0, 1, "w1"): 2,
            (0, 1, "w2"): 1,
            (0, 1, "w3"): 3,
        },
    )
    manager.apply_cache_delta(delta)

    assert manager.block_table.block_table.gpu[0, 1, 0].item() == -1

    daemon = _make_test_daemon(manager)
    manager.prefetch_daemon = daemon
    manager._active_prefetch_context_id = 1
    PrefetchDaemon.create_context(daemon, 1)
    PrefetchDaemon.schedule_prefetch(
        daemon,
        1,
        0,
        torch.empty((0,), dtype=torch.long),
        {1},
    )
    context = daemon.contexts[1]
    request = context.comp_prefetch.pending_request
    assert request is not None
    context.comp_prefetch.pending_request = None
    PrefetchDaemon._activate_prefetch_request(daemon, request)

    work_item = PrefetchDaemon._pop_next_work_locked(daemon)
    assert work_item == ("block", 1, 0, 1)
    assert PrefetchDaemon._load_expert_to_blocks(daemon, 1, 0, 1)
    completion_event = context.block_load.ready_events[1]
    assert completion_event is not None
    completion_event.synchronize()

    ready_mask, handoff_event, has_block_loads = PrefetchDaemon.notify_layer_loading(
        daemon,
        1,
        0,
        buffer_idx=1,
        num_experts=manager.w13_weight_1.size(0),
        device=manager.w13_weight_1.device,
    )
    assert has_block_loads
    assert not ready_mask.any()
    assert handoff_event is None

    manager._activate_pending_loads(0)
    assert manager.block_table.block_table.gpu[0, 1].tolist() == [2, 1, 3]
    assert manager.block_table.block_table.cpu[0, 1].tolist() == [2, 1, 3]
    assert manager._active_cache_delta_id is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_apply_cache_delta_keeps_evictions_visible_until_post_execute_commit():
    manager, _layer_prefix, _sentinel = _make_dynamic_backend_manager()
    manager.block_table.block_table.cpu[0, 0] = torch.tensor(
        [0, 0, 1], dtype=torch.int32
    )
    manager.block_table.block_table.gpu[0, 0] = torch.tensor(
        [0, 0, 1], dtype=torch.int32, device="cuda"
    )

    manager.apply_cache_delta(
        ExpertCacheDelta(
            delta_id=19,
            experts_to_load=[],
            experts_to_evict=[(0, 0)],
            new_expert_to_block={},
            evict_commit_mode="row",
        ))

    assert manager.block_table.block_table.gpu[0, 0].tolist() == [0, 0, 1]
    assert manager.block_table.block_table.cpu[0, 0].tolist() == [0, 0, 1]
    assert manager._active_cache_delta_id == 19

    manager.commit_post_execute_evictions()

    assert manager.block_table.block_table.gpu[0, 0].tolist() == [-1, -1, -1]
    assert manager.block_table.block_table.cpu[0, 0].tolist() == [-1, -1, -1]
    assert manager.block_table.commit_row_calls == [(0, 0)]
    assert manager.block_table.commit_all_calls == 0
    assert manager._active_cache_delta_id is None

    manager.commit_post_execute_evictions()
    assert manager._active_cache_delta_id is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_multi_layer_shrink_can_commit_entire_block_table_once():
    gpu_table = torch.full((2, 3, 3), -1, dtype=torch.int32, device="cuda")
    manager = _make_prefetch_manager(gpu_table)
    manager.block_table.block_table.cpu[0, 0] = torch.tensor([0, 0, 1], dtype=torch.int32)
    manager.block_table.block_table.gpu[0, 0] = torch.tensor(
        [0, 0, 1], dtype=torch.int32, device="cuda")
    manager.block_table.block_table.cpu[1, 1] = torch.tensor([2, 1, 3], dtype=torch.int32)
    manager.block_table.block_table.gpu[1, 1] = torch.tensor(
        [2, 1, 3], dtype=torch.int32, device="cuda")

    manager.apply_cache_delta(
        ExpertCacheDelta(
            delta_id=20,
            experts_to_load=[],
            experts_to_evict=[(0, 0), (1, 1)],
            new_expert_to_block={},
            evict_commit_mode="table",
        ))

    manager.commit_post_execute_evictions()

    assert manager.block_table.block_table.gpu[0, 0].tolist() == [-1, -1, -1]
    assert manager.block_table.block_table.gpu[1, 1].tolist() == [-1, -1, -1]
    assert manager.block_table.commit_row_calls == []
    assert manager.block_table.commit_all_calls == 1
    assert manager._active_cache_delta_id is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_delta_completion_waits_for_post_execute_eviction_commit():
    manager, _layer_prefix, _sentinel = _make_dynamic_backend_manager()
    manager.block_table.block_table.cpu[0, 0] = torch.tensor(
        [0, 0, 1], dtype=torch.int32
    )
    manager.block_table.block_table.gpu[0, 0] = torch.tensor(
        [0, 0, 1], dtype=torch.int32, device="cuda"
    )

    manager.apply_cache_delta(
        ExpertCacheDelta(
            delta_id=23,
            experts_to_load=[(0, 1)],
            experts_to_evict=[(0, 0)],
            new_expert_to_block={
                (0, 1, "w1"): 2,
                (0, 1, "w2"): 1,
                (0, 1, "w3"): 3,
            },
            evict_commit_mode="row",
        ))

    manager._activate_pending_loads(0)
    assert manager.block_table.block_table.gpu[0, 0].tolist() == [0, 0, 1]
    assert manager.block_table.block_table.gpu[0, 1].tolist() == [2, 1, 3]
    assert manager._active_cache_delta_id == 23

    manager.commit_post_execute_evictions()

    assert manager.block_table.block_table.gpu[0, 0].tolist() == [-1, -1, -1]
    assert manager.block_table.block_table.gpu[0, 1].tolist() == [2, 1, 3]
    assert manager._active_cache_delta_id is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_apply_cache_delta_rejects_new_delta_while_previous_delta_is_active():
    manager, _layer_prefix, _sentinel = _make_dynamic_backend_manager()
    manager.apply_cache_delta(
        ExpertCacheDelta(
            delta_id=31,
            experts_to_load=[(0, 1)],
            experts_to_evict=[],
            new_expert_to_block={
                (0, 1, "w1"): 2,
                (0, 1, "w2"): 1,
                (0, 1, "w3"): 3,
            },
            evict_commit_mode="row",
        ))

    with pytest.raises(RuntimeError, match="previous delta was fully applied"):
        manager.apply_cache_delta(
            ExpertCacheDelta(
                delta_id=32,
                experts_to_load=[],
                experts_to_evict=[],
                new_expert_to_block={},
                evict_commit_mode="row",
            ))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_prefetch_contexts_are_isolated_across_batches():
    manager = _make_prefetch_manager(
        torch.full((2, 6, 3), -1, dtype=torch.int32, device="cuda"))
    manager._pending_reserved_blocks = {1: {3: (4, 2, 5)}}
    daemon = _make_test_daemon(manager)

    PrefetchDaemon.create_context(daemon, 1)
    PrefetchDaemon.create_context(daemon, 2)

    PrefetchDaemon.schedule_prefetch(
        daemon,
        1,
        1,
        torch.tensor([0, 1], dtype=torch.long),
        {3},
    )
    PrefetchDaemon.schedule_prefetch(
        daemon,
        2,
        1,
        torch.tensor([4, 5], dtype=torch.long),
        set(),
    )

    request1 = daemon.contexts[1].comp_prefetch.pending_request
    request2 = daemon.contexts[2].comp_prefetch.pending_request
    assert request1 is not None
    assert request2 is not None
    daemon.contexts[1].comp_prefetch.pending_request = None
    daemon.contexts[2].comp_prefetch.pending_request = None
    PrefetchDaemon._activate_prefetch_request(daemon, request1)
    PrefetchDaemon._activate_prefetch_request(daemon, request2)

    assert daemon.contexts[1].block_load.layer_idx == 1
    assert daemon.contexts[1].block_load.target_ids == {3}
    assert daemon.contexts[1].block_load.queue == [3]
    assert daemon.contexts[2].comp_prefetch.layer_idx == 1
    assert daemon.contexts[2].block_load.target_ids == set()
    assert daemon.contexts[2].block_load.queue == []


def test_prefetch_daemon_prioritizes_foreground_context():
    manager = _make_prefetch_manager(
        torch.full((2, 6, 3), -1, dtype=torch.int32, device="cuda"))
    daemon = _make_test_daemon(manager)

    PrefetchDaemon.create_context(daemon, 1)
    PrefetchDaemon.create_context(daemon, 2)
    daemon.contexts[1] = PrefetchContext(
        context_id=1,
        block_load=BlockLoadState(),
        comp_prefetch=CompPrefetchState(
            layer_idx=0,
            buffer_idx=1,
            version=1,
            queue=[1],
        ),
    )
    daemon.contexts[2] = PrefetchContext(
        context_id=2,
        block_load=BlockLoadState(),
        comp_prefetch=CompPrefetchState(
            layer_idx=0,
            buffer_idx=1,
            version=1,
            queue=[2],
        ),
    )
    PrefetchDaemon.set_foreground_context(daemon, 2)

    work_item = PrefetchDaemon._pop_next_work_locked(daemon)

    assert work_item == ("comp", 2, 0, 2, 1, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_next_load_overlap_is_materialized_in_blocks_and_consumed_from_cache():
    manager, layer_prefix, sentinel = _make_dynamic_backend_manager()
    delta = ExpertCacheDelta(
        delta_id=17,
        experts_to_load=[(0, 1)],
        experts_to_evict=[],
        new_expert_to_block={
            (0, 1, "w1"): 2,
            (0, 1, "w2"): 1,
            (0, 1, "w3"): 3,
        },
        evict_commit_mode="row",
    )
    manager.apply_cache_delta(delta)

    daemon = _make_test_daemon(manager)
    manager.prefetch_daemon = daemon
    manager._active_prefetch_context_id = 1
    PrefetchDaemon.create_context(daemon, 1)
    PrefetchDaemon.schedule_prefetch(
        daemon,
        1,
        0,
        torch.tensor([[1]], dtype=torch.int32),
        {1},
    )
    context = daemon.contexts[1]
    request = context.comp_prefetch.pending_request
    assert request is not None
    context.comp_prefetch.pending_request = None
    PrefetchDaemon._activate_prefetch_request(daemon, request)

    work_item = PrefetchDaemon._pop_next_work_locked(daemon)
    assert work_item == ("block", 1, 0, 1)
    assert PrefetchDaemon._load_expert_to_blocks(daemon, 1, 0, 1)
    completion_event = context.block_load.ready_events[1]
    assert completion_event is not None
    completion_event.synchronize()

    inputs = manager.get_offloaded_compute_inputs(
        layer_prefix,
        torch.tensor([[1]], dtype=torch.int32, device="cuda"),
        torch.tensor([[1]], dtype=torch.int32, device="cuda"),
    )

    assert inputs is not None
    assert inputs.expert_source[1].item() == OffloadedExpertComputeInputs.SOURCE_CACHE
    assert inputs.cache_w1_block_ids[1].item() == 2
    assert inputs.cache_w2_block_ids[1].item() == 1
    assert inputs.cache_w3_block_ids[1].item() == 3
    torch.testing.assert_close(
        inputs.w13_weight_comp[1],
        torch.full_like(inputs.w13_weight_comp[1], sentinel),
    )
    torch.testing.assert_close(
        inputs.w2_weight_comp[1],
        torch.full_like(inputs.w2_weight_comp[1], sentinel),
    )
    torch.testing.assert_close(
        manager.w13_blocks[2],
        manager.expert_params[layer_prefix].w13_cpu[1, :2].cuda(),
    )
    torch.testing.assert_close(
        manager.w13_blocks[3],
        manager.expert_params[layer_prefix].w13_cpu[1, 2:].cuda(),
    )
    torch.testing.assert_close(
        manager.w2_blocks[1],
        manager.expert_params[layer_prefix].w2_cpu[1].cuda(),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_notify_layer_loading_returns_handoff_event_for_comm_prefetch_buffer():
    manager, _layer_prefix, _sentinel = _make_dynamic_backend_manager()
    daemon = _make_test_daemon(manager)
    PrefetchDaemon.create_context(daemon, 1)
    handoff_event = torch.cuda.Event()
    handoff_event.record()
    handoff_event.synchronize()
    context = daemon.contexts[1]
    context.comp_prefetch.loaded_queue = [(1, handoff_event)]
    context.comp_prefetch.layer_idx = 0
    context.comp_prefetch.buffer_idx = 1
    context.comp_prefetch.version = 1

    ready_mask, handoff_event, has_block_loads = PrefetchDaemon.notify_layer_loading(
        daemon,
        1,
        0,
        buffer_idx=1,
        num_experts=manager.w13_weight_1.size(0),
        device=manager.w13_weight_1.device,
    )

    assert ready_mask[1].item()
    assert handoff_event is not None
    assert not has_block_loads
    assert context.comp_prefetch.layer_idx is None
    assert context.comp_prefetch.queue == []
