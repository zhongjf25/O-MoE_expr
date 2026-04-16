# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    ExpertHotsetCollector,
    RoutedExpertsCapturer,
    RoutedExpertsReader,
)

pytestmark = pytest.mark.cpu_test


class DummyRouter(BaseRouter):
    @property
    def routing_method_type(self) -> RoutingMethodType:
        return RoutingMethodType.FUSED_TOPK

    def _compute_routing(self, hidden_states, router_logits, indices_type):
        topk_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
        return topk_weights, topk_ids

    def _apply_eplb_mapping(self, topk_ids: torch.Tensor) -> torch.Tensor:
        # Make mapping observable without requiring CUDA EPLB path.
        return topk_ids + 10


def _make_router() -> DummyRouter:
    return DummyRouter(
        top_k=2,
        global_num_experts=16,
        eplb_state=EplbLayerState(),
        enable_eplb=False,
        indices_type_getter=None,
    )


def test_base_router_capture_pre_eplb_mapping():
    router = _make_router()
    captured = []

    def capture_fn(ids):
        captured.append(ids.clone())

    router.set_capture_fn(capture_fn)
    topk_weights, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert topk_weights.shape == topk_ids.shape
    assert len(captured) == 1
    assert torch.equal(captured[0], torch.tensor([[1, 2], [3, 4]]))
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_base_router_capture_with_eplb_enabled():
    router = _make_router()
    router.enable_eplb = True
    router.eplb_state.expert_load_view = torch.zeros(32, dtype=torch.int64)
    router.eplb_state.logical_to_physical_map = torch.arange(32).view(32, 1)
    router.eplb_state.logical_replica_count = torch.ones(32, dtype=torch.int64)

    captured = []

    def capture_fn(ids):
        captured.append(ids.clone())

    router.set_capture_fn(capture_fn)
    _, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert len(captured) == 1
    # Capture should see logical ids pre-EPLB mapping.
    assert torch.equal(captured[0], torch.tensor([[1, 2], [3, 4]]))
    # Our DummyRouter mapping adds +10.
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_gpu_model_runner_binds_router_capture(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    class DummyFusedMoE:
        def __init__(self):
            self.layer_id = 7
            self.router = _make_router()

    class DummyCapturer:
        def __init__(self):
            self.calls = []

        def capture(self, layer_id, topk_ids):
            self.calls.append((layer_id, topk_ids))

    dummy_module = DummyFusedMoE()

    # Patch the runtime import inside _bind_routed_experts_capturer.
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "FusedMoE", DummyFusedMoE)

    dummy_self = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(
            static_forward_context={"dummy": dummy_module}
        )
    )

    capturer = DummyCapturer()
    gmr.GPUModelRunner._bind_routed_experts_capturer(dummy_self, capturer)

    assert dummy_module.router.capture_fn is not None
    dummy_module.router.capture_fn(torch.tensor([[5, 6]]))

    assert len(capturer.calls) == 1
    layer_id, topk_ids = capturer.calls[0]
    assert layer_id == 7
    assert torch.equal(topk_ids, torch.tensor([[5, 6]]))


def test_gpu_model_runner_binding_stage(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    class DummyFusedMoE:
        def __init__(self):
            self.layer_id = 11
            self.router = _make_router()

    class DummyCapturer:
        def __init__(self):
            self.calls = []

        def capture(self, layer_id, topk_ids):
            self.calls.append((layer_id, topk_ids))

    dummy_module = DummyFusedMoE()

    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "FusedMoE", DummyFusedMoE)

    dummy_self = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(
            static_forward_context={"dummy": dummy_module}
        )
    )

    # Before binding, no capture hook.
    assert dummy_module.router.capture_fn is None

    capturer = DummyCapturer()
    gmr.GPUModelRunner._bind_routed_experts_capturer(dummy_self, capturer)

    # After binding, hook should exist and be callable.
    assert callable(dummy_module.router.capture_fn)
    dummy_module.router.capture_fn(torch.tensor([[9, 10]]))
    assert len(capturer.calls) == 1


def test_expert_hotset_collector_accumulates_and_resets(monkeypatch):
    import vllm.model_executor.layers.fused_moe.routed_experts_capturer as rec

    monkeypatch.setattr(rec, "get_tensor_model_parallel_rank", lambda: 0)
    collector = ExpertHotsetCollector(num_hidden_layers=3, num_experts=6)

    collector.capture(1, torch.tensor([[1, 2], [2, 4]], dtype=torch.int64))
    collector.capture(1, torch.tensor([[2, 5]], dtype=torch.int32))

    snapshot = collector.snapshot()
    assert snapshot[0] == [0, 0, 0, 0, 0, 0]
    assert snapshot[1] == [0, 1, 3, 0, 1, 1]
    assert snapshot[2] == [0, 0, 0, 0, 0, 0]

    snapshot[1][2] = 999
    assert collector.snapshot()[1] == [0, 1, 3, 0, 1, 1]

    collector.reset()
    assert collector.snapshot()[1] == [0, 0, 0, 0, 0, 0]


def test_expert_hotset_collector_skips_non_tp0(monkeypatch):
    import vllm.model_executor.layers.fused_moe.routed_experts_capturer as rec

    monkeypatch.setattr(rec, "get_tensor_model_parallel_rank", lambda: 1)
    collector = ExpertHotsetCollector(num_hidden_layers=2, num_experts=4)

    collector.capture(0, torch.tensor([[1, 2]], dtype=torch.int64))

    assert collector.snapshot() == {}


def test_gpu_model_runner_binds_hotset_collector(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    class DummyFusedMoE:
        def __init__(self):
            self.layer_id = 5
            self.router = _make_router()

    class DummyCollector:
        def __init__(self):
            self.calls = []

        def capture(self, layer_id, topk_ids):
            self.calls.append((layer_id, topk_ids))

    dummy_module = DummyFusedMoE()

    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "FusedMoE", DummyFusedMoE)

    dummy_self = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(
            static_forward_context={"dummy": dummy_module}
        )
    )

    collector = DummyCollector()
    gmr.GPUModelRunner._bind_expert_hotset_collector(dummy_self, collector)

    assert callable(dummy_module.router.capture_fn)
    dummy_module.router.capture_fn(torch.tensor([[7, 8]]))
    assert len(collector.calls) == 1
    layer_id, topk_ids = collector.calls[0]
    assert layer_id == 5
    assert torch.equal(topk_ids, torch.tensor([[7, 8]]))


def test_gpu_model_runner_router_capture_prefers_hotset_collector(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    calls = []
    dummy_self = types.SimpleNamespace(
        vllm_config=object(),
        model_config=types.SimpleNamespace(enable_return_routed_experts=False),
        init_expert_hotset_collector=lambda: calls.append("collector"),
        init_routed_experts_capturer=lambda: calls.append("capturer"),
    )

    monkeypatch.setattr(gmr, "expert_hotset_collect_mode_enabled", lambda _: True)

    gmr.GPUModelRunner.maybe_init_router_capture(dummy_self)

    assert calls == ["collector"]


def test_gpu_model_runner_router_capture_uses_routed_experts_when_enabled(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    calls = []
    dummy_self = types.SimpleNamespace(
        vllm_config=object(),
        model_config=types.SimpleNamespace(enable_return_routed_experts=True),
        init_expert_hotset_collector=lambda: calls.append("collector"),
        init_routed_experts_capturer=lambda: calls.append("capturer"),
    )

    monkeypatch.setattr(gmr, "expert_hotset_collect_mode_enabled", lambda _: False)

    gmr.GPUModelRunner.maybe_init_router_capture(dummy_self)

    assert calls == ["capturer"]


def test_gpu_model_runner_router_capture_rejects_conflicting_modes(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    dummy_self = types.SimpleNamespace(
        vllm_config=object(),
        model_config=types.SimpleNamespace(enable_return_routed_experts=True),
        init_expert_hotset_collector=lambda: None,
        init_routed_experts_capturer=lambda: None,
    )

    monkeypatch.setattr(gmr, "expert_hotset_collect_mode_enabled", lambda _: True)

    with pytest.raises(ValueError, match="cannot both be enabled"):
        gmr.GPUModelRunner.maybe_init_router_capture(dummy_self)


def test_routed_experts_capture_supports_layerwise_slot_mapping(tmp_path, monkeypatch):
    import vllm.model_executor.layers.fused_moe.routed_experts_capturer as rec

    monkeypatch.setattr(rec, "get_tensor_model_parallel_rank", lambda: 0)

    capturer = RoutedExpertsCapturer()
    reader = RoutedExpertsReader()
    lock_file = Path(tmp_path) / "routed.lock"
    lock_file.touch()

    host_buffer = np.zeros((32, 2, 2), dtype=np.int32)
    capturer._lock_file = str(lock_file)
    capturer._host_buffer_view = host_buffer
    capturer._device_buffer = torch.tensor(
        [
            [[11, 12], [21, 22]],
            [[13, 14], [23, 24]],
            [[15, 16], [25, 26]],
        ],
        dtype=torch.int32,
    )

    reader._lock_file = str(lock_file)
    reader._host_buffer_view = host_buffer

    slot_mapping = np.array(
        [
            [3, 4, 5],
            [13, 14, 15],
        ],
        dtype=np.int64,
    )

    capturer.save_captured_experts(slot_mapping)
    routed_experts = reader.get_routed_experts(slot_mapping)

    assert routed_experts.shape == (3, 2, 2)
    np.testing.assert_array_equal(
        routed_experts,
        np.array(
            [
                [[11, 12], [21, 22]],
                [[13, 14], [23, 24]],
                [[15, 16], [25, 26]],
            ],
            dtype=np.int32,
        ),
    )
