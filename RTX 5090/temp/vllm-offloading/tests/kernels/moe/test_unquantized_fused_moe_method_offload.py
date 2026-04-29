# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import types

import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)


class _FakeKernel:
    def __init__(self, return_value):
        self.return_value = return_value
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.return_value


def _make_method() -> UnquantizedFusedMoEMethod:
    method = object.__new__(UnquantizedFusedMoEMethod)
    method.moe = make_dummy_moe_config(
        num_experts=4,
        experts_per_token=2,
        hidden_dim=8,
        intermediate_size_per_partition=8,
        in_dtype=torch.bfloat16,
    )
    method.unquantized_backend = UnquantizedMoeBackend.TRITON
    method._no_copy_fallback_logged = False
    return method


def test_forward_cuda_uses_offloaded_kernel_when_enabled(monkeypatch):
    method = _make_method()
    expected = torch.randn(2, 8)
    legacy_kernel = _FakeKernel(torch.zeros_like(expected))
    offloaded_kernel = _FakeKernel(expected)
    method.kernel = legacy_kernel
    method.offloaded_kernel = offloaded_kernel

    fake_manager = types.SimpleNamespace(
        no_copy_compute_enabled=lambda: True,
        get_offloaded_compute_inputs=lambda prefix, topk_ids, topk_ids_pred: "inputs",
        get_experts_with_topk_ids=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("copy path should not be used")
        ),
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method.BackendExpertManager",
        lambda: fake_manager,
    )

    layer = types.SimpleNamespace(
        prefix="layer",
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=False,
        global_num_experts=4,
        expert_map=None,
    )
    x = torch.randn(2, 8)
    topk_weights = torch.randn(2, 2)
    topk_ids = torch.tensor([[0, 1], [2, 3]])

    actual = UnquantizedFusedMoEMethod.forward_cuda(
        method,
        layer,
        x,
        topk_weights,
        topk_ids,
        [],
        None,
    )

    assert torch.equal(actual, expected)
    assert len(offloaded_kernel.calls) == 1
    assert len(legacy_kernel.calls) == 0


def test_forward_cuda_falls_back_to_copy_path_for_unsupported_inputs(monkeypatch):
    method = _make_method()
    expected = torch.randn(2, 8)
    legacy_kernel = _FakeKernel(expected)
    offloaded_kernel = _FakeKernel(torch.zeros_like(expected))
    method.kernel = legacy_kernel
    method.offloaded_kernel = offloaded_kernel

    w13 = torch.randn(4, 16, 8)
    w2 = torch.randn(4, 8, 8)
    fake_manager = types.SimpleNamespace(
        no_copy_compute_enabled=lambda: True,
        get_offloaded_compute_inputs=lambda prefix, topk_ids, topk_ids_pred: "inputs",
        get_experts_with_topk_ids=lambda prefix, topk_ids, topk_ids_pred: (w13, w2),
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method.BackendExpertManager",
        lambda: fake_manager,
    )

    layer = types.SimpleNamespace(
        prefix="layer",
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=False,
        global_num_experts=4,
        expert_map=torch.arange(4),
    )
    x = torch.randn(2, 8)
    topk_weights = torch.randn(2, 2)
    topk_ids = torch.tensor([[0, 1], [2, 3]])

    actual = UnquantizedFusedMoEMethod.forward_cuda(
        method,
        layer,
        x,
        topk_weights,
        topk_ids,
        [],
        None,
    )

    assert torch.equal(actual, expected)
    assert len(legacy_kernel.calls) == 1
    assert len(offloaded_kernel.calls) == 0


def test_forward_cuda_uses_offloaded_kernel_with_shared_experts_overlap(
    monkeypatch,
):
    method = _make_method()
    expected = torch.randn(2, 8)
    legacy_kernel = _FakeKernel(torch.zeros_like(expected))
    offloaded_kernel = _FakeKernel(expected)
    method.kernel = legacy_kernel
    method.offloaded_kernel = offloaded_kernel

    fake_manager = types.SimpleNamespace(
        no_copy_compute_enabled=lambda: True,
        get_offloaded_compute_inputs=lambda prefix, topk_ids, topk_ids_pred: "inputs",
        get_experts_with_topk_ids=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("copy path should not be used")
        ),
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method.BackendExpertManager",
        lambda: fake_manager,
    )

    shared_experts = object()
    layer = types.SimpleNamespace(
        prefix="layer",
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=False,
        global_num_experts=4,
        expert_map=None,
        shared_experts=shared_experts,
    )
    x = torch.randn(2, 8)
    topk_weights = torch.randn(2, 2)
    topk_ids = torch.tensor([[0, 1], [2, 3]])
    shared_experts_input = torch.randn(2, 8)

    actual = UnquantizedFusedMoEMethod.forward_cuda(
        method,
        layer,
        x,
        topk_weights,
        topk_ids,
        [],
        shared_experts_input,
    )

    assert torch.equal(actual, expected)
    assert len(offloaded_kernel.calls) == 1
    assert offloaded_kernel.calls[0]["shared_experts_input"] is shared_experts_input
    assert "shared_experts" not in offloaded_kernel.calls[0]
    assert len(legacy_kernel.calls) == 0
