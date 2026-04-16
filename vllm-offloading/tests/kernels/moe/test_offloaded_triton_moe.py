# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import types

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.backend_expert_manager import OffloadedExpertComputeInputs
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    OffloadedTritonKernel,
    fused_experts,
)
from vllm.platforms import current_platform


def _make_offloaded_inputs(
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    cache_experts: set[int],
) -> OffloadedExpertComputeInputs:
    device = w13_weight.device
    num_experts, w13_rows, hidden_size = w13_weight.shape
    intermediate_size = w13_rows // 2
    source_invalid = OffloadedExpertComputeInputs.SOURCE_INVALID

    w13_weight_comp = torch.full_like(w13_weight, -321)
    w2_weight_comp = torch.full_like(w2_weight, -321)

    w13_blocks = torch.zeros(
        max(1, len(cache_experts) * 2),
        intermediate_size,
        hidden_size,
        dtype=w13_weight.dtype,
        device=device,
    )
    w2_blocks = torch.zeros(
        max(1, len(cache_experts)),
        w2_weight.size(1),
        w2_weight.size(2),
        dtype=w2_weight.dtype,
        device=device,
    )
    expert_source = torch.full((num_experts,), source_invalid, dtype=torch.int32, device=device)
    cache_w1_block_ids = torch.full_like(expert_source, source_invalid)
    cache_w2_block_ids = torch.full_like(expert_source, source_invalid)
    cache_w3_block_ids = torch.full_like(expert_source, source_invalid)
    comp_expert_to_slot = torch.full((num_experts,), -1, dtype=torch.int32, device=device)

    next_w13_block = 0
    next_w2_block = 0
    for expert_id in range(num_experts):
        if expert_id in cache_experts:
            w13_blocks[next_w13_block].copy_(w13_weight[expert_id, :intermediate_size])
            cache_w1_block_ids[expert_id] = next_w13_block
            next_w13_block += 1

            w13_blocks[next_w13_block].copy_(w13_weight[expert_id, intermediate_size:])
            cache_w3_block_ids[expert_id] = next_w13_block
            next_w13_block += 1

            w2_blocks[next_w2_block].copy_(w2_weight[expert_id])
            cache_w2_block_ids[expert_id] = next_w2_block
            next_w2_block += 1

            expert_source[expert_id] = OffloadedExpertComputeInputs.SOURCE_CACHE
        else:
            w13_weight_comp[expert_id].copy_(w13_weight[expert_id])
            w2_weight_comp[expert_id].copy_(w2_weight[expert_id])
            expert_source[expert_id] = OffloadedExpertComputeInputs.SOURCE_COMP
            comp_expert_to_slot[expert_id] = expert_id

    return OffloadedExpertComputeInputs(
        w13_weight_comp=w13_weight_comp,
        w2_weight_comp=w2_weight_comp,
        w13_blocks=w13_blocks,
        w2_blocks=w2_blocks,
        expert_source=expert_source,
        comp_expert_to_slot=comp_expert_to_slot,
        cache_w1_block_ids=cache_w1_block_ids,
        cache_w2_block_ids=cache_w2_block_ids,
        cache_w3_block_ids=cache_w3_block_ids,
    )


@pytest.mark.skipif(
    not current_platform.has_device_capability(80),
    reason="Requires compute capability >= 8.0",
)
@pytest.mark.parametrize(
    "cache_experts",
    [
        set(),
        {0, 1, 2, 3},
        {1, 3},
    ],
    ids=["all-comp", "all-cache", "mixed"],
)
@pytest.mark.parametrize("topk", [1, 2])
@torch.inference_mode()
def test_offloaded_triton_kernel_matches_copy_path(
    cache_experts: set[int],
    topk: int,
    workspace_init,
):
    num_experts = 4
    hidden_size = 64
    intermediate_size = 128
    num_tokens = 32
    dtype = torch.bfloat16

    hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda")
    w13_weight = torch.randn(
        num_experts,
        2 * intermediate_size,
        hidden_size,
        dtype=dtype,
        device="cuda",
    ) / 10
    w2_weight = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        dtype=dtype,
        device="cuda",
    ) / 10

    topk_weights = torch.rand(num_tokens, topk, dtype=torch.float32, device="cuda")
    topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)
    topk_ids = torch.randint(0, num_experts, (num_tokens, topk), device="cuda")

    expert_inputs = _make_offloaded_inputs(w13_weight, w2_weight, cache_experts)
    kernel = OffloadedTritonKernel(
        moe_config=make_dummy_moe_config(
            num_experts=num_experts,
            experts_per_token=topk,
            hidden_dim=hidden_size,
            intermediate_size_per_partition=intermediate_size,
            in_dtype=dtype,
        ),
        quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
        inplace=False,
    )

    with set_current_vllm_config(VllmConfig()):
        baseline = fused_experts(
            hidden_states,
            w13_weight,
            w2_weight,
            topk_weights,
            topk_ids,
            quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
            activation=MoEActivation.SILU,
        )
        actual = kernel(
            hidden_states=hidden_states,
            expert_inputs=expert_inputs,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=MoEActivation.SILU,
            global_num_experts=num_experts,
        )

    torch.testing.assert_close(actual, baseline, atol=2e-2, rtol=0)


def test_offloaded_triton_kernel_accepts_shared_experts_input():
    kernel = object.__new__(OffloadedTritonKernel)
    torch.nn.Module.__init__(kernel)
    kernel.inplace = False

    class _FakePrepareFinalize:
        def prepare(self, *args, **kwargs):
            hidden_states, topk_weights, topk_ids = args[:3]
            return hidden_states, None, None, topk_ids, topk_weights

        def finalize(self, output, fused_out, *args, **kwargs):
            output.copy_(fused_out)

    class _FakeExperts:
        expects_unquantized_inputs = False
        quant_config = object()

        @staticmethod
        def finalize_weight_and_reduce_impl():
            return object()

    kernel.prepare_finalize = _FakePrepareFinalize()
    kernel.fused_experts = _FakeExperts()
    kernel._fused_experts = types.MethodType(
        lambda self, **kwargs: torch.full_like(kwargs["a1q"], 7.0),
        kernel,
    )

    hidden_states = torch.randn(2, 4)
    shared_experts_input = torch.randn(2, 4)
    topk_weights = torch.ones(2, 1)
    topk_ids = torch.zeros(2, 1, dtype=torch.int64)
    expert_inputs = OffloadedExpertComputeInputs(
        w13_weight_comp=torch.empty(4, 8, 4),
        w2_weight_comp=torch.empty(4, 4, 4),
    )

    actual = kernel(
        hidden_states=hidden_states,
        expert_inputs=expert_inputs,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_experts_input=shared_experts_input,
    )

    torch.testing.assert_close(actual, torch.full_like(hidden_states, 7.0))
