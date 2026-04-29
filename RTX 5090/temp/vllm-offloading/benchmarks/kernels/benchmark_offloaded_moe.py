# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.worker.workspace import init_workspace_manager


def make_moe_config(
    num_experts: int,
    topk: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
) -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=topk,
        hidden_dim=hidden_size,
        intermediate_size_per_partition=intermediate_size,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=dtype,
        device="cuda",
        routing_method=RoutingMethodType.TopK,
    )


def make_offloaded_inputs(
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    cache_experts: set[int],
) -> OffloadedExpertComputeInputs:
    device = w13_weight.device
    num_experts, w13_rows, hidden_size = w13_weight.shape
    intermediate_size = w13_rows // 2
    invalid = OffloadedExpertComputeInputs.SOURCE_INVALID

    w13_weight_comp = torch.empty_like(w13_weight)
    w2_weight_comp = torch.empty_like(w2_weight)
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
    expert_source = torch.full((num_experts,), invalid, dtype=torch.int32, device=device)
    cache_w1_block_ids = torch.full_like(expert_source, invalid)
    cache_w2_block_ids = torch.full_like(expert_source, invalid)
    cache_w3_block_ids = torch.full_like(expert_source, invalid)
    # Maps global expert id -> row in w13_weight_comp / w2_weight_comp for SOURCE_COMP.
    # Benchmark uses a full-size comp buffer (one row per expert), so use identity;
    # SOURCE_CACHE experts never read comp rows (see Triton kernel branch source==1).
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


def benchmark(fn, warmup_iters: int, iters: int) -> float:
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    latencies = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        latencies.append(start.elapsed_time(end))
    return sum(latencies) / len(latencies) * 1000.0


def main():
    init_workspace_manager(torch.device("cuda:0"))

    parser = FlexibleArgumentParser(
        description="Benchmark copy vs hybrid no-copy offloaded MoE Triton kernels."
    )
    parser.add_argument("--num-tokens", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--cache-hit-ratio", type=float, default=0.5)
    parser.add_argument("--mode", choices=["copy", "hybrid", "both"], default="both")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup-iters", type=int, default=10)
    args = parser.parse_args()

    dtype = torch.bfloat16
    torch.manual_seed(0)
    hidden_states = torch.randn(
        args.num_tokens, args.hidden_size, dtype=dtype, device="cuda"
    )
    w13_weight = torch.randn(
        args.num_experts,
        2 * args.intermediate_size,
        args.hidden_size,
        dtype=dtype,
        device="cuda",
    ) / 10
    w2_weight = torch.randn(
        args.num_experts,
        args.hidden_size,
        args.intermediate_size,
        dtype=dtype,
        device="cuda",
    ) / 10
    topk_weights = torch.rand(
        args.num_tokens, args.topk, dtype=torch.float32, device="cuda"
    )
    topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)
    topk_ids = torch.randint(
        0, args.num_experts, (args.num_tokens, args.topk), device="cuda"
    )

    num_cache = int(args.num_experts * args.cache_hit_ratio)
    cache_experts = set(range(num_cache))
    expert_inputs = make_offloaded_inputs(w13_weight, w2_weight, cache_experts)
    hybrid_kernel = OffloadedTritonKernel(
        moe_config=make_moe_config(
            args.num_experts,
            args.topk,
            args.hidden_size,
            args.intermediate_size,
            dtype,
        ),
        quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
        inplace=False,
    )

    with set_current_vllm_config(VllmConfig()):
        def run_copy():
            return fused_experts(
                hidden_states,
                w13_weight,
                w2_weight,
                topk_weights,
                topk_ids,
                quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
                activation=MoEActivation.SILU,
            )

        def run_hybrid():
            return hybrid_kernel(
                hidden_states=hidden_states,
                expert_inputs=expert_inputs,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=MoEActivation.SILU,
                global_num_experts=args.num_experts,
            )

        if args.mode in ("copy", "both"):
            copy_us = benchmark(run_copy, args.warmup_iters, args.iters)
            print(
                f"mode=copy cache_hit_ratio={args.cache_hit_ratio:.2f} latency_us={copy_us:.2f}"
            )

        if args.mode in ("hybrid", "both"):
            hybrid_us = benchmark(run_hybrid, args.warmup_iters, args.iters)
            print(
                f"mode=hybrid cache_hit_ratio={args.cache_hit_ratio:.2f} latency_us={hybrid_us:.2f}"
            )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")
    start = time.time()
    main()
    print(f"elapsed_s={time.time() - start:.2f}")
