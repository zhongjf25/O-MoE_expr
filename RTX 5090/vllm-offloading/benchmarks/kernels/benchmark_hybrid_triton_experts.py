# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# eval "$(conda shell.bash hook)"
# conda activate vllmnew
# python benchmarks/kernels/benchmark_hybrid_triton_experts.py \
#   --num-tokens 512 \
#   --hidden-size 7168 \
#   --intermediate-size 2048 \
#   --num-experts 16 \
#   --topk 2 \
#   --cache-hit-ratio 0.5 \
#   --mode both


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
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.fused_moe import (
    HybridTritonExperts,
    TritonExperts,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser


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

    return OffloadedExpertComputeInputs(
        w13_weight_comp=w13_weight_comp,
        w2_weight_comp=w2_weight_comp,
        w13_blocks=w13_blocks,
        w2_blocks=w2_blocks,
        expert_source=expert_source,
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
    latencies_ms = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        latencies_ms.append(start.elapsed_time(end))
    return sum(latencies_ms) / len(latencies_ms) * 1000.0


def allocate_buffers(
    experts: TritonExperts,
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: MoEActivation,
    global_num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    local_num_experts, num_tokens, n_dim, k_dim, top_k = experts.moe_problem_size(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
    )
    assert local_num_experts == w13_weight.size(0)

    workspace_dtype = experts.workspace_dtype(hidden_states.dtype)
    workspace13_shape, workspace2_shape, output_shape = experts.workspace_shapes(
        num_tokens,
        n_dim,
        k_dim,
        top_k,
        global_num_experts,
        local_num_experts,
        expert_tokens_meta=None,
        activation=activation,
    )

    output = torch.empty(output_shape, dtype=hidden_states.dtype, device=hidden_states.device)
    workspace13 = torch.empty(
        workspace13_shape,
        dtype=workspace_dtype,
        device=hidden_states.device,
    )
    workspace2 = torch.empty(
        workspace2_shape,
        dtype=workspace_dtype,
        device=hidden_states.device,
    )
    return output, workspace13, workspace2


def parse_dtype(dtype_name: str) -> torch.dtype:
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map[dtype_name]


@torch.inference_mode()
def main() -> None:
    parser = FlexibleArgumentParser(
        description=(
            "Benchmark TritonExperts.apply() vs "
            "HybridTritonExperts.apply_offloaded()."
        )
    )
    parser.add_argument("--num-tokens", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--cache-hit-ratio", type=float, default=0.5)
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--mode", choices=["triton", "hybrid", "both"], default="both")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-output-check", action="store_true")
    args = parser.parse_args()

    assert 0.0 <= args.cache_hit_ratio <= 1.0, (
        f"cache_hit_ratio must be in [0, 1], got {args.cache_hit_ratio}"
    )

    dtype = parse_dtype(args.dtype)
    torch.manual_seed(args.seed)

    hidden_states = torch.randn(
        args.num_tokens,
        args.hidden_size,
        dtype=dtype,
        device="cuda",
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
        args.num_tokens,
        args.topk,
        dtype=torch.float32,
        device="cuda",
    )
    topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)
    topk_ids = torch.randint(
        0,
        args.num_experts,
        (args.num_tokens, args.topk),
        device="cuda",
    )

    num_cache = min(args.num_experts, int(args.num_experts * args.cache_hit_ratio))
    cache_experts = set(range(num_cache))
    expert_inputs = make_offloaded_inputs(w13_weight, w2_weight, cache_experts)

    moe_config = make_moe_config(
        args.num_experts,
        args.topk,
        args.hidden_size,
        args.intermediate_size,
        dtype,
    )
    triton_experts = TritonExperts(moe_config, FUSED_MOE_UNQUANTIZED_CONFIG)
    hybrid_experts = HybridTritonExperts(moe_config, FUSED_MOE_UNQUANTIZED_CONFIG)

    activation = MoEActivation.SILU
    triton_output, triton_workspace13, triton_workspace2 = allocate_buffers(
        triton_experts,
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        activation,
        args.num_experts,
    )
    hybrid_output, hybrid_workspace13, hybrid_workspace2 = allocate_buffers(
        hybrid_experts,
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        activation,
        args.num_experts,
    )

    with set_current_vllm_config(VllmConfig()):
        triton_us: float | None = None
        hybrid_us: float | None = None

        def run_triton() -> torch.Tensor:
            triton_experts.apply(
                output=triton_output,
                hidden_states=hidden_states,
                w1=w13_weight,
                w2=w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                global_num_experts=args.num_experts,
                expert_map=None,
                a1q_scale=None,
                a2_scale=None,
                workspace13=triton_workspace13,
                workspace2=triton_workspace2,
                expert_tokens_meta=None,
                apply_router_weight_on_input=False,
            )
            return triton_output

        def run_hybrid() -> torch.Tensor:
            hybrid_experts.apply_offloaded(
                output=hybrid_output,
                hidden_states=hidden_states,
                expert_inputs=expert_inputs,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                global_num_experts=args.num_experts,
                expert_map=None,
                a1q_scale=None,
                a2_scale=None,
                workspace13=hybrid_workspace13,
                workspace2=hybrid_workspace2,
                apply_router_weight_on_input=False,
            )
            return hybrid_output

        if not args.skip_output_check:
            torch.testing.assert_close(
                run_hybrid(),
                run_triton(),
                atol=2e-2,
                rtol=0,
            )

        if args.mode in ("triton", "both"):
            triton_us = benchmark(run_triton, args.warmup_iters, args.iters)
            print(
                "mode=triton "
                f"dtype={args.dtype} "
                f"cache_hit_ratio={args.cache_hit_ratio:.2f} "
                f"latency_us={triton_us:.2f}"
            )

        if args.mode in ("hybrid", "both"):
            hybrid_us = benchmark(run_hybrid, args.warmup_iters, args.iters)
            print(
                "mode=hybrid "
                f"dtype={args.dtype} "
                f"cache_hit_ratio={args.cache_hit_ratio:.2f} "
                f"latency_us={hybrid_us:.2f}"
            )

        if triton_us is not None and hybrid_us is not None:
            print(
                "summary "
                f"hybrid_vs_triton_speedup={triton_us / hybrid_us:.4f}"
            )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")
    start = time.time()
    main()
    print(f"elapsed_s={time.time() - start:.2f}")
