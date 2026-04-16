# SPDX-License-Identifier: Apache-2.0
# eval "$(conda shell.bash hook)" && conda activate vllmnew
# python benchmarks/kernels/benchmark_offloaded_expert_copy.py \
#   --num-experts 8 \
#   --rows 512 \
#   --cols 512

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vllm import _custom_ops as _custom_ops  # noqa: F401
from vllm.model_executor.offloaded_expert_copy_extension import (
    ensure_offloaded_expert_copy_op_loaded,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

DTYPE_CHOICES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _measure_ms(fn, *, warmup_iters: int, iters: int) -> float:
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    total_ms = 0.0
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        total_ms += start.elapsed_time(end)
    return total_ms / iters


def _make_sources(
    *,
    num_experts: int,
    rows: int,
    cols: int,
    dtype: torch.dtype,
    src_backend: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (num_experts, rows, cols)
    if src_backend == "uva":
        src1_cpu = torch.randn(shape, dtype=dtype).pin_memory()
        src2_cpu = torch.randn(shape, dtype=dtype).pin_memory()
        return (
            get_accelerator_view_from_cpu_tensor(src1_cpu),
            get_accelerator_view_from_cpu_tensor(src2_cpu),
        )
    if src_backend == "cuda":
        return (
            torch.randn(shape, dtype=dtype, device="cuda"),
            torch.randn(shape, dtype=dtype, device="cuda"),
        )
    raise ValueError(f"Unsupported src backend: {src_backend}")


def _loop_copy(
    dst1: torch.Tensor,
    dst2: torch.Tensor,
    src1: torch.Tensor,
    src2: torch.Tensor,
    expert_ids: list[int],
) -> None:
    for expert_id in expert_ids:
        dst1[expert_id].copy_(src1[expert_id])
        dst2[expert_id].copy_(src2[expert_id])


def _payload_gbps(
    *,
    num_experts: int,
    rows: int,
    cols: int,
    dtype: torch.dtype,
    latency_ms: float,
) -> float:
    payload_bytes = 2 * num_experts * rows * cols * torch.tensor(
        [], dtype=dtype
    ).element_size()
    return payload_bytes / (latency_ms * 1e-3) / 1e9


def main() -> None:
    parser = FlexibleArgumentParser(
        description="Benchmark the offloaded expert copy op against a loop copy baseline."
    )
    parser.add_argument("--num-experts", type=int, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPE_CHOICES),
        default="bfloat16",
    )
    parser.add_argument(
        "--src-backend",
        choices=("uva", "cuda"),
        default="uva",
        help="`uva` matches the real miss path; `cuda` isolates device copy throughput.",
    )
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()

    if args.num_experts <= 0:
        raise ValueError("--num-experts must be positive.")
    if args.rows <= 0 or args.cols <= 0:
        raise ValueError("--rows and --cols must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    ensure_offloaded_expert_copy_op_loaded(verbose=args.verbose_build)

    dtype = DTYPE_CHOICES[args.dtype]
    src1, src2 = _make_sources(
        num_experts=args.num_experts,
        rows=args.rows,
        cols=args.cols,
        dtype=dtype,
        src_backend=args.src_backend,
    )
    dst_op_1 = torch.empty((args.num_experts, args.rows, args.cols),
                           dtype=dtype,
                           device="cuda")
    dst_op_2 = torch.empty_like(dst_op_1)
    dst_loop_1 = torch.empty_like(dst_op_1)
    dst_loop_2 = torch.empty_like(dst_op_1)
    expert_ids_gpu = torch.arange(args.num_experts, dtype=torch.long, device="cuda")
    expert_ids_list = list(range(args.num_experts))

    def run_custom_op() -> None:
        torch.ops._C.copy_uncached_experts_to_comp(
            dst_op_1,
            dst_op_2,
            src1,
            src2,
            expert_ids_gpu,
        )

    def run_loop_copy() -> None:
        _loop_copy(
            dst_loop_1,
            dst_loop_2,
            src1,
            src2,
            expert_ids_list,
        )

    op_ms = _measure_ms(
        run_custom_op,
        warmup_iters=args.warmup_iters,
        iters=args.iters,
    )
    loop_ms = _measure_ms(
        run_loop_copy,
        warmup_iters=args.warmup_iters,
        iters=args.iters,
    )

    torch.testing.assert_close(dst_op_1, dst_loop_1)
    torch.testing.assert_close(dst_op_2, dst_loop_2)

    op_gbps = _payload_gbps(
        num_experts=args.num_experts,
        rows=args.rows,
        cols=args.cols,
        dtype=dtype,
        latency_ms=op_ms,
    )
    loop_gbps = _payload_gbps(
        num_experts=args.num_experts,
        rows=args.rows,
        cols=args.cols,
        dtype=dtype,
        latency_ms=loop_ms,
    )

    print(
        f"n={args.num_experts} shape=[{args.rows},{args.cols}] "
        f"dtype={args.dtype} src_backend={args.src_backend}"
    )
    print(f"custom_op  : {op_ms:.4f} ms  {op_gbps:.2f} GB/s")
    print(f"loop_copy  : {loop_ms:.4f} ms  {loop_gbps:.2f} GB/s")
    print(f"speedup    : {loop_ms / op_ms:.2f}x")


if __name__ == "__main__":
    main()
