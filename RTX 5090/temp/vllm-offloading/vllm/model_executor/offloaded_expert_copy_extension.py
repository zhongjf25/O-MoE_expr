# SPDX-License-Identifier: Apache-2.0
# python -m vllm.model_executor.offloaded_expert_copy_extension --verbose

from __future__ import annotations

import argparse
from pathlib import Path

import torch

COPY_OP_NAME = "copy_uncached_experts_to_comp"
PREPARE_OP_NAME = "prepare_offloaded_compute_inputs"
PREPARE_AND_COPY_OP_NAME = "prepare_and_copy_offloaded_compute_inputs"
PREPARE_AND_COPY_COMPACT_OP_NAME = "prepare_and_copy_offloaded_compute_inputs_compact"
EXTENSION_NAME = "vllm_offloaded_expert_copy_ext"


def offloaded_expert_copy_ops_are_available() -> bool:
    return (
        hasattr(torch.ops._C, COPY_OP_NAME)
        and hasattr(torch.ops._C, PREPARE_OP_NAME)
        and hasattr(torch.ops._C, PREPARE_AND_COPY_OP_NAME)
        and hasattr(torch.ops._C, PREPARE_AND_COPY_COMPACT_OP_NAME)
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_offloaded_expert_copy_op_loaded(
    *,
    build_directory: str | Path | None = None,
    verbose: bool = False,
) -> bool:
    """Build and load the minimal extension if the op is not available."""
    if offloaded_expert_copy_ops_are_available():
        return False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to build this extension.")

    from torch.utils.cpp_extension import load

    repo_root = _repo_root()
    csrc_dir = repo_root / "csrc"
    load_kwargs = {
        "name": EXTENSION_NAME,
        "sources": [
            str(csrc_dir / "offloaded_expert_copy_jit_binding.cpp"),
            str(csrc_dir / "offloaded_expert_copy.cu"),
        ],
        "extra_include_paths": [str(csrc_dir)],
        "extra_cflags": ["-O2", "-std=c++17"],
        "extra_cuda_cflags": ["-O2", "--threads=1", "-std=c++17"],
        "with_cuda": True,
        "is_python_module": False,
        "verbose": verbose,
    }
    if build_directory is not None:
        build_path = Path(build_directory).expanduser().resolve()
        build_path.mkdir(parents=True, exist_ok=True)
        load_kwargs["build_directory"] = str(build_path)

    load(**load_kwargs)

    if not offloaded_expert_copy_ops_are_available():
        raise RuntimeError(
            "The minimal extension finished loading but "
            f"`torch.ops._C.{COPY_OP_NAME}` or "
            f"`torch.ops._C.{PREPARE_OP_NAME}` or "
            f"`torch.ops._C.{PREPARE_AND_COPY_OP_NAME}` or "
            f"`torch.ops._C.{PREPARE_AND_COPY_COMPACT_OP_NAME}` "
            "is still unavailable."
        )
    return True


def _make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and load the minimal offloaded expert copy op."
    )
    parser.add_argument(
        "--build-directory",
        type=str,
        default=None,
        help="Optional build directory passed to torch.utils.cpp_extension.load.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print torch extension build output.",
    )
    return parser


def main() -> None:
    args = _make_arg_parser().parse_args()
    compiled = ensure_offloaded_expert_copy_op_loaded(
        build_directory=args.build_directory,
        verbose=args.verbose,
    )
    if compiled:
        print(
            "Built and loaded "
            f"torch.ops._C.{COPY_OP_NAME}, "
            f"torch.ops._C.{PREPARE_OP_NAME}, and "
            f"torch.ops._C.{PREPARE_AND_COPY_OP_NAME}, and "
            f"torch.ops._C.{PREPARE_AND_COPY_COMPACT_OP_NAME}"
        )
    else:
        print(
            "torch.ops._C."
            f"{COPY_OP_NAME}, torch.ops._C.{PREPARE_OP_NAME}, and "
            f"torch.ops._C.{PREPARE_AND_COPY_OP_NAME}, and "
            f"torch.ops._C.{PREPARE_AND_COPY_COMPACT_OP_NAME} "
            "are already available"
        )


if __name__ == "__main__":
    main()
