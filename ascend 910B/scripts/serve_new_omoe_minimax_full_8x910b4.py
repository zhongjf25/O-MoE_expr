from __future__ import annotations

import os
import sys

MODEL = os.getenv("MODEL_PATH", "/home/ma-user/work/models/minimax-m2.7")
SERVED_MODEL = os.getenv("SERVED_MODEL_NAME", "MiniMax-M2.7")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

os.environ["DS_EXPERT_OFFLOAD"] = "1"
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMOE_W8A8_COMPUTE", "ascendc_discrete")

from vllm_ascend.expert_offload.patch import install_expert_offload_patch

install_expert_offload_patch()

from vllm.entrypoints.cli.main import main


def run() -> None:
    cached_num_experts = os.getenv("CACHED_NUM_EXPERTS", "8")
    offload_expert_limit = os.getenv("OFFLOAD_EXPERT_LIMIT", "248")
    gpu_util = os.getenv("GPU_MEMORY_UTIL", "0.85")
    max_num_seqs = os.getenv("MAX_NUM_SEQS")
    sys.argv = [
        "vllm",
        "serve",
        MODEL,
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--served-model-name",
        SERVED_MODEL,
        "--tensor-parallel-size",
        "8",
        "--gpu-memory-utilization",
        gpu_util,
        "--max-model-len",
        os.getenv("MAX_MODEL_LEN", "4096"),
        "--quantization",
        "ascend",
        "--enforce-eager",
        "--trust-remote-code",
        "--cached-num-experts",
        cached_num_experts,
        "--offload-expert-limit",
        offload_expert_limit,
        "--dynamic-cache-enabled",
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--no-async-scheduling",
    ]
    if max_num_seqs:
        sys.argv.extend(["--max-num-seqs", max_num_seqs])
    print(
        "[new-omoe-minimax-full-8x910b4] "
        f"model={MODEL} tp=8 util={gpu_util} "
        f"cached={cached_num_experts} offload_limit={offload_expert_limit} "
        f"max_num_seqs={max_num_seqs or 'default'} "
        f"w8a8_compute={os.environ['OMOE_W8A8_COMPUTE']} port={PORT}",
        flush=True,
    )
    main()


if __name__ == "__main__":
    run()
