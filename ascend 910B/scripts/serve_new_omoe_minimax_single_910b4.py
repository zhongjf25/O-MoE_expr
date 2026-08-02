from __future__ import annotations

import json
import os
import sys

MODEL = os.getenv("MODEL_PATH", "/home/ma-user/work/models/minimax-m2.7")
SERVED_MODEL = os.getenv("SERVED_MODEL_NAME", "MiniMax-M2.7")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
GPU_UTIL = float(os.getenv("GPU_MEMORY_UTIL", "0.90"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "2048"))
CACHED_NUM_EXPERTS = int(os.getenv("CACHED_NUM_EXPERTS", "8"))
OFFLOAD_EXPERT_LIMIT = int(os.getenv("OFFLOAD_EXPERT_LIMIT", "248"))

os.environ["DS_EXPERT_OFFLOAD"] = "1"
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMOE_W8A8_COMPUTE", "ascendc_discrete")

from vllm_ascend.expert_offload.patch import install_expert_offload_patch

install_expert_offload_patch()

from vllm.entrypoints.cli.main import main

def run() -> None:
    additional_config = {
        "expert_hotset_collect_mode": True,
        "expert_hotset_use_configured_cache": True,
    }
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
        "1",
        "--gpu-memory-utilization",
        str(GPU_UTIL),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--quantization",
        "ascend",
        "--enforce-eager",
        "--trust-remote-code",
        "--cached-num-experts",
        str(CACHED_NUM_EXPERTS),
        "--offload-expert-limit",
        str(OFFLOAD_EXPERT_LIMIT),
        "--dynamic-cache-enabled",
        "--additional-config",
        json.dumps(additional_config),
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--no-async-scheduling",
    ]
    print(
        "[new-omoe-minimax] "
        f"model={MODEL} tp=1 util={GPU_UTIL} max_len={MAX_MODEL_LEN} "
        f"cached={CACHED_NUM_EXPERTS} offload_limit={OFFLOAD_EXPERT_LIMIT} "
        f"w8a8_compute={os.environ['OMOE_W8A8_COMPUTE']} port={PORT}",
        flush=True,
    )
    main()


if __name__ == "__main__":
    run()
