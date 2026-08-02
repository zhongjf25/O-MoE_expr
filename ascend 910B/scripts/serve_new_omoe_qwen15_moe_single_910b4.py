from __future__ import annotations

import json
import os
import sys

MODEL = os.getenv(
    "MODEL_PATH", "/home/ma-user/work/models/Qwen1.5-MoE-A2.7B"
)
SERVED_MODEL = os.getenv("SERVED_MODEL_NAME", "qwen1.5-moe-a2.7b")
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
    cached_num_experts = os.getenv("CACHED_NUM_EXPERTS", "10")
    offload_expert_limit = os.getenv("OFFLOAD_EXPERT_LIMIT", "50")
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
        os.getenv("GPU_MEMORY_UTIL", "0.90"),
        "--max-model-len",
        os.getenv("MAX_MODEL_LEN", "1024"),
        "--block-size",
        os.getenv("BLOCK_SIZE", "64"),
        "--enforce-eager",
        "--trust-remote-code",
        "--cached-num-experts",
        cached_num_experts,
        "--offload-expert-limit",
        offload_expert_limit,
        "--dynamic-cache-enabled",
        "--additional-config",
        json.dumps(additional_config),
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--no-async-scheduling",
    ]
    print(
        "[new-omoe-qwen15-moe] "
        f"model={MODEL} tp=1 util={os.getenv('GPU_MEMORY_UTIL', '0.90')} "
        f"cached={cached_num_experts} offload_limit={offload_expert_limit} "
        f"w8a8_compute={os.environ['OMOE_W8A8_COMPUTE']} port={PORT}",
        flush=True,
    )
    main()


if __name__ == "__main__":
    run()
