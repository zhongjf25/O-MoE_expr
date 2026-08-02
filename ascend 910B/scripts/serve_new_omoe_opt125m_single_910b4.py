from __future__ import annotations

import os
import sys

MODEL = os.getenv("MODEL_PATH", "/home/ma-user/work/models/opt-125m")
SERVED_MODEL = os.getenv("SERVED_MODEL_NAME", "opt-125m")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMOE_W8A8_COMPUTE", "ascendc_discrete")

from vllm_ascend.expert_offload.patch import install_expert_offload_patch

install_expert_offload_patch()

from vllm.entrypoints.cli.main import main


def run() -> None:
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
        os.getenv("GPU_MEMORY_UTIL", "0.80"),
        "--max-model-len",
        os.getenv("MAX_MODEL_LEN", "1024"),
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--no-async-scheduling",
    ]
    print(
        "[new-omoe-opt125m] "
        f"model={MODEL} tp=1 w8a8_compute={os.environ['OMOE_W8A8_COMPUTE']} "
        f"port={PORT}",
        flush=True,
    )
    main()


if __name__ == "__main__":
    run()
