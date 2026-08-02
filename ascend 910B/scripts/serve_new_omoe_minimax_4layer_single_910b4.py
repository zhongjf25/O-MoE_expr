from __future__ import annotations

import os
import re
import sys

MODEL = os.getenv("MODEL_PATH", "/home/ma-user/work/models/minimax-m2.7")
SERVED_MODEL = os.getenv("SERVED_MODEL_NAME", "MiniMax-M2.7-4layer")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

os.environ["DS_EXPERT_OFFLOAD"] = "1"
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMOE_W8A8_COMPUTE", "ascendc_discrete")

from vllm_ascend.expert_offload.patch import install_expert_offload_patch

install_expert_offload_patch()

from vllm.model_executor.models.minimax_m2 import MiniMaxM2ForCausalLM

_original_load_weights = MiniMaxM2ForCausalLM.load_weights
_layer_pattern = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _load_first_four_layers(self, weights):
    def selected_weights():
        skipped = 0
        for name, tensor in weights:
            match = _layer_pattern.search(name)
            if match is not None and int(match.group(1)) >= 4:
                skipped += 1
                continue
            yield name, tensor
        print(f"[minimax-4layer] skipped_later_layer_tensors={skipped}", flush=True)

    return _original_load_weights(self, selected_weights())


MiniMaxM2ForCausalLM.load_weights = _load_first_four_layers

from vllm.entrypoints.cli.main import main


def run() -> None:
    cached_num_experts = os.getenv("CACHED_NUM_EXPERTS", "8")
    offload_expert_limit = os.getenv("OFFLOAD_EXPERT_LIMIT", "248")
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
        os.getenv("GPU_MEMORY_UTIL", "0.70"),
        "--max-model-len",
        os.getenv("MAX_MODEL_LEN", "1024"),
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
    print(
        "[new-omoe-minimax-4layer] "
        f"model={MODEL} tp=1 util={os.getenv('GPU_MEMORY_UTIL', '0.70')} "
        f"cached={cached_num_experts} offload_limit={offload_expert_limit} "
        f"w8a8_compute={os.environ['OMOE_W8A8_COMPUTE']} port={PORT}",
        flush=True,
    )
    main()


if __name__ == "__main__":
    run()
