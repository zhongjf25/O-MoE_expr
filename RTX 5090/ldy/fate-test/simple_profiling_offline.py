#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import os
import time

from vllm import LLM, SamplingParams

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRITON_CACHE_AUTOTUNING"] = "1"
os.environ["DS_EXPERT_OFFLOAD"] = "1"
os.environ["DS_CACHED_EXPERTS_COUNT"] = "230"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# MODEL_PATH = "/root/autodl-tmp/models/Qwen3-30B-A3B"
MODEL_PATH = "/root/autodl-tmp/models/Qwen3.5-35B-A3B"
PROFILE_DIR = "/root/autodl-tmp/workspace/ldy/vllm-offloading/vllm_profile_offline"

PROMPTS = [
    "Hello, my name is",
]

SAMPLING_PARAMS = SamplingParams(temperature=0, top_p=1)


def main() -> None:
    llm = LLM(
        model=MODEL_PATH,
        gpu_memory_utilization=0.77,
        max_model_len=16384,
        enforce_eager=True,
        tensor_parallel_size=2,
        enable_prefix_caching=False,
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": PROFILE_DIR,
            "torch_profiler_with_stack": True,
            "torch_profiler_with_memory": True,
        },
    )

    outputs = []
    try:
        llm.start_profile("offline_run")
        outputs = llm.generate(PROMPTS, SAMPLING_PARAMS)
    finally:
        # Ensure traces are flushed even if generation raises.
        llm.stop_profile()
        # Leave a short buffer for async trace writing.
        time.sleep(3)

    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)

    print(f"Profiler outputs should be under: {PROFILE_DIR}")


if __name__ == "__main__":
    main()
