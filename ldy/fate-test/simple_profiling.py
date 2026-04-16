# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time

from vllm import LLM, SamplingParams

# enable torch profiler, can also be set on cmd line
os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"
os.environ["DS_EXPERT_OFFLOAD"] = "1"
os.environ["DS_CACHED_EXPERTS_COUNT"] = "20"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def main():
    # Create an LLM.
    # model = "/root/workspace/Fate_before_hw/model_weights/qwen1.5-moe-a2.7b/weights"
    # model = "/root/workspace/model_weights/qwen1.5-moe-a2.7b"
    model = "/data/share/models/Qwen3-30B-A3B-Instruct-2507"
    llm = LLM(model=model, tensor_parallel_size=2, gpu_memory_utilization=0.7, max_model_len=512, enforce_eager=True, max_num_seqs=2)

    # llm.start_profile()

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)

    # llm.stop_profile()

    # Print the outputs.
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)

    # Add a buffer to wait for profiler in the background process
    # (in case MP is on) to finish writing profiling output.
    time.sleep(10)


if __name__ == "__main__":
    main()