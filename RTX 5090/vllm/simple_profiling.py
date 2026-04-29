# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time

from vllm import LLM, SamplingParams

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    # "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=16)


def main():
    # Create an LLM.
    # model = "/data/share/models/qwen3-30b"
    model = "/root/autodl-tmp/models/qwen35_122b"
    model = "/root/autodl-tmp/models/Qwen3-30B-A3B"
    llm = LLM(model=model, gpu_memory_utilization=0.95,
              tensor_parallel_size=8,
              max_model_len=2048, 
              max_num_batched_tokens=512,
              enforce_eager=True, 
              enable_prefix_caching=False,
              # enable_chunked_prefill=False,
    )

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)

    # Add a buffer to wait for profiler in the background process
    # (in case MP is on) to finish writing profiling output.


if __name__ == "__main__":
    main()
