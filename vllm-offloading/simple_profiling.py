# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time

from vllm import LLM, SamplingParams

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
# os.environ["VLLM_BATCH_INVARIANT"] = "1"
# os.environ["TRITON_CACHE_AUTOTUNING"] = "1"
# os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

# os.environ["VLLM_MEMORY_TRACE"] = "1"
# os.environ["VLLM_MEMORY_TRACE_OUTPUT"] = "my_trace.csv"
# python scripts/plot_memory_trace.py --input my_trace.csv --output memory_snapshot.png

# Sample prompts.
prompts = [
    "Hello, my name is",
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=16)


def main():
    # Create an LLM.
    # model = "/root/autodl-tmp/models/Qwen3.5-35B-A3B"
    # model = "/root/autodl-tmp/models/qwen35_122b"
    model = "/root/autodl-tmp/models/Qwen3-30B-A3B"
    llm = LLM(model=model, gpu_memory_utilization=0.85,
              max_model_len=8192, 
              enforce_eager=True, 
              tensor_parallel_size=2,
              enable_prefix_caching=False,
              enable_chunked_prefill=False,
              offload_expert=True,
              cached_num_experts=60,
              offload_expert_limit=100,
              enable_dynamic_cache=False,
              expert_no_copy_compute=False,
              async_scheduling=False,
            #   attention_backend='FLASH_ATTN'
              profiler_config={
                "profiler": "torch",
                "torch_profiler_dir": "./vllm_profile",
              },
              )
    # outputs = llm.generate(prompts, sampling_params)
    llm.start_profile()

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)

    llm.stop_profile()

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
