# Build the matching hotset profile first:
# python simple_build_expert_hotset_profile.py

export CUDA_VISIBLE_DEVICES=3,4
PYTHONPATH=/root/autodl-tmp/workspace/vllm-offloading:$PYTHONPATH vllm serve /root/autodl-tmp/models/Qwen3-30B-A3B \
    --gpu-memory-utilization 0.7 \
    --enforce-eager \
    --offload-expert \
    --cached-num-experts 55 \
    --offload-expert-limit 40 \
    --tensor-parallel-size 2 \
    --max-model-len 4096 \
    --dynamic-cache-enabled \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --expert-no-copy-compute \
    --no-async-scheduling \
    --expert-numa-binding \
    --profiler-config '{"profiler":"torch","torch_profiler_dir":"/root/workspace/mycode/vllm-offloading/vllm_profile_bench", "delay_iterations":"50", "max_iterations":"100"}'