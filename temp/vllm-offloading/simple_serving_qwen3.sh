# Build the matching hotset profile first:
# python simple_build_expert_hotset_profile.py

export CUDA_VISIBLE_DEVICES=3,4
PYTHONPATH=/root/autodl-tmp/workspace/vllm-offloading:$PYTHONPATH vllm serve /root/autodl-tmp/models/Qwen3-30B-A3B \
    --gpu-memory-utilization 0.7 \
    --enforce-eager \
    --offload-expert \
    --offload-expert-limit 40 \
    --dynamic-cache-enabled \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --expert-no-copy-compute \
    --no-async-scheduling