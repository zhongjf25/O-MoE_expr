# Build the matching hotset profile first:
# python simple_build_expert_hotset_profile.py

export CUDA_VISIBLE_DEVICES=0,1
PYTHONPATH=/root/autodl-tmp/workspace/vllm-offloading:$PYTHONPATH vllm serve /root/autodl-tmp/models/Qwen3.5-35B-A3B \
    --gpu-memory-utilization 0.7 \
    --tensor-parallel-size 2 \
    --enforce-eager \
    --offload-expert \
    --max-model-len 32768 \
    --offload-expert-limit 80 \
    --dynamic-cache-enabled \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --expert-no-copy-compute \
    --no-async-scheduling
