# Build the matching hotset profile first:
# python simple_build_expert_hotset_profile.py

PYTHONPATH=/root/workspace/mycode/vllm-offloading:$PYTHONPATH vllm serve /root/workspace/model_weights/qwen1.5-moe-a2.7b \
    --gpu-memory-utilization 0.7 \
    --enforce-eager \
    --offload-expert \
    --cached-num-experts 55 \
    --offload-expert-limit 30 \
    --dynamic-cache-enabled \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --no-async-scheduling