CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/root/autodl-tmp/workspace/fate:$PYTHONPATH vllm serve /root/autodl-tmp/models/Qwen1.5-MoE-A2.7B \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 1 \
    --enforce-eager \
    --offload-expert \
    --cached-num-experts 60 \
    --offload-expert-limit 15 \
    --dynamic-cache-enabled \
    --no-enable-prefix-caching