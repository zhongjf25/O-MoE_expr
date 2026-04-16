CUDA_VISIBLE_DEVICES=6,7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=/root/autodl-tmp/workspace/ldy/vllm-offloading:$PYTHONPATH \
vllm serve /root/autodl-tmp/models/Qwen3-30B-A3B \
    --port 8001 \
    --gpu-memory-utilization 0.93 \
    --offload-expert \
    --offload-expert-limit 20 \
    --tensor-parallel-size 2 \
    --enforce-eager \
    --max-model-len 16384 \
    --dynamic-cache-enabled \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --expert-no-copy-compute \
    --no-async-scheduling \
    --expert-numa-binding