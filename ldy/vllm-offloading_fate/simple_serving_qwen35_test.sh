CUDA_VISIBLE_DEVICES=0,1 \
DS_EXPERT_OFFLOAD=1 \
DS_CACHED_EXPERTS_COUNT=230 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=/root/autodl-tmp/workspace/ldy/vllm-offloading_fate:$PYTHONPATH \
vllm serve /root/autodl-tmp/models/Qwen3.5-35B-A3B \
    --port 8001 \
    --gpu-memory-utilization 0.76 \
    --tensor-parallel-size 2 \
    --enforce-eager \
    --max-model-len 16384 \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \