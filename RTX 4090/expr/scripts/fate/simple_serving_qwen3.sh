CUDA_VISIBLE_DEVICES=1,2 \
DS_EXPERT_OFFLOAD=1 \
DS_CACHED_EXPERTS_COUNT=50 \
PYTHONPATH=/root/workspace/ldy/vllm-offloading:$PYTHONPATH \
vllm serve /data/share/models/Qwen3-30B-A3B \
    --gpu-memory-utilization 0.89 \
    --tensor-parallel-size 2 \
    --enforce-eager \
    --max-model-len 8192 \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \