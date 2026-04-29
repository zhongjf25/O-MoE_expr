CUDA_VISIBLE_DEVICES=7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=/root/autodl-tmp/workspace/ldy/vllm:$PYTHONPATH \
vllm serve /root/autodl-tmp/models/Qwen1.5-MoE-A2.7B \
    --port 8001 \
    --gpu-memory-utilization 0.935 \
    --tensor-parallel-size 1 \
    --enforce-eager \
    --max-model-len 8192 \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --log-kv-allocation-per-step \
    --kv-allocation-csv-path vllm_kv_allocation.csv