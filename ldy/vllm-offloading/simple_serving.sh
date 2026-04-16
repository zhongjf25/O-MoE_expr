CUDA_VISIBLE_DEVICES=7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=/root/autodl-tmp/workspace/ldy/vllm-offloading:$PYTHONPATH \
vllm serve /root/autodl-tmp/models/Qwen1.5-MoE-A2.7B \
    --port 8001 \
    --gpu-memory-utilization 0.9 \
    --offload-expert \
    --offload-expert-limit 10 \
    --tensor-parallel-size 1 \
    --enforce-eager \
    --max-model-len 8192 \
    --dynamic-cache-enabled \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --expert-no-copy-compute \
    --no-async-scheduling \
    --log-kv-allocation-per-step \
    --kv-allocation-csv-path omoe_kv_allocation.csv