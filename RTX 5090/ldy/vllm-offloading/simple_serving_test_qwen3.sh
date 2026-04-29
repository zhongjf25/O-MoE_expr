CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=/root/autodl-tmp/workspace/ldy/vllm-offloading:$PYTHONPATH \
vllm serve /root/autodl-tmp/models/Qwen3-30B-A3B \
    --port 8001 \
    --gpu-memory-utilization 0.94 \
    --offload-expert \
    --offload-expert-limit 100 \
    --tensor-parallel-size 1 \
    --enforce-eager \
    --max-model-len 8192 \
    --dynamic-cache-enabled \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --expert-no-copy-compute \
    --no-async-scheduling \
    # --profiler-config '{"profiler":"torch","torch_profiler_dir":"./vllm_profile","torch_profiler_with_stack":true,"torch_profiler_with_memory":true}'
