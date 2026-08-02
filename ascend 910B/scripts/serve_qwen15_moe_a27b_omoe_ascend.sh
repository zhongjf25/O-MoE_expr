#!/usr/bin/env bash
set -eo pipefail

source /home/ma-user/work/omoe_runtime/activate_omoe_ascend.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

exec vllm serve /home/ma-user/work/models/Qwen1.5-MoE-A2.7B \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name Qwen1.5-MoE-A2.7B \
    --tensor-parallel-size 4 \
    --block-size 64 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --enforce-eager \
    --cached-num-experts 40 \
    --offload-expert-limit 20 \
    --no-dynamic-cache-enabled \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --no-async-scheduling
