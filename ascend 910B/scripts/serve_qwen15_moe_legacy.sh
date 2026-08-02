#!/usr/bin/env bash
#
# Legacy NVIDIA example retained for reference only.

set -Eeuo pipefail

PYTHONPATH=/root/workspace/mycode/vllm-offloading:${PYTHONPATH:-} \
exec vllm serve /root/workspace/model_weights/qwen1.5-moe-a2.7b \
    --gpu-memory-utilization 0.7 \
    --enforce-eager \
    --offload-expert \
    --cached-num-experts 55 \
    --offload-expert-limit 30 \
    --dynamic-cache-enabled \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --no-async-scheduling
