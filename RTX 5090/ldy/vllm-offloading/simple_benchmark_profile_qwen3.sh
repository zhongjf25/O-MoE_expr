#!/usr/bin/env bash
set -euo pipefail

HOST=127.0.0.1
PORT=8001
PROFILE_DIR=/root/autodl-tmp/workspace/ldy/vllm-offloading/vllm_profile

echo "[1/3] start profiler"
curl -fsS -X POST "http://${HOST}:${PORT}/start_profile"

echo "[2/3] run benchmark requests"
PYTHONPATH=/root/autodl-tmp/workspace/ldy/vllm-offloading:${PYTHONPATH:-} \
vllm bench serve \
    --host "${HOST}" \
    --port "${PORT}" \
    --model /root/autodl-tmp/models/Qwen3-30B-A3B \
    --dataset-name sharegpt \
    --dataset-path /root/autodl-tmp/dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
    --request-rate 1 \
    --num-prompts 1

echo "[3/3] stop profiler"
curl -fsS -X POST "http://${HOST}:${PORT}/stop_profile"

echo "Profiler files should be under: ${PROFILE_DIR}"
ls -lah "${PROFILE_DIR}" || true

