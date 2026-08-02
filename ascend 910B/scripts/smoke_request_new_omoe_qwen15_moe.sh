#!/usr/bin/env bash
set -Eeuo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

curl --fail-with-body --silent --show-error \
  "http://${HOST}:${PORT}/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen1.5-moe-a2.7b",
    "prompt": "The capital of France is",
    "max_tokens": 16,
    "temperature": 0
  }'
printf '\n'
