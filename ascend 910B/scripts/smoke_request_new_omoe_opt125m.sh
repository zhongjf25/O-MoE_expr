#!/usr/bin/env bash
set -Eeuo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

curl --fail-with-body --silent --show-error \
  "http://${HOST}:${PORT}/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "opt-125m",
    "prompt": "Reliable software deployment requires",
    "max_tokens": 16,
    "temperature": 0
  }'
printf '\n'
