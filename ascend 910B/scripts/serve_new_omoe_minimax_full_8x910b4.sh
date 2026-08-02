#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=/home/ma-user/work/experiments/scripts
OMOE_VLLM=/home/ma-user/work/new_omoe/O-MoE
VENV=/home/ma-user/work/envs/vllm-0.16.1rc0-py311

set +u
source "${SCRIPT_DIR}/activate_new_omoe.sh"
set -u
"${VENV}/bin/python" "${SCRIPT_DIR}/prepare_new_omoe_minimax_full.py"

cd "$OMOE_VLLM"
exec python "${SCRIPT_DIR}/serve_new_omoe_minimax_full_8x910b4.py"
