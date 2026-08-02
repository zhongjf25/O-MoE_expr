#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=/home/ma-user/work/experiments/scripts

export ACTIVATE_SCRIPT="${SCRIPT_DIR}/activate_new_omoe.sh"
export MODEL_PATH="${MODEL_PATH:-/home/ma-user/work/models/minimax-m2.7}"
export DATASET_PATH="${DATASET_PATH:-/home/ma-user/work/models/ShareGPT_V3_unfiltered_cleaned_split.json}"
export RR_START="${RR_START:-15}"
export RR_END="${RR_END:-15}"
export RR_STEP="${RR_STEP:-1}"
export NUM_PROMPTS="${NUM_PROMPTS:-2000}"
export EXP_LABEL="${EXP_LABEL:-new-omoe-minimax-full-rr15-n2000}"
export FRAMEWORK="${FRAMEWORK:-new-omoe-ascend}"
export MODEL_NAME="${MODEL_NAME:-minimax-m2.7}"
export BACKEND="${BACKEND:-openai}"

exec bash "${SCRIPT_DIR}/benchmark_qwen35_122b.sh" --temperature 0 "$@"
