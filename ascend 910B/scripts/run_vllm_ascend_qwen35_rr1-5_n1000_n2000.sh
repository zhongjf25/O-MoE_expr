#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="/home/ma-user/work/experiments/scripts"
BENCHMARK_SCRIPT="$SCRIPT_DIR/benchmark_qwen35_122b.sh"
ACTIVATE_SCRIPT="$SCRIPT_DIR/activate_vllm_ascend.sh"
RESULT_ROOT="/home/ma-user/work/experiments/results/vllm-ascend/qwen35-122b"
STATUS_FILE="/home/ma-user/work/experiments/vllm_ascend_qwen35_rr1-5_n1000_n2000.status"
RESULTS_FILE="/home/ma-user/work/experiments/vllm_ascend_qwen35_rr1-5_n1000_n2000.results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [[ ! -f "$BENCHMARK_SCRIPT" ]]; then
    echo "[ERROR] Benchmark script not found: $BENCHMARK_SCRIPT" >&2
    exit 1
fi
if [[ ! -f "$ACTIVATE_SCRIPT" ]]; then
    echo "[ERROR] Activation script not found: $ACTIVATE_SCRIPT" >&2
    exit 1
fi

printf 'running\n' >"$STATUS_FILE"

on_exit() {
    rc=$?
    printf '%s\n' "$rc" >"$STATUS_FILE"
}
trap on_exit EXIT

run_case() {
    local num_prompts="$1"
    local label="baseline-rr1-5-n${num_prompts}"
    local results_dir="${RESULT_ROOT}/${TIMESTAMP}_${label}"

    printf 'n%s=%s\n' "$num_prompts" "$results_dir" >>"$RESULTS_FILE"

    ACTIVATE_SCRIPT="$ACTIVATE_SCRIPT" \
    FRAMEWORK="vllm-ascend" \
    MODEL_NAME="qwen35-122b" \
    NUM_PROMPTS="$num_prompts" \
    RR_START=1 \
    RR_END=5 \
    RR_STEP=1 \
    EXP_LABEL="$label" \
    RESULTS_DIR="$results_dir" \
    bash "$BENCHMARK_SCRIPT"
}

: >"$RESULTS_FILE"
run_case 1000
run_case 2000
