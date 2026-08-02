#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="/home/ma-user/work/experiments/scripts"
BENCHMARK_SCRIPT="$SCRIPT_DIR/benchmark_qwen35_122b.sh"
ACTIVATE_SCRIPT="$SCRIPT_DIR/activate_vllm_ascend.sh"
MODEL_PATH="/home/ma-user/work/models/minimax-m2.7"
RESULT_ROOT="/home/ma-user/work/experiments/results/vllm-ascend/minimax-m2.7"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="${RESULTS_DIR:-${RESULT_ROOT}/${RUN_ID}_baseline-util0.985-rr15-n2000}"
STATUS_FILE="/home/ma-user/work/experiments/minimax_m27_vllm_ascend_util0.985_rr15_n2000.status"
RESULTS_FILE="/home/ma-user/work/experiments/minimax_m27_vllm_ascend_util0.985_rr15_n2000.results"

write_status() {
    local state="$1"
    local exit_code="${2:-}"
    {
        printf 'state=%s\n' "$state"
        printf 'run_id=%s\n' "$RUN_ID"
        printf 'framework=vllm-ascend\n'
        printf 'model=minimax-m2.7\n'
        printf 'gpu_memory_utilization=0.985\n'
        printf 'num_prompts=2000\n'
        printf 'request_rate=15\n'
        printf 'temperature=0\n'
        printf 'results_dir=%s\n' "$RESULTS_DIR"
        printf 'updated_at=%s\n' "$(date '+%F %T %Z')"
        if [[ -n "$exit_code" ]]; then
            printf 'exit_code=%s\n' "$exit_code"
        fi
    } >"$STATUS_FILE"
}

on_exit() {
    local rc=$?
    if (( rc == 0 )); then
        write_status complete "$rc"
    else
        write_status failed "$rc"
    fi
}
trap on_exit EXIT

mkdir -p "$RESULTS_DIR"
printf 'results_dir=%s\n' "$RESULTS_DIR" >"$RESULTS_FILE"
write_status running

ACTIVATE_SCRIPT="$ACTIVATE_SCRIPT" \
MODEL_PATH="$MODEL_PATH" \
FRAMEWORK="vllm-ascend" \
MODEL_NAME="minimax-m2.7" \
NUM_PROMPTS=2000 \
RR_START=15 \
RR_END=15 \
RR_STEP=1 \
EXP_LABEL="baseline-minimax-m2.7-util0.985-rr15-n2000" \
RESULTS_DIR="$RESULTS_DIR" \
bash "$BENCHMARK_SCRIPT" --temperature=0
