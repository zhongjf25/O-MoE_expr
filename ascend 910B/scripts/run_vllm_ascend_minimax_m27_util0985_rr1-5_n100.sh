#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="/home/ma-user/work/experiments/scripts"
BENCHMARK_SCRIPT="$SCRIPT_DIR/benchmark_qwen35_122b.sh"
ACTIVATE_SCRIPT="$SCRIPT_DIR/activate_vllm_ascend.sh"
MODEL_PATH="/home/ma-user/work/models/minimax-m2.7"
RESULT_ROOT="/home/ma-user/work/experiments/results/vllm-ascend/minimax-m2.7"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="${RESULTS_DIR:-${RESULT_ROOT}/${RUN_ID}_baseline-util0.985-rr1-5-n100}"
STATUS_FILE="/home/ma-user/work/experiments/minimax_m27_vllm_ascend_util0.985_rr1-5_n100.status"
RESULTS_FILE="/home/ma-user/work/experiments/minimax_m27_vllm_ascend_util0.985_rr1-5_n100.results"

write_status() {
    local state="$1"
    local exit_code="${2:-}"
    {
        printf 'state=%s\n' "$state"
        printf 'run_id=%s\n' "$RUN_ID"
        printf 'framework=vllm-ascend\n'
        printf 'model=minimax-m2.7\n'
        printf 'gpu_memory_utilization=0.985\n'
        printf 'cpu_offload_gb_per_npu=0\n'
        printf 'num_prompts_per_rr=100\n'
        printf 'rr_start=1\n'
        printf 'rr_end=5\n'
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
NUM_PROMPTS=100 \
RR_START=1 \
RR_END=5 \
RR_STEP=1 \
EXP_LABEL="baseline-minimax-m2.7-util0.985-rr1-5-n100" \
RESULTS_DIR="$RESULTS_DIR" \
bash "$BENCHMARK_SCRIPT" --temperature=0
