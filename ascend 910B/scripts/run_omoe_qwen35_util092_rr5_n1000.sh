#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="/home/ma-user/work/experiments/scripts"
BENCHMARK_SCRIPT="$SCRIPT_DIR/benchmark_qwen35_122b.sh"
ACTIVATE_SCRIPT="/home/ma-user/work/omoe_runtime/activate_omoe_ascend.sh"
RESULT_ROOT="/home/ma-user/work/experiments/results/omoe-ascend/qwen35-122b"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="${RESULTS_DIR:-${RESULT_ROOT}/${RUN_ID}_omoe-util0.92-offload100-rr5-n1000}"
STATUS_FILE="/home/ma-user/work/experiments/omoe_qwen35_util0.92_rr5_n1000.status"
RESULTS_FILE="/home/ma-user/work/experiments/omoe_qwen35_util0.92_rr5_n1000.results"

if [[ ! -f "$BENCHMARK_SCRIPT" ]]; then
    echo "[ERROR] Benchmark script not found: $BENCHMARK_SCRIPT" >&2
    exit 1
fi
if [[ ! -f "$ACTIVATE_SCRIPT" ]]; then
    echo "[ERROR] Activation script not found: $ACTIVATE_SCRIPT" >&2
    exit 1
fi

write_status() {
    local state="$1"
    local exit_code="${2:-}"
    {
        printf 'state=%s\n' "$state"
        printf 'run_id=%s\n' "$RUN_ID"
        printf 'gpu_memory_utilization=0.92\n'
        printf 'offload_expert_limit=100\n'
        printf 'num_prompts=1000\n'
        printf 'rr=5\n'
        printf 'results_dir=%s\n' "$RESULTS_DIR"
        printf 'updated_at=%s\n' "$(date '+%F %T %Z')"
        if [[ -n "$exit_code" ]]; then
            printf 'exit_code=%s\n' "$exit_code"
        fi
    } >"$STATUS_FILE"
}

on_exit() {
    local rc=$?
    if [[ "$rc" -eq 0 ]]; then
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
FRAMEWORK="omoe-ascend" \
MODEL_NAME="qwen35-122b" \
NUM_PROMPTS=1000 \
RR_START=5 \
RR_END=5 \
RR_STEP=1 \
EXP_LABEL="omoe-util0.92-offload100-rr5-n1000" \
RESULTS_DIR="$RESULTS_DIR" \
bash "$BENCHMARK_SCRIPT"
