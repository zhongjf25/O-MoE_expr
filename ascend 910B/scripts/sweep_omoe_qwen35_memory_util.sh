#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="/home/ma-user/work/experiments/scripts"
SERVICE_SCRIPT="$SCRIPT_DIR/serve_qwen35_122b_omoe_ascend.sh"
BENCHMARK_SCRIPT="$SCRIPT_DIR/benchmark_qwen35_122b.sh"
ACTIVATE_SCRIPT="/home/ma-user/work/omoe_runtime/activate_omoe_ascend.sh"
RESULT_ROOT="/home/ma-user/work/experiments/results/omoe-ascend/qwen35-122b"
LOG_ROOT="/home/ma-user/work/experiments/logs"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$RESULT_ROOT/${RUN_ID}_memory-util-sweep-offload100"
STATUS_FILE="/home/ma-user/work/experiments/omoe_memory_util_sweep.status"
PID_FILE="/home/ma-user/work/experiments/omoe_memory_util_sweep.pid"
SELECTED_PID_FILE="/home/ma-user/work/experiments/omoe_selected_service.pid"
METRICS_FILE="$RUN_DIR/metrics.tsv"
INITIAL_SERVICE_PID="${INITIAL_SERVICE_PID:-}"
RATIOS=(0.96 0.94 0.92 0.90 0.88 0.86 0.84 0.82 0.80)

mkdir -p "$RUN_DIR" "$LOG_ROOT"
printf '%s\n' "$$" >"$PID_FILE"
printf 'util\tstatus\tcompleted\tfailed\trequest_throughput\toutput_throughput\tmean_ttft_ms\tmean_tpot_ms\n' >"$METRICS_FILE"

CURRENT_SERVICE_PID=""

write_status() {
    local state="$1"
    local util="${2:-}"
    local detail="${3:-}"
    {
        printf 'state=%s\n' "$state"
        printf 'run_id=%s\n' "$RUN_ID"
        printf 'current_util=%s\n' "$util"
        printf 'detail=%s\n' "$detail"
        printf 'run_dir=%s\n' "$RUN_DIR"
        printf 'metrics_file=%s\n' "$METRICS_FILE"
        printf 'updated_at=%s\n' "$(date '+%F %T %Z')"
    } >"$STATUS_FILE"
}

stop_service() {
    local pid="${1:-}"
    [[ -n "$pid" ]] || return 0
    if ! kill -0 "$pid" 2>/dev/null; then
        return 0
    fi

    kill -TERM "$pid" 2>/dev/null || true
    for _ in {1..90}; do
        if ! kill -0 "$pid" 2>/dev/null; then
            sleep 10
            return 0
        fi
        sleep 2
    done
    return 1
}

wait_ready() {
    local pid="$1"
    for _ in {1..180}; do
        if [[ "$(curl -sS -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8000/v1/models 2>/dev/null || true)" == "200" ]]; then
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            return 1
        fi
        sleep 5
    done
    return 1
}

start_service() {
    local util="$1"
    local log_file="$2"

    env \
        GPU_MEMORY_UTIL="$util" \
        OFFLOAD_EXPERT_LIMIT=100 \
        bash "$SERVICE_SCRIPT" >"$log_file" 2>&1 &
    CURRENT_SERVICE_PID=$!
    wait_ready "$CURRENT_SERVICE_PID"
}

record_result() {
    local util="$1"
    local result_file="$2"
    env -u PYTHONPATH VLLM_PLUGINS= python3 - "$util" "$result_file" "$METRICS_FILE" <<'PY'
import json
import sys

util, result_path, metrics_path = sys.argv[1:]
with open(result_path, encoding="utf-8") as stream:
    result = json.load(stream)

row = [
    util,
    "ok",
    str(result.get("completed", 0)),
    str(result.get("failed", 0)),
    str(result.get("request_throughput", 0.0)),
    str(result.get("output_throughput", 0.0)),
    str(result.get("mean_ttft_ms", 0.0)),
    str(result.get("mean_tpot_ms", 0.0)),
]
with open(metrics_path, "a", encoding="utf-8") as stream:
    stream.write("\t".join(row) + "\n")
PY
}

on_signal() {
    write_status interrupted "" "controller received a signal"
    stop_service "$CURRENT_SERVICE_PID" || true
    exit 130
}
trap on_signal INT TERM

if [[ -n "$INITIAL_SERVICE_PID" ]]; then
    write_status stopping_initial 0.96 "stopping service pid $INITIAL_SERVICE_PID"
    if ! stop_service "$INITIAL_SERVICE_PID"; then
        write_status failed 0.96 "initial service did not stop"
        exit 1
    fi
fi

for util in "${RATIOS[@]}"; do
    server_log="$LOG_ROOT/omoe_memory_sweep_util_${util}.server.log"
    result_dir="$RUN_DIR/util_${util}"
    result_file="$result_dir/rr_5/result.json"
    mkdir -p "$result_dir"

    write_status starting "$util" "starting O-MoE service"
    if ! start_service "$util" "$server_log"; then
        printf '%s\tstartup_failed\t0\t0\t0\t0\t0\t0\n' "$util" >>"$METRICS_FILE"
        write_status startup_failed "$util" "$server_log"
        stop_service "$CURRENT_SERVICE_PID" || true
        CURRENT_SERVICE_PID=""
        continue
    fi

    write_status benchmarking "$util" "10 prompts, RR=5, output_len=128, temperature=0"
    if env \
        ACTIVATE_SCRIPT="$ACTIVATE_SCRIPT" \
        FRAMEWORK=omoe-ascend \
        MODEL_NAME=qwen35-122b \
        NUM_PROMPTS=10 \
        RR_START=5 \
        RR_END=5 \
        RR_STEP=1 \
        SHAREGPT_OUTPUT_LEN=128 \
        EXP_LABEL="memory-util-${util}-offload100-n10" \
        RESULTS_DIR="$result_dir" \
        bash "$BENCHMARK_SCRIPT" --temperature=0; then
        if [[ -f "$result_file" ]]; then
            record_result "$util" "$result_file"
        else
            printf '%s\tmissing_result\t0\t0\t0\t0\t0\t0\n' "$util" >>"$METRICS_FILE"
        fi
    else
        printf '%s\tbenchmark_failed\t0\t1\t0\t0\t0\t0\n' "$util" >>"$METRICS_FILE"
    fi

    write_status stopping "$util" "benchmark finished"
    stop_service "$CURRENT_SERVICE_PID" || true
    CURRENT_SERVICE_PID=""
done

BEST_UTIL=$(awk -F '\t' '
    NR > 1 && $2 == "ok" && $3 == 10 && $4 == 0 {
        if (!found || $6 > best) {
            found = 1
            best = $6
            util = $1
        }
    }
    END {if (found) print util}
' "$METRICS_FILE")

if [[ -z "$BEST_UTIL" ]]; then
    write_status failed "" "no stable configuration completed"
    exit 1
fi

FINAL_LOG="$LOG_ROOT/omoe_selected_util_${BEST_UTIL}.server.log"
write_status starting_best "$BEST_UTIL" "starting selected service"
if ! start_service "$BEST_UTIL" "$FINAL_LOG"; then
    write_status failed "$BEST_UTIL" "selected service failed to restart"
    exit 1
fi

printf '%s\n' "$CURRENT_SERVICE_PID" >"$SELECTED_PID_FILE"
write_status complete "$BEST_UTIL" "selected service pid $CURRENT_SERVICE_PID"

echo "[OK] Memory utilization sweep complete"
echo "[OK] Selected utilization: $BEST_UTIL"
echo "[OK] Service PID: $CURRENT_SERVICE_PID"
echo "[OK] Metrics: $METRICS_FILE"
