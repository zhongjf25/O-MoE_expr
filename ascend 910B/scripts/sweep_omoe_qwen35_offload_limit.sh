#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="/home/ma-user/work/experiments/scripts"
SERVICE_SCRIPT="$SCRIPT_DIR/serve_qwen35_122b_omoe_ascend.sh"
BENCHMARK_SCRIPT="$SCRIPT_DIR/benchmark_qwen35_122b.sh"
ACTIVATE_SCRIPT="/home/ma-user/work/omoe_runtime/activate_omoe_ascend.sh"
RESULT_ROOT="/home/ma-user/work/experiments/results/omoe-ascend/qwen35-122b"
LOG_ROOT="/home/ma-user/work/experiments/logs"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$RESULT_ROOT/${RUN_ID}_offload-limit-sweep-util0.92-rr5-n100"
STATUS_FILE="/home/ma-user/work/experiments/omoe_offload_limit_sweep.status"
PID_FILE="/home/ma-user/work/experiments/omoe_offload_limit_sweep.pid"
SELECTED_PID_FILE="/home/ma-user/work/experiments/omoe_selected_service.pid"
BEST_CONFIG_FILE="$RUN_DIR/best_config.env"
METRICS_FILE="$RUN_DIR/metrics.tsv"
SUMMARY_FILE="$RUN_DIR/summary.json"
INITIAL_SERVICE_PID="${INITIAL_SERVICE_PID:-}"

GPU_MEMORY_UTIL="0.92"
NUM_PROMPTS="100"
REQUEST_RATE="5"
NUM_EXPERTS="256"
DEFAULT_CACHED_EXPERTS="128"
OFFLOAD_LIMITS=(20 40 60 80 100 120 140 160)

mkdir -p "$RUN_DIR" "$LOG_ROOT"
printf '%s\n' "$$" >"$PID_FILE"
printf '%s\n' \
    $'offload_limit\tstatus\tcached_experts\tcompleted\tfailed\trequest_throughput\toutput_throughput\ttotal_token_throughput\tduration_s\tmean_ttft_ms\tp99_ttft_ms\tmean_tpot_ms\tp99_tpot_ms\tresult_file\tserver_log' \
    >"$METRICS_FILE"

CURRENT_SERVICE_PID=""
CURRENT_LIMIT=""
FINISHED=0

cached_for_limit() {
    local limit="$1"
    local minimum=$((NUM_EXPERTS - limit))
    if (( minimum < 0 )); then
        minimum=0
    fi
    if (( DEFAULT_CACHED_EXPERTS < minimum )); then
        printf '%s\n' "$minimum"
    else
        printf '%s\n' "$DEFAULT_CACHED_EXPERTS"
    fi
}

write_status() {
    local state="$1"
    local limit="${2:-}"
    local detail="${3:-}"
    {
        printf 'state=%s\n' "$state"
        printf 'run_id=%s\n' "$RUN_ID"
        printf 'gpu_memory_utilization=%s\n' "$GPU_MEMORY_UTIL"
        printf 'current_offload_limit=%s\n' "$limit"
        printf 'num_prompts=%s\n' "$NUM_PROMPTS"
        printf 'request_rate=%s\n' "$REQUEST_RATE"
        printf 'detail=%s\n' "$detail"
        printf 'run_dir=%s\n' "$RUN_DIR"
        printf 'metrics_file=%s\n' "$METRICS_FILE"
        printf 'summary_file=%s\n' "$SUMMARY_FILE"
        printf 'updated_at=%s\n' "$(date '+%F %T %Z')"
    } >"$STATUS_FILE"
}

stop_service() {
    local pid="${1:-}"
    [[ -n "$pid" ]] || return 0

    if kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$pid" 2>/dev/null || true
        for _ in {1..120}; do
            if ! kill -0 "$pid" 2>/dev/null; then
                break
            fi
            sleep 2
        done
    fi

    if kill -0 "$pid" 2>/dev/null; then
        return 1
    fi

    for _ in {1..60}; do
        if ! curl -fsS --max-time 2 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
            sleep 10
            return 0
        fi
        sleep 2
    done
    return 1
}

wait_ready() {
    local pid="$1"
    for _ in {1..240}; do
        if [[ "$(curl -sS -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8000/v1/models 2>/dev/null || true)" == "200" ]]; then
            sleep 5
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
    local limit="$1"
    local log_file="$2"

    env \
        GPU_MEMORY_UTIL="$GPU_MEMORY_UTIL" \
        OFFLOAD_EXPERT_LIMIT="$limit" \
        bash "$SERVICE_SCRIPT" >"$log_file" 2>&1 &
    CURRENT_SERVICE_PID=$!
    CURRENT_LIMIT="$limit"
    wait_ready "$CURRENT_SERVICE_PID"
}

record_failure() {
    local limit="$1"
    local status="$2"
    local result_file="$3"
    local server_log="$4"
    local cached
    cached="$(cached_for_limit "$limit")"
    printf '%s\t%s\t%s\t0\t1\t0\t0\t0\t0\t0\t0\t0\t0\t%s\t%s\n' \
        "$limit" "$status" "$cached" "$result_file" "$server_log" \
        >>"$METRICS_FILE"
}

record_result() {
    local limit="$1"
    local result_file="$2"
    local server_log="$3"
    local cached
    cached="$(cached_for_limit "$limit")"

    env -u PYTHONPATH VLLM_PLUGINS= python3 - \
        "$limit" "$cached" "$result_file" "$server_log" "$METRICS_FILE" <<'PY'
import json
import sys

limit, cached, result_path, server_log, metrics_path = sys.argv[1:]
with open(result_path, encoding="utf-8") as stream:
    result = json.load(stream)

completed = int(result.get("completed", 0))
failed = int(result.get("failed", 0))
status = "ok" if completed == 100 and failed == 0 else "incomplete"

row = [
    limit,
    status,
    cached,
    str(completed),
    str(failed),
    str(result.get("request_throughput", 0.0)),
    str(result.get("output_throughput", 0.0)),
    str(result.get("total_token_throughput", 0.0)),
    str(result.get("duration", 0.0)),
    str(result.get("mean_ttft_ms", 0.0)),
    str(result.get("p99_ttft_ms", 0.0)),
    str(result.get("mean_tpot_ms", 0.0)),
    str(result.get("p99_tpot_ms", 0.0)),
    result_path,
    server_log,
]
with open(metrics_path, "a", encoding="utf-8") as stream:
    stream.write("\t".join(row) + "\n")
PY
}

on_signal() {
    write_status interrupted "$CURRENT_LIMIT" "controller received a signal"
    stop_service "$CURRENT_SERVICE_PID" || true
    exit 130
}

on_exit() {
    local rc=$?
    if (( FINISHED == 0 )); then
        write_status failed "$CURRENT_LIMIT" "controller exited with code $rc"
        stop_service "$CURRENT_SERVICE_PID" || true
    fi
}

trap on_signal INT TERM
trap on_exit EXIT

if [[ -n "$INITIAL_SERVICE_PID" ]]; then
    write_status stopping_initial "" "stopping service pid $INITIAL_SERVICE_PID"
    if ! stop_service "$INITIAL_SERVICE_PID"; then
        write_status failed "" "initial service did not stop cleanly"
        exit 1
    fi
fi

for limit in "${OFFLOAD_LIMITS[@]}"; do
    server_log="$LOG_ROOT/${RUN_ID}_offload_${limit}.server.log"
    result_dir="$RUN_DIR/offload_${limit}"
    result_file="$result_dir/rr_5/result.json"
    mkdir -p "$result_dir"

    write_status starting "$limit" "starting O-MoE service"
    if ! start_service "$limit" "$server_log"; then
        record_failure "$limit" startup_failed "$result_file" "$server_log"
        write_status startup_failed "$limit" "$server_log"
        stop_service "$CURRENT_SERVICE_PID" || true
        CURRENT_SERVICE_PID=""
        CURRENT_LIMIT=""
        continue
    fi

    write_status benchmarking "$limit" "RR=5, N=100, temperature=0"
    if env \
        ACTIVATE_SCRIPT="$ACTIVATE_SCRIPT" \
        FRAMEWORK=omoe-ascend \
        MODEL_NAME=qwen35-122b \
        NUM_PROMPTS="$NUM_PROMPTS" \
        RR_START="$REQUEST_RATE" \
        RR_END="$REQUEST_RATE" \
        RR_STEP=1 \
        EXP_LABEL="offload-${limit}-util0.92-rr5-n100" \
        RESULTS_DIR="$result_dir" \
        bash "$BENCHMARK_SCRIPT" --temperature=0; then
        if [[ -f "$result_file" ]]; then
            record_result "$limit" "$result_file" "$server_log"
        else
            record_failure "$limit" missing_result "$result_file" "$server_log"
        fi
    else
        if [[ -f "$result_file" ]]; then
            record_result "$limit" "$result_file" "$server_log"
        else
            record_failure "$limit" benchmark_failed "$result_file" "$server_log"
        fi
    fi

    write_status stopping "$limit" "benchmark finished"
    if ! stop_service "$CURRENT_SERVICE_PID"; then
        write_status failed "$limit" "service did not stop cleanly"
        exit 1
    fi
    CURRENT_SERVICE_PID=""
    CURRENT_LIMIT=""
done

BEST_LIMIT=$(env -u PYTHONPATH VLLM_PLUGINS= python3 - \
    "$METRICS_FILE" "$SUMMARY_FILE" "$RUN_DIR" <<'PY'
import csv
import json
import sys

metrics_path, summary_path, run_dir = sys.argv[1:]
with open(metrics_path, encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))

numeric_fields = [
    "offload_limit",
    "cached_experts",
    "completed",
    "failed",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "duration_s",
    "mean_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p99_tpot_ms",
]
for row in rows:
    for field in numeric_fields:
        value = row.get(field, "")
        try:
            row[field] = int(value) if field in {
                "offload_limit", "cached_experts", "completed", "failed"
            } else float(value)
        except (TypeError, ValueError):
            row[field] = None

stable = [
    row for row in rows
    if row.get("status") == "ok"
    and row.get("completed") == 100
    and row.get("failed") == 0
]
best = max(
    stable,
    key=lambda row: (
        row.get("output_throughput") or 0.0,
        row.get("request_throughput") or 0.0,
        -(row.get("mean_tpot_ms") or float("inf")),
        -(row.get("mean_ttft_ms") or float("inf")),
    ),
    default=None,
)
summary = {
    "selection_rule": (
        "Among 100/100 successful runs, maximize output_throughput; "
        "break ties with request_throughput, mean_tpot_ms, then mean_ttft_ms."
    ),
    "run_dir": run_dir,
    "best_offload_limit": best.get("offload_limit") if best else None,
    "best": best,
    "experiments": rows,
}
with open(summary_path, "w", encoding="utf-8") as stream:
    json.dump(summary, stream, indent=2, ensure_ascii=False)
if best:
    print(best["offload_limit"])
PY
)

if [[ -z "$BEST_LIMIT" ]]; then
    write_status failed "" "no configuration completed 100/100 requests"
    exit 1
fi

BEST_CACHED="$(cached_for_limit "$BEST_LIMIT")"
FINAL_LOG="$LOG_ROOT/${RUN_ID}_selected_offload_${BEST_LIMIT}.server.log"
write_status starting_best "$BEST_LIMIT" "starting selected service"
if ! start_service "$BEST_LIMIT" "$FINAL_LOG"; then
    write_status failed "$BEST_LIMIT" "selected service failed to restart"
    exit 1
fi

printf '%s\n' "$CURRENT_SERVICE_PID" >"$SELECTED_PID_FILE"
{
    printf 'GPU_MEMORY_UTIL=%s\n' "$GPU_MEMORY_UTIL"
    printf 'OFFLOAD_EXPERT_LIMIT=%s\n' "$BEST_LIMIT"
    printf 'CACHED_NUM_EXPERTS=%s\n' "$BEST_CACHED"
    printf 'SERVICE_PID=%s\n' "$CURRENT_SERVICE_PID"
    printf 'SERVICE_LOG=%s\n' "$FINAL_LOG"
} >"$BEST_CONFIG_FILE"

FINISHED=1
write_status complete "$BEST_LIMIT" "selected service pid $CURRENT_SERVICE_PID"

echo "[OK] Offload-limit sweep complete"
echo "[OK] Selected offload limit: $BEST_LIMIT"
echo "[OK] Selected cached experts: $BEST_CACHED"
echo "[OK] Service PID: $CURRENT_SERVICE_PID"
echo "[OK] Metrics: $METRICS_FILE"
echo "[OK] Summary: $SUMMARY_FILE"
