#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=/home/ma-user/work/experiments/scripts
RUNTIME_DIR=/home/ma-user/work/experiments/new_omoe_minimax_full_8x910b4
PID_FILE="${RUNTIME_DIR}/benchmark_rr15_n2000.pid"
LOG_FILE="${RUNTIME_DIR}/benchmark_rr15_n2000.log"
ACTION="${1:-status}"

mkdir -p "$RUNTIME_DIR"

is_running() {
  [[ -s "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

case "$ACTION" in
  start)
    if is_running; then
      printf 'already running pid=%s log=%s\n' "$(cat "$PID_FILE")" "$LOG_FILE"
      exit 0
    fi
    nohup "${SCRIPT_DIR}/benchmark_new_omoe_minimax_full_8x910b4.sh" \
      >"$LOG_FILE" 2>&1 </dev/null &
    pid=$!
    printf '%s\n' "$pid" >"$PID_FILE"
    sleep 2
    if ! kill -0 "$pid" 2>/dev/null; then
      printf 'benchmark exited during startup; inspect %s\n' "$LOG_FILE" >&2
      exit 1
    fi
    printf 'started pid=%s log=%s\n' "$pid" "$LOG_FILE"
    ;;
  status)
    if is_running; then
      printf 'running pid=%s log=%s\n' "$(cat "$PID_FILE")" "$LOG_FILE"
    else
      printf 'not running log=%s\n' "$LOG_FILE"
      exit 1
    fi
    ;;
  stop)
    if ! is_running; then
      printf 'not running\n'
      exit 0
    fi
    pid=$(cat "$PID_FILE")
    kill -TERM "$pid"
    printf 'stop requested pid=%s\n' "$pid"
    ;;
  logs)
    tail -n "${LINES:-160}" "$LOG_FILE"
    ;;
  *)
    printf 'usage: %s {start|status|stop|logs}\n' "$0" >&2
    exit 2
    ;;
esac
