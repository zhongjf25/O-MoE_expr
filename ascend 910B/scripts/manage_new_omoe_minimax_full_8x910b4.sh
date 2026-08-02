#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=/home/ma-user/work/experiments/scripts
RUNTIME_DIR=/home/ma-user/work/experiments/new_omoe_minimax_full_8x910b4
PID_FILE="${RUNTIME_DIR}/server.pid"
LOG_FILE="${RUNTIME_DIR}/server.log"
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
    nohup "${SCRIPT_DIR}/serve_new_omoe_minimax_full_8x910b4.sh" \
      >"$LOG_FILE" 2>&1 </dev/null &
    pid=$!
    printf '%s\n' "$pid" >"$PID_FILE"
    sleep 2
    if ! kill -0 "$pid" 2>/dev/null; then
      printf 'server exited during startup; inspect %s\n' "$LOG_FILE" >&2
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
    for _ in $(seq 1 60); do
      if ! kill -0 "$pid" 2>/dev/null; then
        printf 'stopped pid=%s\n' "$pid"
        exit 0
      fi
      sleep 1
    done
    printf 'pid=%s did not exit after 60 seconds\n' "$pid" >&2
    exit 1
    ;;
  logs)
    tail -n "${LINES:-200}" "$LOG_FILE"
    ;;
  *)
    printf 'usage: %s {start|status|stop|logs}\n' "$0" >&2
    exit 2
    ;;
esac
