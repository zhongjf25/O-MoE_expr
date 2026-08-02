#!/usr/bin/env bash

set -Eeuo pipefail

ACTIVATE_SCRIPT="${ACTIVATE_SCRIPT:-/home/ma-user/work/experiments/scripts/activate_vllm_ascend.sh}"
MODEL_PATH="${MODEL_PATH:-/home/ma-user/work/models/minimax-m2.7}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-MiniMax-M2.7}"
REQUESTED_ASCEND_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TP_SIZE="${TP_SIZE:-8}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.985}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-0}"
QUANTIZATION="${QUANTIZATION:-ascend}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-0}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
PID_FILE="${PID_FILE:-/home/ma-user/work/experiments/minimax_m27_vllm_ascend.pid}"

is_true() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        0|false|no|off) return 1 ;;
        *)
            echo "[ERROR] Invalid boolean value: $1" >&2
            exit 1
            ;;
    esac
}

if [[ ! -f "$ACTIVATE_SCRIPT" ]]; then
    echo "[ERROR] Activation script not found: $ACTIVATE_SCRIPT" >&2
    exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[ERROR] Model path not found: $MODEL_PATH" >&2
    exit 1
fi
if [[ ! -f "$MODEL_PATH/quant_model_description.json" ]]; then
    echo "[ERROR] ModelSlim quantization description not found" >&2
    exit 1
fi

set +u
# shellcheck disable=SC1090
source "$ACTIVATE_SCRIPT"
set -u

export ASCEND_RT_VISIBLE_DEVICES="$REQUESTED_ASCEND_DEVICES"
if awk "BEGIN {exit !(($CPU_OFFLOAD_GB + 0) > 0)}"; then
    export VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1
fi

DEVICE_COUNT=$(awk -F, '{print NF}' <<<"$ASCEND_RT_VISIBLE_DEVICES")
if (( TP_SIZE > DEVICE_COUNT )); then
    echo "[ERROR] TP_SIZE=$TP_SIZE exceeds visible NPU count $DEVICE_COUNT" >&2
    exit 1
fi

VLLM_ARGS=(
    serve "$MODEL_PATH"
    --host "$HOST"
    --port "$PORT"
    --served-model-name "$SERVED_MODEL_NAME"
    --tensor-parallel-size "$TP_SIZE"
    --gpu-memory-utilization "$GPU_MEMORY_UTIL"
    --max-model-len "$MAX_MODEL_LEN"
    --quantization "$QUANTIZATION"
    --enforce-eager
)

if awk "BEGIN {exit !(($CPU_OFFLOAD_GB + 0) > 0)}"; then
    VLLM_ARGS+=(--cpu-offload-gb "$CPU_OFFLOAD_GB")
fi
if is_true "$TRUST_REMOTE_CODE"; then
    VLLM_ARGS+=(--trust-remote-code)
fi
if is_true "$ENABLE_PREFIX_CACHING"; then
    VLLM_ARGS+=(--enable-prefix-caching)
else
    VLLM_ARGS+=(--no-enable-prefix-caching)
fi
if is_true "$ENABLE_CHUNKED_PREFILL"; then
    VLLM_ARGS+=(--enable-chunked-prefill)
else
    VLLM_ARGS+=(--no-enable-chunked-prefill)
fi
if is_true "$ASYNC_SCHEDULING"; then
    VLLM_ARGS+=(--async-scheduling)
else
    VLLM_ARGS+=(--no-async-scheduling)
fi

echo "============================================================"
echo " Baseline vLLM Ascend MiniMax-M2.7 Serving"
echo "============================================================"
echo " Model:                    $MODEL_PATH"
echo " Served model name:        $SERVED_MODEL_NAME"
echo " Visible NPUs:             $ASCEND_RT_VISIBLE_DEVICES"
echo " Tensor parallel size:     $TP_SIZE"
echo " NPU memory utilization:   $GPU_MEMORY_UTIL"
echo " Max model length:         $MAX_MODEL_LEN"
echo " Quantization:             $QUANTIZATION"
echo " CPU offload per NPU:      $CPU_OFFLOAD_GB GiB"
echo " Prefix caching:           $ENABLE_PREFIX_CACHING"
echo " Chunked prefill:          $ENABLE_CHUNKED_PREFILL"
echo " Async scheduling:         $ASYNC_SCHEDULING"
echo " Endpoint:                 http://$HOST:$PORT"
echo "============================================================"

printf '%s\n' "$$" >"$PID_FILE"
echo "[INFO] Starting baseline vLLM Ascend server..."
exec vllm "${VLLM_ARGS[@]}" "$@"
