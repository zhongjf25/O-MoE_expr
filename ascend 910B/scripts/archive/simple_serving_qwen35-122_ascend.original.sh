#!/usr/bin/env bash
#
# O-MoE Ascend Qwen3.5-122B serving launcher.
#
# Usage:
#   bash simple_serving_qwen35-122_ascend.sh
#
# Any additional arguments are appended to `vllm serve`.

set -Eeuo pipefail

ACTIVATE_SCRIPT="${ACTIVATE_SCRIPT:-/home/ma-user/work/omoe_runtime/activate_omoe_ascend.sh}"
MODEL_PATH="${MODEL_PATH:-/home/ma-user/work/models/Qwen3.5-122B-A10B}"
HOTSET_PATH="${HOTSET_PATH:-/home/ma-user/work/O-MoE/expert_hotset_profiles/Qwen3.5-122B-A10B.json}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-122B-A10B}"
REQUESTED_ASCEND_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TP_SIZE="${TP_SIZE:-8}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
CACHED_NUM_EXPERTS_OVERRIDE="${CACHED_NUM_EXPERTS:-}"
CACHED_NUM_EXPERTS="${CACHED_NUM_EXPERTS:-128}"
OFFLOAD_EXPERT_LIMIT="${OFFLOAD_EXPERT_LIMIT:-0}"
NUM_EXPERTS="${NUM_EXPERTS:-256}"
DYNAMIC_CACHE_ENABLED="${DYNAMIC_CACHE_ENABLED:-0}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-0}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

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
if [[ ! -f "$HOTSET_PATH" ]]; then
    echo "[ERROR] Expert hotset profile not found: $HOTSET_PATH" >&2
    exit 1
fi

if (( OFFLOAD_EXPERT_LIMIT > 0 )); then
    MIN_CACHED_EXPERTS=$((NUM_EXPERTS - OFFLOAD_EXPERT_LIMIT))
    if (( MIN_CACHED_EXPERTS < 0 )); then
        MIN_CACHED_EXPERTS=0
    fi
    if (( CACHED_NUM_EXPERTS < MIN_CACHED_EXPERTS )); then
        if [[ -n "$CACHED_NUM_EXPERTS_OVERRIDE" ]]; then
            echo "[ERROR] CACHED_NUM_EXPERTS=$CACHED_NUM_EXPERTS is too small" >&2
            echo "        for OFFLOAD_EXPERT_LIMIT=$OFFLOAD_EXPERT_LIMIT." >&2
            echo "        Required minimum: $MIN_CACHED_EXPERTS" >&2
            exit 1
        fi
        CACHED_NUM_EXPERTS="$MIN_CACHED_EXPERTS"
    fi
fi

# Ascend vendor setup scripts read optional variables without default values.
set +u
# shellcheck disable=SC1090
source "$ACTIVATE_SCRIPT"
set -u

export ASCEND_RT_VISIBLE_DEVICES="$REQUESTED_ASCEND_DEVICES"
export OMOE_QWEN35_TUPLE_SHARD_COMPAT=1
export OMOE_QWEN35_ASCEND_BF16_SSM=1

python3 -m json.tool "$HOTSET_PATH" >/dev/null

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
    --enforce-eager
    --cached-num-experts "$CACHED_NUM_EXPERTS"
    --offload-expert-limit "$OFFLOAD_EXPERT_LIMIT"
)

if is_true "$DYNAMIC_CACHE_ENABLED"; then
    VLLM_ARGS+=(--dynamic-cache-enabled)
else
    VLLM_ARGS+=(--no-dynamic-cache-enabled)
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
echo " O-MoE Ascend Qwen3.5-122B Serving"
echo "============================================================"
echo " Model:                    $MODEL_PATH"
echo " Served model name:        $SERVED_MODEL_NAME"
echo " Hotset:                   $HOTSET_PATH"
echo " Visible NPUs:             $ASCEND_RT_VISIBLE_DEVICES"
echo " Tensor parallel size:     $TP_SIZE"
echo " NPU memory utilization:   $GPU_MEMORY_UTIL"
echo " Max model length:         $MAX_MODEL_LEN"
echo " Cached experts/layer:     $CACHED_NUM_EXPERTS"
echo " Offload expert limit:     $OFFLOAD_EXPERT_LIMIT"
echo " Dynamic cache:            $DYNAMIC_CACHE_ENABLED"
echo " Prefix caching:           $ENABLE_PREFIX_CACHING"
echo " Chunked prefill:          $ENABLE_CHUNKED_PREFILL"
echo " Async scheduling:         $ASYNC_SCHEDULING"
echo " Endpoint:                 http://$HOST:$PORT"
echo "============================================================"

if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info
fi

echo "[INFO] Starting vLLM OpenAI-compatible server..."
exec vllm "${VLLM_ARGS[@]}" "$@"
