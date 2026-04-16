#!/bin/bash
#
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.77}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen3.5-35B-A3B}"
TP_SIZE="${TP_SIZE:-1}"
PORT="${PORT:-8002}"
CACHE_EXPERT="${CACHE_EXPERT:-40}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

echo "============================================"
echo "  FATE Qwen3.5 Serving Configuration"
echo "============================================"
echo "  Model:                  $MODEL_PATH"
echo "  CUDA_VISIBLE_DEVICES:   $CUDA_VISIBLE_DEVICES"
echo "  GPU Memory Util:        $GPU_MEMORY_UTIL"
echo "  Max Model Length:       $MAX_MODEL_LEN"
echo "  TP Size:                $TP_SIZE"
echo "  Cached Experts:         $CACHE_EXPERT"
echo "  Port:                   $PORT"
echo "============================================"

# 检查模型路径
if [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] Model path not found: $MODEL_PATH"
    exit 1
fi

# 检查 GPU
if command -v nvidia-smi &> /dev/null; then
    echo "[INFO] GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    echo ""
fi

echo "[INFO] Starting Fate Qwen3 service..."
echo "[INFO] Press Ctrl+C to stop"
echo ""

export CUDA_VISIBLE_DEVICES

DS_EXPERT_OFFLOAD=1 \
DS_CACHED_EXPERTS_COUNT="$CACHE_EXPERT" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=/root/autodl-tmp/workspace/ldy/fate-test:$PYTHONPATH \
vllm serve "$MODEL_PATH" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
    --tensor-parallel-size "$TP_SIZE" \
    --enforce-eager \
    --max-model-len "$MAX_MODEL_LEN" \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
