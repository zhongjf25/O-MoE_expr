#!/bin/bash
#
# O-MoE Simple Serving Qwen3.5 启动脚本

# ========== 参数配置 ==========
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.92}"
OFFLOAD_EXPERT_LIMIT="${OFFLOAD_EXPERT_LIMIT:-190}"
MODEL_PATH="${MODEL_PATH:-/data/share/models/qwen35_35b}"
TP_SIZE="${TP_SIZE:-2}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
NUM_CACHE="${NUM_CACHE:-30}"

echo "============================================"
echo "  O-MoE Qwen3 Serving Configuration"
echo "============================================"
echo "  Model:                  $MODEL_PATH"
echo "  CUDA_VISIBLE_DEVICES:   $CUDA_VISIBLE_DEVICES"
echo "  GPU Memory Util:        $GPU_MEMORY_UTIL"
echo "  Max Model Length:       $MAX_MODEL_LEN"
echo "  TP Size:                $TP_SIZE"
echo "  Offload Expert Limit:   $OFFLOAD_EXPERT_LIMIT"
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

echo "[INFO] Starting O-MoE Qwen3 service..."
echo "[INFO] Press Ctrl+C to stop"
echo ""

export CUDA_VISIBLE_DEVICES

PYTHONPATH=/root/workspace/vllm-offloading:$PYTHONPATH vllm serve "$MODEL_PATH" \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
    --tensor-parallel-size "$TP_SIZE" \
    --offload-expert \
    --offload-expert-limit "$OFFLOAD_EXPERT_LIMIT" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --no-async-scheduling \
    --enforce-eager \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --cached-num-experts "$NUM_CACHE" \
    # --dynamic-cache-enabled \
    # --expert-no-copy-compute \
