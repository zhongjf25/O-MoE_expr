#!/bin/bash
#
# vLLM 双卡 RTX 5090 启动脚本
# 用法: bash start_vllm_server.sh
# 可用环境变量覆盖: MODEL_PATH, TP_SIZE, HOST, PORT
#

# GPU_MEMORY_UTIL=0.969 MAX_MODEL_LEN=1024 TP_SIZE=8 MODEL_PATH bash /root/autodl-tmp/workspace/experiment/scripts/start_vllm_server.sh
# GPU_MEMORY_UTIL=0.969 MAX_MODEL_LEN=1024 TP_SIZE=8 MODEL_PATH=/root/autodl-tmp/models/qwen35_122b bash /root/autodl-tmp/workspace/experiment/scripts/start_vllm_server.sh
set -e

# ========== 参数配置 ==========
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen3-30B-A3B}"
TP_SIZE="${TP_SIZE:-2}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.95}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

echo "============================================"
echo "  vLLM Server Configuration"
echo "============================================"
echo "  Model:          $MODEL_PATH"
echo "  Tensor Parallel: $TP_SIZE"
echo "  Host:Port:      $HOST:$PORT"
echo "  GPU Memory:     $GPU_MEMORY_UTIL"
echo "  Max Model Len:  $MAX_MODEL_LEN"
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

echo "[INFO] Starting vLLM service..."
echo "[INFO] Press Ctrl+C to stop"
echo ""

vllm serve "$MODEL_PATH" \
    --tensor-parallel-size "$TP_SIZE" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --no-enable-prefix-caching \
    --no-async-scheduling \
    --max_num_batched_tokens 1024 \
    --disable-custom-all-reduce \
    # --enforce-eager \