#!/bin/bash
#
# SGLang Server Startup Script
# Usage: bash start_sglang_server.sh
# Environment variables to override: MODEL_PATH, TP_SIZE, HOST, PORT, MEM_FRACTION, CONTEXT_LENGTH
#

set -e

# qwen3.5-122b server配置
# MODEL_PATH=/root/autodl-tmp/models/qwen35_122b TP_SIZE=8 CONTEXT_LENGTH=4096 bash /root/autodl-tmp/workspace/experiment/scripts/start_sglang_server.sh

# ========== 参数配置 ==========
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen3-30B-A3B}"
TP_SIZE="${TP_SIZE:-2}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MEM_FRACTION="${MEM_FRACTION:-0.95}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"

echo "============================================"
echo "  SGLang Server Configuration"
echo "============================================"
echo "  Model:            $MODEL_PATH"
echo "  Tensor Parallel:  $TP_SIZE"
echo "  Host:Port:        $HOST:$PORT"
echo "  Mem Fraction:     $MEM_FRACTION"
echo "  Context Length:   $CONTEXT_LENGTH"
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

echo "[INFO] Starting SGLang service..."
echo "[INFO] Press Ctrl+C to stop"
echo ""

# ========== CUDA 环境变量 ==========
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
# export MAX_JOBS=2


# 使用 miniconda 的 sglang 环境
source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
conda activate sglang

sglang serve \
    --model-path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --context-length "$CONTEXT_LENGTH" \
    --mem-fraction-static "$MEM_FRACTION" \
    --skip-server-warmup \
    --disable-cuda-graph \
    --attention-backend triton \
    # --disable-custom-all-reduce \


