#!/bin/bash
#
# O-MoE Simple Serving Qwen3 启动脚本
# 用法: bash simple_serving_qwen3.sh
# 可用环境变量覆盖:
#   CUDA_VISIBLE_DEVICES  - 使用的GPU编号 (默认: 0,1)
#   GPU_MEMORY_UTIL       - GPU内存利用率 (默认: 0.9)
#   OFFLOAD_EXPERT_LIMIT  - offload-expert-limit (默认: 40)
#   MODEL_PATH            - 模型路径 (默认: /root/autodl-tmp/models/Qwen3-30B-A3B)
#   TP_SIZE               - Tensor Parallel大小 (默认: 2)
#   PORT                  - 服务端口 (默认: 8000)
#

# ========== 参数配置 ==========
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.93}"
OFFLOAD_EXPERT_LIMIT="${OFFLOAD_EXPERT_LIMIT:-20}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen3-30B-A3B}"
TP_SIZE="${TP_SIZE:-2}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

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

PYTHONPATH=/root/autodl-tmp/workspace/vllm-offloading:$PYTHONPATH vllm serve "$MODEL_PATH" \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
    --tensor-parallel-size "$TP_SIZE" \
    --offload-expert \
    --offload-expert-limit "$OFFLOAD_EXPERT_LIMIT" \
    --dynamic-cache-enabled \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --expert-no-copy-compute \
    --no-async-scheduling \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enforce-eager \

# PYTHONPATH=/root/autodl-tmp/workspace/vllm-offloading:$PYTHONPATH vllm serve "$MODEL_PATH" \
#     --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
#     --tensor-parallel-size "$TP_SIZE" \
#     --offload-expert \
#     --offload-expert-limit "$OFFLOAD_EXPERT_LIMIT" \
#     --dynamic-cache-enabled \
#     --no-enable-prefix-caching \
#     --no-enable-chunked-prefill \
#     --no-async-scheduling \
#     --port "$PORT" \
#     --max-model-len "$MAX_MODEL_LEN" \
#     --enforce-eager \