#!/bin/bash
# Llama.cpp Server 启动脚本
# 使用 BF16 GGUF 权重（58GB），单卡推理 + MoE Expert CPU 卸载
#
# Qwen3-30B-A3B MoE: 128 experts, 8 active, BF16, 58GB
# 单卡 RTX 5090 = 32GB VRAM
#
# 解霸策略：
#   --cpu-moe       将所有 MoE Expert 权保留在 CPU（只留 8 个 active expert 在 GPU 计算）
#                   → 大幅减少 GPU VRAM 占用，30B MoE 仅需 ~10GB GPU 显存（attention + 共享 expert）
#   -ngl 99         所有层尽可能放在 GPU（剩余不fit的自动 offload）
#   --cont-batching 连续批处理，最大化 GPU 利用率
#   -fa auto        Flash Attention 加速长序列
#   -b 2048 -ub 512 大 batch 提升吞吐
#
set -x

# 模型路径
MODEL_PATH="/root/autodl-tmp/models/Qwen3-30B-A3B-GGUF/BF16/Qwen3-30B-A3B-BF16-00001-of-00002.gguf"

# 启动服务（单卡，CUDA_VISIBLE_DEVICES=0）
CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/workspace/llama.cpp/build/bin/llama-server \
  -m "$MODEL_PATH" \
  -ngl auto \
  --port 8080 \
  --host 0.0.0.0 \
  --numa numactl \
  --parallel 8 \
  --ctx-size 32768 \
  -fa auto \
  --cont-batching \
  # --cpu-moe \
  # -t 64 \
  # -tb 64 \
  # -b 2048 \
  # -ub 512 \