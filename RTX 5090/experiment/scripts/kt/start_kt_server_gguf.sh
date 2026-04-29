#!/bin/bash
# KT-Kernel + SGLang server with LLAMAFILE backend for Qwen3-30B-A3B
# Single GPU (RTX 5090), LLAMAFILE CPU backend with BF16 GGUF weights

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH

CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model /root/autodl-tmp/models/Qwen3-30B-A3B \
  --trust-remote-code \
  --mem-fraction-static 0.95 \
  --chunked-prefill-size 4096 \
  --served-model-name Qwen3-30B-A3B \
  --enable-mixed-chunk \
  --kt-method LLAMAFILE \
  --kt-weight-path /root/autodl-tmp/models/Qwen3-30B-A3B-GGUF/BF16 \
  --kt-cpuinfer 104 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 60 \
  --kt-max-deferred-experts-per-token 2 \
  --disable-cuda-graph \
  --skip-server-warmup