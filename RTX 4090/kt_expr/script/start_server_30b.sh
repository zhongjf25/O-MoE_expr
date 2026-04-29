#!/bin/bash
# KT-Kernel + SGLang server with LLAMAFILE backend for Qwen3-30B-A3B

export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH

CUDA_VISIBLE_DEVICES=1,2 python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model /data/share/models/Qwen3-30B-A3B \
  --trust-remote-code \
  --mem-fraction-static 0.93 \
  --chunked-prefill-size 4096 \
  --served-model-name Qwen3-30B-A3B \
  --enable-mixed-chunk \
  --kt-method LLAMAFILE \
  --kt-weight-path /data/share/models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf \
  --kt-cpuinfer 76 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 40 \
  --kt-max-deferred-experts-per-token 6 \
  --disable-cuda-graph \
  --tensor-parallel-size 2 \
#   --skip-server-warmup