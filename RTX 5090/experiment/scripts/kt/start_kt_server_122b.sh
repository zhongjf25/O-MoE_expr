export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export SGLANG_DISABLE_CUDNN_CHECK=1

python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model /root/autodl-tmp/models/qwen35_122b \
  --trust-remote-code \
  --mem-fraction-static 0.95 \
  --tp 8 \
  --chunked-prefill-size 4096 \
  --served-model-name Qwen3-30B-A3B \
  --enable-mixed-chunk \
  --attention-backend triton \
  --kt-method AMXINT8 \
  --kt-weight-path /root/autodl-tmp/models/qwen35_122b-INT8 \
  --kt-cpuinfer 104 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 60 \
  --kt-max-deferred-experts-per-token 2 \
  --disable-cuda-graph \
  --skip-server-warmup