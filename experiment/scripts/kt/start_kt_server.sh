export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH

python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model /root/autodl-tmp/models/Qwen3-30B-A3B \
  --trust-remote-code \
  --mem-fraction-static 0.95 \
  --chunked-prefill-size 4096 \
  --served-model-name Qwen3-30B-A3B \
  --enable-mixed-chunk \
  --kt-method AMXINT8 \
  --kt-weight-path /root/autodl-tmp/models/Qwen3-30B-A3B-INT8 \
  --kt-cpuinfer 104 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 60 \
  --kt-max-deferred-experts-per-token 2 \
  --disable-cuda-graph \
  --skip-server-warmup