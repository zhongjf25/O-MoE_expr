export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export CUDA_VISIBLE_DEVICES=7
export SGLANG_DISABLE_CUDNN_CHECK=1


python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 8036 \
  --model /root/autodl-tmp/models/Qwen3.5-35B-A3B \
  --trust-remote-code \
  --mem-fraction-static 0.95 \
  --chunked-prefill-size 4096 \
  --served-model-name Qwen3.5-35B-A3B \
  --enable-mixed-chunk \
  --kt-method AMXINT8 \
  --kt-weight-path /root/autodl-tmp/models/Qwen3.5-35B-A3B-INT8 \
  --kt-cpuinfer 104 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 40 \
  --kt-max-deferred-experts-per-token 2 \
  --disable-cuda-graph \
  --attention-backend triton \
  --skip-server-warmup

# AssertionError: triton or trtllm_mha backend are the only supported backends on Blackwell GPUs for hybrid GDN models, use --attention-backend triton or --attention-backend trtllm_mha to specify the backend.