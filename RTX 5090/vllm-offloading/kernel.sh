export CUDA_VISIBLE_DEVICES=4,5
PYTHONPATH=/root/autodl-tmp/workspace/vllm-offloading:$PYTHONPATH \
  python benchmarks/kernels/benchmark_offloaded_moe.py \
    --num-experts 128 --num-tokens 2048 --hidden-size 2048 \
    --intermediate-size 768 --topk 8 --cache-hit-ratio 0.95 \
    --mode both --iters 100 --warmup-iters 20