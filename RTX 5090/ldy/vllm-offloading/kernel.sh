export CUDA_VISIBLE_DEVICES=4,5,6,7
PYTHONPATH=/root/autodl-tmp/workspace/vllm-offloading:$PYTHONPATH \
  python benchmarks/kernels/benchmark_offloaded_moe.py \
    --num-experts 128 --num-tokens 512 --hidden-size 4096 \
    --intermediate-size 2048 --topk 8 --cache-hit-ratio 0.7 \
    --mode both --iters 100 --warmup-iters 20