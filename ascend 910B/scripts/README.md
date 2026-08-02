# Experiment Scripts

Managed service launchers:

- `serve_qwen35_122b_vllm_ascend.sh`: text-only baseline vLLM Ascend,
  TP=8, with 8 GiB of generic vLLM CPU weight offload per NPU so the
  BF16 checkpoint fits on 8 x 32 GiB.
- `serve_qwen35_122b_omoe_ascend.sh`: O-MoE Ascend, TP=8.
- `serve_qwen15_moe_a27b_omoe_ascend.sh`: O-MoE Ascend smoke model.
- `serve_qwen15_moe_legacy.sh`: legacy NVIDIA example.

Experiment entry points:

- `benchmark_qwen35_122b.sh`: shared Qwen3.5 serving benchmark.
- `run_vllm_ascend_qwen35_rr1-5_n1000_n2000.sh`: sequential baseline
  benchmark for 1000 and 2000 prompts at request rates 1 through 5.

Environment:

- `activate_vllm_ascend.sh`: baseline vLLM and vLLM Ascend paths.
- `runtime_vllm_ascend/sitecustomize.py`: environment-only Qwen3.5
  compatibility shims for tuple shard loading and BF16 SSM state.
