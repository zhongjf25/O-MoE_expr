# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** llama
- **Model:** /root/autodl-tmp/models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf
- **Dataset:** sharegpt
- **Num Prompts:** 100
- **RR Range:** 1 ~ 1 (step 1)
- **Timestamp:** 20260405_192412
- **Label:** llama

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.66 | 10277.02 | 50633.38 | 39.98 | 103.86 | 12.23 | 100 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Total Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 1.0 | 22.59 | 8.98 | 30543.15 | 53072.38 | 530.62 | 40 |