# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** llama
- **Model:** /root/autodl-tmp/models/Qwen3-30B-A3B-GGUF/BF16/
- **Dataset:** /root/autodl-tmp/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num Prompts:** 100
- **RR Range:** 2.6 ~ 2.6 (step 0.4)
- **Timestamp:** 20260416_000427
- **Label:** llama

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.6 | 0.13 | 283075.33 | 547823.64 | 232.60 | 1197.63 | 255.41 | 100 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 2.6 | 180.62 | 90.07 | 0.00 | 0.00 | 28.39 | 93 |