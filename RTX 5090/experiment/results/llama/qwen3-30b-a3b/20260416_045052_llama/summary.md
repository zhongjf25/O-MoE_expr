# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** llama
- **Model:** /root/autodl-tmp/models/Qwen3-30B-A3B-GGUF/BF16/
- **Dataset:** /root/autodl-tmp/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num Prompts:** 100
- **RR Range:** 0.2 ~ 2.6 (step 0.4)
- **Timestamp:** 20260416_045052
- **Label:** llama

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | 0.18 | 6952.51 | 22435.77 | 138.39 | 583.71 | 109.80 | 100 | 0 |
| 0.6 | 0.38 | 19053.97 | 64716.90 | 85.58 | 162.87 | 84.20 | 100 | 0 |
| 1.0 | 0.39 | 48373.66 | 123854.30 | 84.28 | 111.08 | 85.62 | 100 | 0 |
| 1.4 | 0.38 | 60256.02 | 150191.76 | 84.75 | 112.96 | 87.07 | 100 | 0 |
| 1.8 | 0.39 | 69848.28 | 165667.92 | 85.99 | 116.22 | 86.81 | 100 | 0 |
| 2.2 | 0.39 | 74480.49 | 175402.15 | 85.43 | 114.32 | 87.22 | 100 | 0 |
| 2.6 | 0.37 | 79573.23 | 191864.06 | 88.82 | 119.76 | 90.65 | 100 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 0.2 | 105.07 | 74.88 | 0.00 | 0.00 | 39.42 | 12 |
| 0.6 | 83.78 | 77.85 | 0.00 | 0.00 | 84.28 | 31 |
| 1.0 | 83.12 | 78.53 | 0.00 | 0.00 | 85.17 | 63 |
| 1.4 | 83.35 | 79.46 | 0.00 | 0.00 | 84.42 | 73 |
| 1.8 | 84.65 | 79.26 | 0.00 | 0.00 | 85.27 | 78 |
| 2.2 | 83.63 | 79.39 | 0.00 | 0.00 | 85.11 | 84 |
| 2.6 | 86.04 | 82.83 | 0.00 | 0.00 | 81.91 | 87 |