# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** omoe
- **Model:** /root/autodl-tmp/models/Qwen3-30B-A3B
- **Dataset:** /root/autodl-tmp/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num Prompts:** 1000
- **RR Range:** 11 ~ 20 (step 3)
- **Timestamp:** 20260407_232012
- **Label:** limit40

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 11.0 | 3.86 | 21741.59 | 61332.49 | 210.35 | 263.61 | 212.18 | 1000 | 0 |
| 14.0 | 3.72 | 32220.20 | 86365.78 | 222.66 | 296.51 | 223.06 | 1000 | 0 |
| 17.0 | 3.81 | 35125.97 | 93183.64 | 218.52 | 285.70 | 219.21 | 1000 | 0 |
| 20.0 | 3.80 | 40867.33 | 102882.89 | 217.78 | 287.17 | 219.17 | 1000 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 11.0 | 221.08 | 216.90 | 0.00 | 0.00 | 779.98 | 610 |
| 14.0 | 228.86 | 224.50 | 0.00 | 0.00 | 749.92 | 714 |
| 17.0 | 226.26 | 221.49 | 0.00 | 0.00 | 769.62 | 746 |
| 20.0 | 226.49 | 223.29 | 0.00 | 0.00 | 766.51 | 776 |