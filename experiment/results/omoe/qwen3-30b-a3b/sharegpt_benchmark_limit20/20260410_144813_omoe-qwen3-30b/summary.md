# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** omoe
- **Model:** /root/autodl-tmp/models/Qwen3-30B-A3B
- **Dataset:** /root/autodl-tmp/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num Prompts:** 1000
- **RR Range:** 4 ~ 4 (step 1)
- **Timestamp:** 20260410_152740
- **Label:** omoe-qwen3-30b

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.70 | 165.36 | 270.26 | 97.37 | 107.26 | 97.22 | 725 | 275 |
| 2.0 | 1.83 | 189.80 | 399.09 | 103.07 | 115.80 | 103.07 | 1000 | 0 |
| 3.0 | 2.58 | 252.18 | 667.89 | 124.27 | 143.26 | 124.16 | 1000 | 0 |
| 4.0 | 3.13 | 354.23 | 1233.79 | 146.84 | 179.49 | 146.28 | 999 | 1 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 1.0 | 97.14 | 94.04 | 0.00 | 0.00 | 136.34 | 34 |
| 2.0 | 102.83 | 99.85 | 0.00 | 0.00 | 368.83 | 64 |
| 3.0 | 125.89 | 121.56 | 0.00 | 0.00 | 521.02 | 110 |
| 4.0 | 152.14 | 144.75 | 0.00 | 0.00 | 632.52 | 153 |