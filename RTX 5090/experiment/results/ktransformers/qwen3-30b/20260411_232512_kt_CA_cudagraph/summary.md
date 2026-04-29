# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** ktransformers
- **Model:** /root/autodl-tmp/models/Qwen3-30B-A3B
- **Dataset:** sharegpt
- **Random Input/Output:** 2048 / 512
- **Num Prompts:** 100
- **RR Range:** 0.2 ~ 2.2 (step 0.4)
- **Timestamp:** 20260411_232512
- **Label:** kt_CA

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | 0.21 | 1166.54 | 8140.38 | 61.73 | 262.73 | 54.14 | 100 | None |
| 0.6 | 0.53 | 321.83 | 590.40 | 84.06 | 102.93 | 83.82 | 100 | None |
| 1.0 | 0.73 | 302.15 | 487.19 | 99.19 | 132.71 | 97.32 | 100 | None |
| 1.4 | 0.85 | 323.62 | 450.24 | 113.57 | 129.50 | 110.90 | 100 | None |
| 1.8 | 0.92 | 312.65 | 455.38 | 115.93 | 133.60 | 112.46 | 100 | None |
| 2.2 | 0.99 | 314.10 | 470.94 | 119.74 | 139.93 | 114.65 | 100 | None |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 0.2 | 53.77 | 44.31 | 0.00 | 0.00 | 50.25 | 9 |
| 0.6 | 87.20 | 81.44 | 0.00 | 0.00 | 126.05 | 21 |
| 1.0 | 98.61 | 95.64 | 0.00 | 0.00 | 174.43 | 32 |
| 1.4 | 117.62 | 118.12 | 0.00 | 0.00 | 202.22 | 39 |
| 1.8 | 118.84 | 116.84 | 0.00 | 0.00 | 219.83 | 49 |
| 2.2 | 122.37 | 121.23 | 0.00 | 0.00 | 235.23 | 57 |