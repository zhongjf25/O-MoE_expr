# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** ktransformers
- **Model:** /root/autodl-tmp/models/Qwen3-30B-A3B
- **Dataset:** sharegpt
- **Random Input/Output:** 2048 / 512
- **Num Prompts:** 100
- **RR Range:** 0.2 ~ 2.2 (step 0.4)
- **Timestamp:** 20260411_235117
- **Label:** kt_CA

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | 0.20 | 1283.36 | 8047.39 | 138.72 | 385.24 | 130.04 | 100 | None |
| 0.6 | 0.46 | 292.90 | 382.07 | 111.93 | 122.76 | 111.19 | 100 | None |
| 1.0 | 0.58 | 319.96 | 421.85 | 123.46 | 135.94 | 125.04 | 100 | None |
| 1.4 | 0.73 | 304.38 | 378.55 | 115.32 | 122.47 | 113.81 | 100 | None |
| 1.8 | 0.77 | 317.97 | 391.92 | 121.92 | 131.79 | 121.71 | 100 | None |
| 2.2 | 0.85 | 323.34 | 391.17 | 120.47 | 132.45 | 117.67 | 100 | None |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 0.2 | 124.44 | 111.93 | 0.00 | 0.00 | 47.16 | 14 |
| 0.6 | 112.95 | 113.20 | 0.00 | 0.00 | 109.74 | 22 |
| 1.0 | 121.82 | 122.96 | 0.00 | 0.00 | 139.01 | 35 |
| 1.4 | 115.56 | 114.59 | 0.00 | 0.00 | 173.73 | 38 |
| 1.8 | 122.56 | 122.09 | 0.00 | 0.00 | 183.68 | 51 |
| 2.2 | 120.65 | 117.46 | 0.00 | 0.00 | 203.88 | 57 |