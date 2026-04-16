# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** sglang
- **Model:** /root/autodl-tmp/models/qwen35_122b
- **Dataset:** sharegpt
- **Random Input/Output:** 2048 / 512
- **Num Prompts:** 100
- **RR Range:** 0.1 ~ 0.3 (step 0.1)
- **Timestamp:** 20260414_005930
- **Label:** sglang

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.10 | 34362.56 | 74895.30 | 74.63 | 77.52 | 74.50 | 100 | None |
| 0.2 | 0.11 | 228828.26 | 442435.74 | 74.04 | 77.45 | 73.88 | 100 | None |
| 0.3 | 0.11 | 293703.96 | 582319.73 | 73.36 | 75.96 | 73.33 | 100 | None |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 0.1 | 74.71 | 73.66 | 0.00 | 0.00 | 24.71 | 15 |
| 0.2 | 73.92 | 72.82 | 0.00 | 0.00 | 26.24 | 51 |
| 0.3 | 73.14 | 71.90 | 0.00 | 0.00 | 26.60 | 68 |