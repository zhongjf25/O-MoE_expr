# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** sglang
- **Model:** /root/autodl-tmp/models/qwen35_122b
- **Dataset:** sharegpt
- **Random Input/Output:** 2048 / 512
- **Num Prompts:** 100
- **RR Range:** 0.4 ~ 1 (step 0.1)
- **Timestamp:** 20260411_101428
- **Label:** sglang_qwen35_122b

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.4 | 0.11 | 337343.71 | 655705.21 | 73.37 | 76.83 | 73.40 | 100 | None |
| 0.5 | 0.11 | 356167.34 | 696631.23 | 72.93 | 75.80 | 73.12 | 100 | None |
| 0.6 | 0.11 | 368243.98 | 723172.61 | 72.77 | 76.15 | 72.78 | 100 | None |
| 0.7 | 0.11 | 388083.92 | 762975.58 | 74.30 | 77.14 | 74.30 | 100 | None |
| 0.8 | 0.11 | 390841.26 | 765189.58 | 73.02 | 75.68 | 73.13 | 100 | None |
| 0.9 | 0.11 | 395853.79 | 775757.28 | 72.95 | 77.76 | 72.95 | 100 | None |
| 1.0 | 0.11 | 400647.38 | 787163.20 | 73.40 | 76.81 | 73.06 | 100 | None |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 0.4 | 73.12 | 71.83 | 0.00 | 0.00 | 26.77 | 79 |
| 0.5 | 72.81 | 71.73 | 0.00 | 0.00 | 26.89 | 86 |
| 0.6 | 72.67 | 71.25 | 0.00 | 0.00 | 27.02 | 89 |
| 0.7 | 74.24 | 72.98 | 0.00 | 0.00 | 26.47 | 90 |
| 0.8 | 72.98 | 71.66 | 0.00 | 0.00 | 26.89 | 93 |
| 0.9 | 72.81 | 71.46 | 0.00 | 0.00 | 26.96 | 95 |
| 1.0 | 72.84 | 71.57 | 0.00 | 0.00 | 26.92 | 95 |