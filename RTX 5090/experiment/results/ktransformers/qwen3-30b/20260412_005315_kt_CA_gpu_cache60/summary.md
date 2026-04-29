# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** ktransformers
- **Model:** /root/autodl-tmp/models/Qwen3-30B-A3B
- **Dataset:** sharegpt
- **Random Input/Output:** 2048 / 512
- **Num Prompts:** 100
- **RR Range:** 1 ~ 10 (step 3)
- **Timestamp:** 20260412_005315
- **Label:** kt_CA

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.40 | 48821.62 | 90482.26 | 149.84 | 365.85 | 138.60 | 100 | None |
| 4.0 | 0.41 | 77577.63 | 147974.40 | 152.03 | 554.65 | 138.59 | 100 | None |
| 7.0 | 0.41 | 83113.69 | 156643.58 | 145.08 | 514.82 | 134.22 | 100 | None |
| 10.0 | 0.41 | 85445.05 | 160511.37 | 145.31 | 506.62 | 134.20 | 100 | None |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 1.0 | 137.63 | 108.69 | 0.00 | 0.00 | 96.32 | 72 |
| 4.0 | 138.65 | 108.51 | 0.00 | 0.00 | 97.92 | 95 |
| 7.0 | 133.52 | 104.57 | 0.00 | 0.00 | 98.31 | 98 |
| 10.0 | 133.41 | 104.86 | 0.00 | 0.00 | 98.66 | 99 |