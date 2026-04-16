# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** ktransformers
- **Model:** /root/autodl-tmp/models/Qwen3.5-35B-A3B
- **Dataset:** sharegpt
- **Random Input/Output:** 2048 / 512
- **Num Prompts:** 1000
- **RR Range:** 2.5 ~ 4 (step 1.5)
- **Timestamp:** 20260414_175313
- **Label:** cache40_num1000

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.5 | 1.31 | 129274.65 | 280354.02 | 129.08 | 186.06 | 127.75 | 1000 | None |
| 4.0 | 1.32 | 200480.57 | 422784.86 | 137.24 | 175.85 | 127.44 | 1000 | None |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 2.5 | 127.45 | 112.80 | 0.00 | 0.00 | 264.21 | 432 |
| 4.0 | 127.17 | 113.05 | 0.00 | 0.00 | 265.79 | 657 |