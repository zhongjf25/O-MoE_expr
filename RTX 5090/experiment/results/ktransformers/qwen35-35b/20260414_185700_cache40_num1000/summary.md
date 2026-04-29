# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** ktransformers
- **Model:** /root/autodl-tmp/models/Qwen3.5-35B-A3B
- **Dataset:** sharegpt
- **Random Input/Output:** 2048 / 512
- **Num Prompts:** 1000
- **RR Range:** 3.5 ~ 3.5 (step 1.5)
- **Timestamp:** 20260414_185700
- **Label:** cache40_num1000

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3.5 | 1.31 | 188178.60 | 394175.48 | 129.57 | 173.91 | 128.48 | 1000 | None |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 3.5 | 128.68 | 114.24 | 0.00 | 0.00 | 263.36 | 603 |