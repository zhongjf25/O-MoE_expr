# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** ktransformers
- **Model:** /root/autodl-tmp/models/Qwen3-30B-A3B
- **Dataset:** sharegpt
- **Random Input/Output:** 2048 / 512
- **Num Prompts:** 100
- **RR Range:** 2.6 ~ 2.6 (step 0.4)
- **Timestamp:** 20260413_201906
- **Label:** kt

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.6 | 0.94 | 281.37 | 350.96 | 106.81 | 112.27 | 105.98 | 100 | None |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 2.6 | 106.42 | 105.80 | 0.00 | 0.00 | 224.04 | 56 |