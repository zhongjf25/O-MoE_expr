# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** sglang
- **Model:** /root/autodl-tmp/models/qwen35_122b
- **Dataset:** sharegpt
- **Random Input/Output:** 2048 / 512
- **Num Prompts:** 500
- **RR Range:** 0.3 ~ 0.3 (step 0.05)
- **Timestamp:** 20260405_144930
- **Label:** sglang

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 0.12 | 1299711.71 | 2411171.29 | 75.13 | 79.83 | 75.17 | 500 | None |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 0.3 | 74.78 | 73.52 | 0.00 | 0.00 | 26.44 | 305 |