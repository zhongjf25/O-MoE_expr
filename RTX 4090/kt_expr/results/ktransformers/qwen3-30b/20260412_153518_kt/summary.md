# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** ktransformers
- **Model:** /data/share/models/Qwen3-30B-A3B
- **Dataset:** sharegpt
- **Random Input/Output:** 2048 / 512
- **Num Prompts:** 50
- **RR Range:** 2 ~ 2 (step 0.4)
- **Timestamp:** 20260412_153518
- **Label:** kt

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.0 | 0.01 | 64666.21 | 109073.90 | 5942.58 | 12873.02 | 5181.81 | 50 | None |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 2.0 | 5810.66 | 4202.46 | 0.00 | 0.00 | 2.12 | 50 |