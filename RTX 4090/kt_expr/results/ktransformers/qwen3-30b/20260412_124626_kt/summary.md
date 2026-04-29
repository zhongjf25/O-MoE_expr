# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** ktransformers
- **Model:** /data/share/models/Qwen3-30B-A3B
- **Dataset:** sharegpt
- **Random Input/Output:** 2048 / 512
- **Num Prompts:** 100
- **RR Range:** 2 ~ 2 (step 0.4)
- **Timestamp:** 20260412_124626
- **Label:** kt

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.0 | 0.02 | 111682.30 | 169567.34 | 8628.28 | 21282.40 | 7020.73 | 100 | None |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 2.0 | 8008.19 | 6392.87 | 0.00 | 0.00 | 4.66 | 100 |