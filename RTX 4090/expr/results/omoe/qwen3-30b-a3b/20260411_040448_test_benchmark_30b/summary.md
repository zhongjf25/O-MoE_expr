# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** omoe
- **Model:** /data/share/models/Qwen3-30B-A3B
- **Dataset:** /root/workspace/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num Prompts:** 50
- **RR Range:** 5 ~ 6 (step 1)
- **Timestamp:** 20260411_040448
- **Label:** test_benchmark_30b

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5.0 | 0.26 | 5598.75 | 9750.18 | 434.98 | 1245.14 | 324.47 | 50 | 0 |
| 6.0 | 0.28 | 4038.95 | 9612.46 | 392.07 | 593.54 | 317.63 | 50 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 5.0 | 384.22 | 311.26 | 0.00 | 0.00 | 53.12 | 50 |
| 6.0 | 388.69 | 318.93 | 0.00 | 0.00 | 57.40 | 49 |