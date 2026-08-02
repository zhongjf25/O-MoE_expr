# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** omoe-ascend
- **Model:** /home/ma-user/work/models/Qwen3.5-122B-A10B
- **Served model:** Qwen3.5-122B-A10B
- **Dataset:** /home/ma-user/work/models/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num prompts:** 1000
- **Request rate:** {'start': 5.0, 'end': 5.0, 'step': 1.0}
- **Timestamp:** 20260801_111031
- **Label:** omoe-util0.92-offload100-rr5-n1000

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5.0 | 0.16 | 1695161.17 | 4341194.22 | 6992.02 | 9838.85 | 6327.50 | 1000 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 5.0 | 7133.81 | 5989.87 | N/A | N/A | 33.82 | 964 |
