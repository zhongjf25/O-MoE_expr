# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** fate
- **Model:** /data/share/models/qwen35_35b
- **Dataset:** /root/workspace/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num Prompts:** 1000
- **RR Range:** 8 ~ 8 (step 1)
- **Timestamp:** 20260605_073432
- **Label:** fate-qwen35-35b

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8.0 | 0.56 | 353362.24 | 906355.30 | 1668.74 | 1941.93 | 1619.44 | 1000 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 8.0 | 1670.43 | 1647.72 | 0.00 | 0.00 | 113.15 | 880 |