# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** fate
- **Model:** /data/share/models/qwen35_35b
- **Dataset:** /root/workspace/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num Prompts:** 1000
- **RR Range:** 8 ~ 8 (step 1)
- **Timestamp:** 20260604_165416
- **Label:** fate-qwen35-35b

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8.0 | 3.60 | 22813.29 | 64430.31 | 256.11 | 293.30 | 244.95 | 1000 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 8.0 | 263.48 | 246.41 | 0.00 | 0.00 | 745.14 | 566 |