# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** omoe
- **Model:** /data/share/models/qwen35_35b
- **Dataset:** /root/workspace/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num Prompts:** 100
- **RR Range:** 3 ~ 3 (step 1)
- **Timestamp:** 20260412_064035
- **Label:** omoe-qwen35

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3.0 | 1.11 | 4789.96 | 14312.75 | 110.01 | 682.75 | 87.68 | 100 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 3.0 | 89.75 | 76.74 | 0.00 | 0.00 | 252.11 | 64 |