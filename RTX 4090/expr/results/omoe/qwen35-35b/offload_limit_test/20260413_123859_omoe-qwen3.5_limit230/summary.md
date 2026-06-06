# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** omoe
- **Model:** /data/share/models/qwen35_35b
- **Dataset:** /root/workspace/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num Prompts:** 1000
- **RR Range:** 10 ~ 10 (step 1)
- **Timestamp:** 20260413_123859
- **Label:** omoe-qwen3.5_limit230

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10.0 | 2.06 | 120160.34 | 268497.68 | 446.26 | 685.37 | 424.22 | 1000 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 10.0 | 466.48 | 328.62 | 0.00 | 0.00 | 428.91 | 854 |