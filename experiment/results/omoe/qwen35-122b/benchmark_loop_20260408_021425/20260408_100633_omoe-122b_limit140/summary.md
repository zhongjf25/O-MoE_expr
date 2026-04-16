# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** omoe
- **Model:** /root/autodl-tmp/models/qwen35_122b
- **Dataset:** /root/autodl-tmp/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num Prompts:** 200
- **RR Range:** 1 ~ 1 (step 1)
- **Timestamp:** 20260408_100633
- **Label:** omoe-122b_limit140

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.19 | 375382.88 | 772897.43 | 198.01 | 272.66 | 195.47 | 200 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 1.0 | 192.82 | 182.78 | 0.00 | 0.00 | 43.19 | 166 |