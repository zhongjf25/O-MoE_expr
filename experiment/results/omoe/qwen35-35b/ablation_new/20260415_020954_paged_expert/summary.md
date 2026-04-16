# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** omoe
- **Model:** /root/autodl-tmp/models/Qwen3.5-35B-A3B
- **Dataset:** /root/autodl-tmp/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num Prompts:** 1000
- **RR Range:** 2.5 ~ 2.5 (step 1)
- **Timestamp:** 20260415_020954
- **Label:** paged_expert

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.5 | 0.87 | 363707.11 | 688823.23 | 215.45 | 427.71 | 207.87 | 1000 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 2.5 | 204.58 | 127.24 | 0.00 | 0.00 | 180.05 | 699 |