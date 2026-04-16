# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** omoe
- **Model:** /root/autodl-tmp/models/Qwen3-30B-A3B
- **Dataset:** /root/autodl-tmp/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num Prompts:** 1000
- **RR Range:** 10 ~ 10 (step 1)
- **Timestamp:** 20260408_154111
- **Label:** omoe-qwen3_limit25

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10.0 | 3.86 | 17278.19 | 51471.56 | 208.53 | 256.23 | 209.94 | 1000 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 10.0 | 218.96 | 213.53 | 0.00 | 0.00 | 779.77 | 552 |