# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** omoe
- **Model:** /data/share/models/qwen35_35b
- **Dataset:** /root/workspace/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num Prompts:** 1000
- **RR Range:** 10 ~ 10 (step 1)
- **Timestamp:** 20260412_073206
- **Label:** omoe-qwen3.5_limit170

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10.0 | 4.73 | 29624.04 | 51502.27 | 170.54 | 217.10 | 161.55 | 1000 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 10.0 | 175.57 | 164.66 | 0.00 | 0.00 | 983.65 | 619 |