# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** omoe
- **Model:** /data/share/models/Qwen3-30B-A3B
- **Dataset:** /root/workspace/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num Prompts:** 1000
- **RR Range:** 1.8 ~ 1.8 (step 1)
- **Timestamp:** 
- **Label:** omoe-qwen3_limit60

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.8 | 0.91 | 48420.92 | 143108.77 | 919.81 | 1171.18 | 917.34 | 1000 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 1.8 | 972.78 | 965.15 | 0.00 | 0.00 | 184.03 | 465 |