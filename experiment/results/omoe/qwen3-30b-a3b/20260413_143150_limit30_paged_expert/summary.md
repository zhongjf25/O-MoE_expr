# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** omoe
- **Model:** /root/autodl-tmp/models/Qwen3-30B-A3B
- **Dataset:** /root/autodl-tmp/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num Prompts:** 1000
- **RR Range:** 6 ~ 6 (step 1)
- **Timestamp:** 20260413_143150
- **Label:** limit30_paged_expert

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 6.0 | 2.11 | 120711.19 | 244806.60 | 157.22 | 317.27 | 148.96 | 1000 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 6.0 | 150.04 | 141.44 | 0.00 | 0.00 | 425.46 | 677 |