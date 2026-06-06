# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** omoe
- **Model:** /data/share/models/Qwen3-30B-A3B
- **Dataset:** /root/workspace/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num Prompts:** 250
- **RR Range:** 2 ~ 2 (step 1)
- **Timestamp:** 20260415_061643
- **Label:** paged_expert_rr2

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.0 | 0.29 | 2108.23 | 3007.23 | 1332.48 | 1489.94 | 1257.81 | 250 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 2.0 | 1351.97 | 1314.65 | 0.00 | 0.00 | 63.94 | 189 |