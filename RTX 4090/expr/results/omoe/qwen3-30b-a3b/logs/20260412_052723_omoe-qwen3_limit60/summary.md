# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** omoe
- **Model:** /data/share/models/Qwen3-30B-A3B
- **Dataset:** /root/workspace/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num Prompts:** 1000
- **RR Range:** 2.0 ~ 2.0 (step 1)
- **Timestamp:** 
- **Label:** omoe-qwen3_limit60

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.8 | 1.33 | 36431.59 | 122049.17 | 921.36 | 1095.43 | 929.19 | 887 | 113 |
| 2.0 | 0.92 | 62146.95 | 183226.83 | 924.13 | 1210.97 | 920.45 | 1000 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 1.8 | 981.63 | 973.97 | 0.00 | 0.00 | 202.16 | 409 |
| 2.0 | 970.37 | 965.79 | 0.00 | 0.00 | 186.23 | 505 |