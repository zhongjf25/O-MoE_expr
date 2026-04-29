# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** omoe
- **Model:** /root/autodl-tmp/models/qwen35_122b
- **Dataset:** /root/autodl-tmp/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num Prompts:** 250
- **RR Range:** 5.0 ~ 5.0 (step 1)
- **Timestamp:** 
- **Label:** omoe-qwen3.5-122b_limit100

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.29 | 262248.81 | 526442.15 | 234.33 | 292.72 | 231.39 | 250 | 0 |
| 2.0 | 0.34 | 261380.58 | 514741.58 | 242.70 | 311.53 | 239.24 | 250 | 0 |
| 3.0 | 0.33 | 308338.30 | 573621.76 | 244.46 | 324.66 | 238.73 | 250 | 0 |
| 4.0 | 0.34 | 294670.40 | 557256.59 | 229.47 | 274.52 | 227.11 | 250 | 0 |
| 5.0 | 0.32 | 317305.42 | 602336.18 | 243.16 | 301.45 | 240.75 | 250 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 1.0 | 239.46 | 225.11 | 0.00 | 0.00 | 64.39 | 177 |
| 2.0 | 256.63 | 239.15 | 0.00 | 0.00 | 75.82 | 213 |
| 3.0 | 236.79 | 226.85 | 0.00 | 0.00 | 73.78 | 245 |
| 4.0 | 226.25 | 214.82 | 0.00 | 0.00 | 76.53 | 250 |
| 5.0 | 238.87 | 227.25 | 0.00 | 0.00 | 71.95 | 250 |