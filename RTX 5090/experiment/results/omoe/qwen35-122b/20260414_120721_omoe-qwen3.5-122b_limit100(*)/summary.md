# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** omoe
- **Model:** /root/autodl-tmp/models/qwen35_122b
- **Dataset:** /root/autodl-tmp/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num Prompts:** 100
- **RR Range:** 5.0 ~ 5.0 (step 1)
- **Timestamp:** 
- **Label:** omoe-qwen3.5-122b_limit100

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.05 | 558712.71 | 976810.75 | 174.94 | 329.44 | 178.06 | 100 | 0 |
| 2.0 | 0.28 | 77166.01 | 175055.17 | 240.28 | 287.84 | 238.00 | 100 | 0 |
| 3.0 | 0.28 | 120409.75 | 210293.64 | 232.12 | 296.36 | 226.82 | 100 | 0 |
| 4.0 | 0.25 | 140446.42 | 248913.20 | 266.06 | 352.54 | 255.53 | 100 | 0 |
| 5.0 | 0.25 | 140540.39 | 248690.03 | 259.94 | 311.84 | 251.49 | 100 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 0.1 | 149.23 | 149.50 | 0.00 | 0.00 | 10.86 | 64 |
| 2.0 | 239.75 | 227.46 | 0.00 | 0.00 | 64.48 | 87 |
| 3.0 | 225.51 | 210.45 | 0.00 | 0.00 | 62.63 | 100 |
| 4.0 | 260.05 | 241.93 | 0.00 | 0.00 | 56.82 | 100 |
| 5.0 | 252.60 | 237.09 | 0.00 | 0.00 | 57.70 | 100 |