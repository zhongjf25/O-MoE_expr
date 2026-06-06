# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** omoe
- **Model:** /data/share/models/qwen35_35b
- **Dataset:** /root/workspace/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num Prompts:** 1000
- **RR Range:** 9.0 ~ 9.0 (step 1)
- **Timestamp:** 
- **Label:** omoe-qwen3.5_limit120

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.97 | 372.13 | 5434.45 | 88.01 | 115.47 | 87.46 | 999 | 1 |
| 3.0 | 2.68 | 878.18 | 13373.20 | 107.73 | 154.79 | 106.03 | 1000 | 0 |
| 5.0 | 4.12 | 1718.13 | 15204.55 | 116.49 | 151.27 | 114.52 | 1000 | 0 |
| 7.0 | 4.94 | 3628.80 | 16413.11 | 155.64 | 198.08 | 149.39 | 1000 | 0 |
| 9.0 | 4.84 | 19782.54 | 33181.35 | 164.10 | 201.56 | 157.02 | 1000 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 1.0 | 86.86 | 88.47 | 0.00 | 0.00 | 200.96 | 38 |
| 3.0 | 107.86 | 103.12 | 0.00 | 0.00 | 556.55 | 102 |
| 5.0 | 117.34 | 114.05 | 0.00 | 0.00 | 856.98 | 174 |
| 7.0 | 158.36 | 154.09 | 0.00 | 0.00 | 1026.93 | 286 |
| 9.0 | 171.50 | 159.84 | 0.00 | 0.00 | 1006.62 | 505 |