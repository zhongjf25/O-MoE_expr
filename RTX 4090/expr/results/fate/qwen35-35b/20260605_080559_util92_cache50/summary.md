# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** fate
- **Model:** /data/share/models/qwen35_35b
- **Dataset:** /root/workspace/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num Prompts:** 1000
- **RR Range:** 1 ~ 8 (step 1)
- **Timestamp:** 20260605_080559
- **Label:** util92_cache50

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.14 | 3398.56 | 4670.44 | 1358.49 | 1522.94 | 1315.36 | 140 | 860 |
| 2.0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 1000 |
| 3.0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 1000 |
| 4.0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 1000 |
| 5.0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 1000 |
| 6.0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 1000 |
| 7.0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 1000 |
| 8.0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 1000 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 1.0 | 1397.62 | 1385.90 | 0.00 | 0.00 | 5.41 | 97 |
| 2.0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 |
| 3.0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 |
| 4.0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 |
| 5.0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 |
| 6.0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 |
| 7.0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 |
| 8.0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 |