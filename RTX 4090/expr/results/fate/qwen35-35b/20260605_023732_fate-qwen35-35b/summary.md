# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** fate
- **Model:** /data/share/models/qwen35_35b
- **Dataset:** /root/workspace/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num Prompts:** 250
- **RR Range:** 8 ~ 8 (step 1)
- **Timestamp:** 20260605_023732
- **Label:** fate-qwen35-35b

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8.0 | 1.89 | 4471.87 | 15339.33 | 218.42 | 310.55 | 182.05 | 250 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 8.0 | 212.75 | 181.26 | 0.00 | 0.00 | 423.89 | 188 |