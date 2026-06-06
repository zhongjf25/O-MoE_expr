# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** fate
- **Model:** /data/share/models/qwen35_35b
- **Dataset:** /root/workspace/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num Prompts:** 50
- **RR Range:** 5 ~ 5 (step 0.2)
- **Timestamp:** 20260414_092633
- **Label:** fate-qwen35-35b

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5.0 | 0.12 | 13324.92 | 17903.97 | 887.34 | 1198.02 | 736.54 | 50 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 5.0 | 914.41 | 809.50 | 0.00 | 0.00 | 25.81 | 50 |