# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** omoe-ascend
- **Model:** /home/ma-user/work/models/Qwen3.5-122B-A10B
- **Served model:** Qwen3.5-122B-A10B
- **Dataset:** /home/ma-user/work/models/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num prompts:** 1
- **Request rate:** {'start': 1.0, 'end': 1.0, 'step': 1.0}
- **Timestamp:** 20260729_150724
- **Label:** smoke

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.17 | 1534.17 | 1534.17 | 490.12 | 490.12 | 490.12 | 1 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 1.0 | 490.12 | 485.96 | N/A | N/A | 1.34 | 1 |
