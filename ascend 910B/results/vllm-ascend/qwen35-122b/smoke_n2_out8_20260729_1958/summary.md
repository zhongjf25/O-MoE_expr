# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** vllm-ascend
- **Model:** /home/ma-user/work/models/Qwen3.5-122B-A10B
- **Served model:** Qwen3.5-122B-A10B
- **Dataset:** /home/ma-user/work/models/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num prompts:** 2
- **Request rate:** {'start': 1.0, 'end': 1.0, 'step': 1.0}
- **Timestamp:** 20260729_195905
- **Label:** smoke-n2-out8

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.12 | 2120.47 | 2303.67 | 1704.49 | 1724.37 | 1704.49 | 2 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 1.0 | 1704.49 | 1352.00 | N/A | N/A | 0.99 | 2 |
