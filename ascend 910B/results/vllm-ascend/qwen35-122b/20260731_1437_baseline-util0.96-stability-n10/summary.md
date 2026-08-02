# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** vllm-ascend
- **Model:** /home/ma-user/work/models/Qwen3.5-122B-A10B
- **Served model:** Qwen3.5-122B-A10B
- **Dataset:** /home/ma-user/work/models/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num prompts:** 10
- **Request rate:** {'start': 5.0, 'end': 5.0, 'step': 1.0}
- **Timestamp:** 20260731_143815
- **Label:** baseline-util0.96-stability-n10

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5.0 | 0.01 | 72224.30 | 73838.23 | 1815.28 | 4003.50 | 1311.35 | 10 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 5.0 | 1388.81 | 1211.68 | N/A | N/A | 2.51 | 10 |
