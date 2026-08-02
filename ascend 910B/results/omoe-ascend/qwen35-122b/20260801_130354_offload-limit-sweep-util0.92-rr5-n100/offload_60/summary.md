# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** omoe-ascend
- **Model:** /home/ma-user/work/models/Qwen3.5-122B-A10B
- **Served model:** Qwen3.5-122B-A10B
- **Dataset:** /home/ma-user/work/models/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num prompts:** 100
- **Request rate:** {'start': 5.0, 'end': 5.0, 'step': 1.0}
- **Timestamp:** 20260801_133626
- **Label:** offload-60-util0.92-rr5-n100

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5.0 | 0.14 | 19551.13 | 33175.23 | 1775.15 | 6820.93 | 1126.56 | 100 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 5.0 | 1306.66 | 1066.32 | N/A | N/A | 30.88 | 100 |
