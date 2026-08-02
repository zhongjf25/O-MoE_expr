# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** omoe-ascend
- **Model:** /home/ma-user/work/models/Qwen3.5-122B-A10B
- **Served model:** Qwen3.5-122B-A10B
- **Dataset:** /home/ma-user/work/models/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num prompts:** 250
- **Request rate:** {'start': 1.0, 'end': 5.0, 'step': 1.0}
- **Timestamp:** 20260729_154733
- **Label:** offload100-rr1-5-n250

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.14 | 10404.88 | 37115.94 | 3953.49 | 6547.07 | 2822.43 | 250 | 0 |
| 2.0 | 0.18 | 17930.80 | 83362.28 | 3503.97 | 6785.72 | 2242.98 | 250 | 0 |
| 3.0 | 0.19 | 26228.00 | 96257.04 | 3101.47 | 6846.84 | 2060.85 | 250 | 0 |
| 4.0 | 0.20 | 21696.40 | 75865.27 | 2623.34 | 5239.54 | 1937.12 | 250 | 0 |
| 5.0 | 0.20 | 28126.48 | 81299.66 | 2506.75 | 4893.10 | 1906.78 | 250 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 1.0 | 3430.95 | 1917.42 | N/A | N/A | 31.55 | 212 |
| 2.0 | 2760.08 | 1861.52 | N/A | N/A | 40.44 | 236 |
| 3.0 | 2418.51 | 1847.70 | N/A | N/A | 42.56 | 245 |
| 4.0 | 2231.99 | 1844.14 | N/A | N/A | 44.18 | 245 |
| 5.0 | 2184.65 | 1869.80 | N/A | N/A | 44.36 | 248 |
