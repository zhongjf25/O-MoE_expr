# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** vllm-ascend
- **Model:** /home/ma-user/work/models/Qwen3.5-122B-A10B
- **Served model:** Qwen3.5-122B-A10B
- **Dataset:** /home/ma-user/work/models/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num prompts:** 1000
- **Request rate:** {'start': 1.0, 'end': 5.0, 'step': 1.0}
- **Timestamp:** 20260731_151143
- **Label:** baseline-util0.96-rr1-5-n1000

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.26 | 584646.96 | 1409259.81 | 2966.08 | 4684.32 | 2759.26 | 1000 | 0 |
| 2.0 | 0.33 | 572372.14 | 1425265.97 | 2418.28 | 4375.32 | 2212.55 | 1000 | 0 |
| 3.0 | 0.36 | 592645.32 | 1460839.13 | 2279.87 | 4587.55 | 2050.26 | 1000 | 0 |
| 4.0 | 0.37 | 580556.02 | 1428722.30 | 2115.27 | 4414.62 | 1933.85 | 1000 | 0 |
| 5.0 | 0.39 | 560899.37 | 1370861.31 | 1974.17 | 4134.06 | 1825.17 | 1000 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 1.0 | 2819.56 | 1935.47 | N/A | N/A | 54.01 | 729 |
| 2.0 | 2313.22 | 1527.92 | N/A | N/A | 67.74 | 817 |
| 3.0 | 2140.50 | 1818.43 | N/A | N/A | 73.03 | 871 |
| 4.0 | 2019.95 | 1801.53 | N/A | N/A | 76.44 | 882 |
| 5.0 | 1883.49 | 1478.85 | N/A | N/A | 79.24 | 888 |
