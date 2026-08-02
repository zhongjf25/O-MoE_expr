# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** vllm-ascend
- **Model:** /home/ma-user/work/models/minimax-m2.7
- **Served model:** MiniMax-M2.7
- **Dataset:** /home/ma-user/work/models/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num prompts:** 100
- **Request rate:** {'start': 1.0, 'end': 5.0, 'step': 1.0}
- **Timestamp:** 20260801_152357
- **Label:** baseline-minimax-m2.7-util0.985-rr1-5-n100

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.17 | 121727.71 | 320849.18 | 311.51 | 629.65 | 289.07 | 100 | 0 |
| 2.0 | 0.17 | 145509.17 | 317687.03 | 363.67 | 3490.22 | 299.79 | 100 | 0 |
| 3.0 | 0.17 | 137900.14 | 383330.27 | 321.74 | 1710.12 | 295.50 | 100 | 0 |
| 4.0 | 0.17 | 152322.99 | 388874.38 | 296.34 | 623.34 | 287.31 | 100 | 0 |
| 5.0 | 0.17 | 147518.86 | 302314.03 | 326.96 | 819.64 | 292.34 | 100 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 1.0 | 273.04 | 269.16 | N/A | N/A | 37.20 | 78 |
| 2.0 | 278.33 | 280.09 | N/A | N/A | 36.45 | 89 |
| 3.0 | 274.91 | 269.79 | N/A | N/A | 36.39 | 91 |
| 4.0 | 269.91 | 259.92 | N/A | N/A | 37.61 | 94 |
| 5.0 | 271.77 | 263.38 | N/A | N/A | 36.70 | 95 |
