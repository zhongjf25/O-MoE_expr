# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** omoe
- **Model:** /root/autodl-tmp/models/Qwen3-30B-A3B
- **Dataset:** /root/autodl-tmp/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num Prompts:** 1000
- **RR Range:** 5.0 ~ 5.0 (step 1)
- **Timestamp:** 
- **Label:** omoe-qwen3_limit100

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.77 | 1856.95 | 6258.54 | 735.71 | 842.10 | 733.35 | 1000 | 0 |
| 2.0 | 0.99 | 46771.89 | 130398.85 | 837.31 | 1045.54 | 831.39 | 1000 | 0 |
| 3.0 | 1.00 | 106713.89 | 274045.43 | 847.88 | 1056.90 | 841.30 | 1000 | 0 |
| 4.0 | 1.00 | 143556.53 | 352315.31 | 849.40 | 1067.58 | 845.40 | 1000 | 0 |
| 5.0 | 1.00 | 168407.81 | 402395.79 | 856.28 | 1101.29 | 850.13 | 1000 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 1.0 | 765.50 | 753.61 | 0.00 | 0.00 | 155.18 | 182 |
| 2.0 | 863.91 | 859.74 | 0.00 | 0.00 | 199.49 | 471 |
| 3.0 | 875.10 | 870.67 | 0.00 | 0.00 | 202.68 | 664 |
| 4.0 | 872.25 | 866.54 | 0.00 | 0.00 | 201.42 | 744 |
| 5.0 | 875.91 | 870.25 | 0.00 | 0.00 | 201.16 | 785 |