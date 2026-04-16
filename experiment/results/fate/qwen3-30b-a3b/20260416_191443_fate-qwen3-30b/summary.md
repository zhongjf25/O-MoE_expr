# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** fate
- **Model:** /root/autodl-tmp/models/Qwen3-30B-A3B
- **Dataset:** /root/autodl-tmp/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num Prompts:** 250
- **RR Range:** 0.2 ~ 0.6 (step 0.4)
- **Timestamp:** 20260416_191443
- **Label:** fate-qwen3-30b

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | 0.18 | 1133.41 | 1676.35 | 347.71 | 476.99 | 343.16 | 250 | 0 |
| 0.6 | 0.34 | 15441.72 | 70186.11 | 663.17 | 1912.24 | 616.65 | 250 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 0.2 | 372.77 | 357.41 | 0.00 | 0.00 | 40.00 | 26 |
| 0.6 | 653.02 | 645.75 | 0.00 | 0.00 | 74.84 | 107 |