# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** omoe
- **Model:** /data/share/models/Qwen3-30B-A3B
- **Dataset:** /root/workspace/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num Prompts:** 50
- **RR Range:** 5 ~ 6 (step 1)
- **Timestamp:** 20260411_041327
- **Label:** test_benchmark_30b

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5.0 | 0.25 | 1270.58 | 4382.33 | 437.22 | 622.16 | 349.36 | 50 | 0 |
| 6.0 | 0.27 | 3721.43 | 8113.65 | 398.85 | 665.11 | 320.76 | 50 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 5.0 | 418.86 | 326.40 | 0.00 | 0.00 | 51.51 | 48 |
| 6.0 | 385.39 | 309.74 | 0.00 | 0.00 | 55.13 | 49 |