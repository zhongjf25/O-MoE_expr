# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** fate
- **Model:** /data/share/models/Qwen3-30B-A3B
- **Dataset:** /root/workspace/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num Prompts:** 100
- **RR Range:** 10 ~ 10 (step 1)
- **Timestamp:** 20260414_044817
- **Label:** util92_cache50

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10.0 | 0.17 | 3704.22 | 4759.18 | 1108.07 | 1518.04 | 994.21 | 100 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 10.0 | 1117.41 | 1068.83 | 0.00 | 0.00 | 37.08 | 100 |