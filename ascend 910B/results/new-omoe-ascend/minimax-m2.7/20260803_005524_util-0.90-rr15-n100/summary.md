# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** new-omoe-ascend
- **Model:** /home/ma-user/work/models/minimax-m2.7
- **Served model:** MiniMax-M2.7
- **Dataset:** /home/ma-user/work/models/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num prompts:** 100
- **Request rate:** {'start': 15.0, 'end': 15.0, 'step': 1.0}
- **Timestamp:** 20260803_005524
- **Label:** util-0.90-rr15-n100

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 15.0 | 0.27 | 948.75 | 1474.62 | 488.14 | 629.43 | 472.56 | 100 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 15.0 | 473.25 | 468.17 | N/A | N/A | 58.03 | 98 |
