# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** new-omoe-ascend
- **Model:** /home/ma-user/work/models/minimax-m2.7
- **Served model:** MiniMax-M2.7
- **Dataset:** /home/ma-user/work/models/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num prompts:** 2000
- **Request rate:** {'start': 15.0, 'end': 15.0, 'step': 1.0}
- **Timestamp:** 20260803_012007
- **Label:** util-0.85-rr15-n2000

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 15.0 | 1.20 | 281364.80 | 661647.37 | 587.22 | 656.58 | 592.32 | 2000 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 15.0 | 599.26 | 592.35 | N/A | N/A | 241.88 | 1730 |
