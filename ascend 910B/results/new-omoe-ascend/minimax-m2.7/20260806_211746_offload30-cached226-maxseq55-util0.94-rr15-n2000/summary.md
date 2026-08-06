# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** new-omoe-ascend
- **Model:** /home/ma-user/work/models/minimax-m2.7
- **Served model:** MiniMax-M2.7
- **Dataset:** /home/ma-user/work/models/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num prompts:** 2000
- **Request rate:** {'start': 15.0, 'end': 15.0, 'step': 1.0}
- **Timestamp:** 20260806_211746
- **Label:** offload30-cached226-maxseq55-util0.94-rr15-n2000

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 15.0 | 0.54 | 1550235.93 | 3167553.25 | 475.29 | 625.39 | 470.61 | 2000 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 15.0 | 467.98 | 442.18 | N/A | N/A | 109.95 | 1920 |
