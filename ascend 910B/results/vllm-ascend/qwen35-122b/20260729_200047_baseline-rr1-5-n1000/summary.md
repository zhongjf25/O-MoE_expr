# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** vllm-ascend
- **Model:** /home/ma-user/work/models/Qwen3.5-122B-A10B
- **Served model:** Qwen3.5-122B-A10B
- **Dataset:** /home/ma-user/work/models/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num prompts:** 1000
- **Request rate:** {'start': 1.0, 'end': 5.0, 'step': 1.0}
- **Timestamp:** 20260729_200047
- **Label:** baseline-rr1-5-n1000

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.07 | 6831501.38 | 13036320.59 | 1387.14 | 2000.03 | 1369.52 | 1000 | 0 |
| 2.0 | 0.07 | 6901075.05 | 13243601.05 | 1338.38 | 1863.84 | 1324.50 | 1000 | 0 |
| 3.0 | 0.07 | 6891313.11 | 13210923.01 | 1328.78 | 1780.03 | 1317.31 | 1000 | 0 |
| 4.0 | 0.07 | 7057640.35 | 13402672.34 | 1309.41 | 1549.10 | 1303.60 | 1000 | 0 |
| 5.0 | 0.07 | 6511286.37 | 12474675.92 | 1248.48 | 1819.17 | 1235.74 | 1000 | 0 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 1.0 | 1355.78 | 1275.73 | N/A | N/A | 13.51 | 934 |
| 2.0 | 1313.51 | 1277.27 | N/A | N/A | 13.96 | 968 |
| 3.0 | 1308.97 | 1280.07 | N/A | N/A | 14.11 | 977 |
| 4.0 | 1303.61 | 1271.87 | N/A | N/A | 14.24 | 982 |
| 5.0 | 1229.47 | 1200.60 | N/A | N/A | 15.00 | 985 |
