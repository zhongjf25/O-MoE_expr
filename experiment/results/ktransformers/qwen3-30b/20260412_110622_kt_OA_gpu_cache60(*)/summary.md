# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** ktransformers
- **Model:** /root/autodl-tmp/models/Qwen3-30B-A3B
- **Dataset:** sharegpt
- **Random Input/Output:** 2048 / 512
- **Num Prompts:** 100
- **RR Range:** 0.2 ~ 2.6 (step 0.4)
- **Timestamp:** 20260412_110622
- **Label:** kt_OA_gpu_cache60

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | 0.20 | 306.64 | 973.04 | 96.49 | 109.22 | 95.90 | 100 | None |
| 0.6 | 0.48 | 4354.67 | 28818.57 | 102.96 | 114.23 | 102.37 | 100 | None |
| 1.0 | 0.48 | 21734.69 | 40451.02 | 108.22 | 121.09 | 104.18 | 100 | None |
| 1.4 | 0.54 | 30241.30 | 58336.83 | 111.19 | 297.35 | 104.07 | 100 | None |
| 1.8 | 0.52 | 38409.18 | 78397.51 | 109.70 | 153.99 | 107.32 | 100 | None |
| 2.2 | 0.54 | 40333.78 | 84000.30 | 107.35 | 151.16 | 104.87 | 100 | None |
| 2.6 | 0.53 | 43834.58 | 90850.06 | 108.98 | 152.07 | 106.09 | 100 | None |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 0.2 | 95.66 | 94.64 | 0.00 | 0.00 | 47.83 | 10 |
| 0.6 | 103.07 | 100.27 | 0.00 | 0.00 | 113.76 | 25 |
| 1.0 | 102.73 | 100.29 | 0.00 | 0.00 | 114.61 | 52 |
| 1.4 | 102.82 | 99.86 | 0.00 | 0.00 | 128.18 | 71 |
| 1.8 | 105.55 | 102.58 | 0.00 | 0.00 | 123.25 | 78 |
| 2.2 | 103.48 | 100.45 | 0.00 | 0.00 | 128.14 | 84 |
| 2.6 | 104.33 | 100.66 | 0.00 | 0.00 | 127.32 | 85 |