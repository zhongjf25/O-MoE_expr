# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** ktransformers
- **Model:** /root/autodl-tmp/models/Qwen3-30B-A3B
- **Dataset:** sharegpt
- **Random Input/Output:** 2048 / 512
- **Num Prompts:** 100
- **RR Range:** 0.2 ~ 2.6 (step 0.4)
- **Timestamp:** 20260412_114152
- **Label:** kt_OA_gpu_cache40

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | 0.20 | 333.85 | 1090.17 | 99.08 | 116.69 | 98.42 | 100 | None |
| 0.6 | 0.47 | 272.45 | 336.77 | 103.31 | 107.70 | 102.84 | 100 | None |
| 1.0 | 0.64 | 281.04 | 352.09 | 107.01 | 110.87 | 106.26 | 100 | None |
| 1.4 | 0.75 | 279.28 | 343.19 | 107.14 | 110.86 | 106.61 | 100 | None |
| 1.8 | 0.84 | 278.09 | 343.21 | 106.01 | 110.32 | 105.31 | 100 | None |
| 2.2 | 0.89 | 280.43 | 341.90 | 107.03 | 111.57 | 106.39 | 100 | None |
| 2.6 | 0.94 | 281.37 | 350.96 | 106.81 | 112.27 | 105.98 | 100 | None |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 0.2 | 98.15 | 96.87 | 0.00 | 0.00 | 47.67 | 10 |
| 0.6 | 103.46 | 102.28 | 0.00 | 0.00 | 113.04 | 22 |
| 1.0 | 106.98 | 106.05 | 0.00 | 0.00 | 152.68 | 31 |
| 1.4 | 107.55 | 106.43 | 0.00 | 0.00 | 179.48 | 37 |
| 1.8 | 105.74 | 104.96 | 0.00 | 0.00 | 199.47 | 45 |
| 2.2 | 106.93 | 105.94 | 0.00 | 0.00 | 212.83 | 53 |
| 2.6 | 106.42 | 105.80 | 0.00 | 0.00 | 224.04 | 56 |