# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** ktransformers
- **Model:** /root/autodl-tmp/models/Qwen3-30B-A3B
- **Dataset:** sharegpt
- **Random Input/Output:** 2048 / 512
- **Num Prompts:** 100
- **RR Range:** 1.8 ~ 1.8 (step 0.4)
- **Timestamp:** 20260412_132456
- **Label:** kt_OA_gpu_cache60

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.8 | 0.38 | 40646.01 | 84429.34 | 129.55 | 236.39 | 127.89 | 100 | None |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 1.8 | 105.24 | 102.56 | 0.00 | 0.00 | 90.00 | 76 |