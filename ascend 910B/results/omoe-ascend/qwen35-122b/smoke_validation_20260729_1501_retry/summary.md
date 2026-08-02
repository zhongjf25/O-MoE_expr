# Serving Benchmark Result Summary

## Experiment Config
- **Framework:** omoe-ascend
- **Model:** /home/ma-user/work/models/Qwen3.5-122B-A10B
- **Served model:** INFO 07-29 15:04:39 [__init__.py:44] Available plugins for group vllm.platform_plugins:
INFO 07-29 15:04:39 [__init__.py:46] - ascend -> vllm_ascend:register
INFO 07-29 15:04:39 [__init__.py:58] Loading plugin ascend
INFO 07-29 15:04:39 [__init__.py:212] Platform plugin ascend is activated
Qwen3.5-122B-A10B
- **Dataset:** /home/ma-user/work/models/ShareGPT_V3_unfiltered_cleaned_split.json
- **Num prompts:** 1
- **Request rate:** {'start': 1.0, 'end': 1.0, 'step': 1.0}
- **Timestamp:** 20260729_150410
- **Label:** smoke

## Benchmark Results

| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 1 |

## Detailed Metrics

| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.00 | 0.00 | N/A | N/A | 0.00 | 0 |
