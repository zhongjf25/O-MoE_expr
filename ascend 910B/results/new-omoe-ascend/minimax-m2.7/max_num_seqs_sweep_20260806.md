# MiniMax-M2.7 max-num-seqs sweep

These runs use the full MiniMax-M2.7 model on 8 Ascend SNT9B1 devices.

## Common configuration

| Setting | Value |
| --- | ---: |
| Framework | new O-MoE Ascend |
| Tensor parallel size | 8 |
| GPU memory utilization | 0.94 |
| Offload expert limit | 30 |
| Cached experts per routed layer | 226 |
| Maximum model length | 4096 |
| Request rate | 15 req/s |
| Prompts | 2000 |
| Temperature | 0 |

`cached_num_experts=226` follows `256 - offload_expert_limit` for the 256 experts in each routed layer.

## Results

| max-num-seqs | Completed | Failed | Request throughput (req/s) | Output throughput (tok/s) | Total throughput (tok/s) | Mean TPOT (ms) | Mean TTFT (ms) | Duration (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 55 | 2000 | 0 | 0.543950 | 109.9541 | 230.0798 | 475.294 | 1,550,235.93 | 3,676.81 |
| 50 | 2000 | 0 | 0.504032 | 101.8850 | 213.1951 | 467.814 | 1,700,250.63 | 3,968.00 |
| 45 | 2000 | 0 | 0.450367 | 91.0371 | 190.4958 | 474.902 | 1,930,597.18 | 4,440.83 |

The benchmark client did not cap request concurrency. Requests above the server-side `max-num-seqs` limit remained queued, so TTFT includes substantial queueing time.

## Result directories

- `20260806_211746_offload30-cached226-maxseq55-util0.94-rr15-n2000`
- `20260806_223336_offload30-cached226-maxseq50-util0.94-rr15-n2000`
- `20260806_235431_offload30-cached226-maxseq45-util0.94-rr15-n2000`

Each directory contains benchmark configuration, model metadata, generated summaries, and the detailed `rr_15/result.json`. Runtime logs are intentionally excluded.
