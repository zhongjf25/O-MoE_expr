#!/usr/bin/env bash

set -o pipefail

RESULTS_DIR="/home/ma-user/work/experiments/results/omoe-ascend/qwen35-122b/20260729_1541_offload100_rr1-5_n250"
STATUS_FILE="/home/ma-user/work/experiments/qwen35_offload100_rr1-5_n250.status"

printf 'running\n' >"$STATUS_FILE"

NUM_PROMPTS=250 \
RR_START=1 \
RR_END=5 \
RR_STEP=1 \
EXP_LABEL=offload100-rr1-5-n250 \
RESULTS_DIR="$RESULTS_DIR" \
bash /home/ma-user/work/O-MoE/simple_benchmark_qwen35_122_ascend.sh
rc=$?

printf '%s\n' "$rc" >"$STATUS_FILE"
exit "$rc"
