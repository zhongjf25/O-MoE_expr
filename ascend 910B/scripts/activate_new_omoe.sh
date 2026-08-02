#!/usr/bin/env bash

BASE="${BASE:-/home/ma-user/work/new_omoe}"
OMOE_VLLM="${BASE}/O-MoE"
OMOE_ASCEND="${BASE}/O-MoE_Ascend"
VENV="${VENV:-/home/ma-user/work/envs/vllm-0.16.1rc0-py311}"

set +eu
source /usr/local/Ascend/cann-8.5.2/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
if [[ -f "${OMOE_ASCEND}/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/bin/set_env.bash" ]]; then
  source "${OMOE_ASCEND}/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/bin/set_env.bash"
fi
set -eu

export PATH="${VENV}/bin:${PATH}"
export PYTHONPATH="${OMOE_ASCEND}:${OMOE_VLLM}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${OMOE_ASCEND}/csrc/build:${OMOE_ASCEND}/vllm_ascend/lib64:${LD_LIBRARY_PATH:-}"
export DS_EXPERT_OFFLOAD=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=1
export OMOE_W8A8_COMPUTE="${OMOE_W8A8_COMPUTE:-ascendc_discrete}"

unset VLLM_VERSION || true
unset RANK_TABLE_FILE RANKTABLEFILE RANK_ID RANK_SIZE || true
unset ASCEND_RT_VISIBLE_DEVICES ASCEND_VISIBLE_DEVICES || true
