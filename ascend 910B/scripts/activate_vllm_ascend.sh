#!/usr/bin/env bash

source /home/ma-user/anaconda3/etc/profile.d/conda.sh
conda activate /home/ma-user/work/envs/vllm-0.16.1rc0-py311

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /home/ma-user/work/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/bin/set_env.bash

export VLLM_PLUGINS=ascend
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export TRITON_ALL_BLOCKS_PARALLEL=1

export PYTHONPATH=/home/ma-user/work/experiments/scripts/runtime_vllm_ascend:/home/ma-user/work/vllm:/home/ma-user/work/vllm-ascend:/usr/local/Ascend/cann-8.5.2/python/site-packages:/usr/local/Ascend/cann-8.5.2/opp/built-in/op_impl/ai_core/tbe

export LD_LIBRARY_PATH=/home/ma-user/work/vllm-ascend/vllm_ascend:/home/ma-user/work/vllm-ascend/vllm_ascend/lib64:/home/ma-user/work/envs/vllm-0.16.1rc0-py311/lib/python3.11/site-packages/torch/lib:/home/ma-user/work/envs/vllm-0.16.1rc0-py311/lib/python3.11/site-packages/torch_npu/lib:${LD_LIBRARY_PATH:-}

unset OMOE_ASCEND_EARLY_PATCH
unset OMOE_QWEN35_TUPLE_SHARD_COMPAT
unset OMOE_QWEN35_ASCEND_BF16_SSM
unset VLLM_TARGET_DEVICE
unset TORCH_DEVICE_BACKEND_AUTOLOAD
