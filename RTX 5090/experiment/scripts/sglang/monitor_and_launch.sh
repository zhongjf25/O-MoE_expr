#!/bin/bash
# monitor_and_launch.sh
# 监控GPU空闲状态，启动SGLang服务，完成后运行benchmark

# === 配置 ===
MODEL_PATH=/root/autodl-tmp/models/qwen35_122b
TP_SIZE=8
CONTEXT_LENGTH=4096
BENCHMARK_SCRIPT=/root/autodl-tmp/workspace/experiment/scripts/run_sglang_benchmark.sh
START_SCRIPT=/root/autodl-tmp/workspace/experiment/scripts/start_sglang_server.sh

# GPU占用阈值 (MB) - 小于此值认为GPU空闲
GPU_USED_THRESHOLD=50

# 健康检查配置
MAX_WAIT=600
CHECK_INTERVAL=5

# === 步骤1: 轮询等待所有GPU空闲 ===
echo "[INFO] Waiting for all GPUs to be free (memory.used < ${GPU_USED_THRESHOLD}MB)..."

while true; do
    # 获取所有GPU的已用内存，找到最大值
    MAX_USED=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
               awk -F', ' '{print $2}' | sort -n | tail -1)

    # 获取GPU数量
    GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)

    # 检查是否所有GPU都满足条件（已用内存都小于阈值）
    ALL_FREE=true
    for USED in $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $2}'); do
        if [ "$USED" -gt "$GPU_USED_THRESHOLD" ]; then
            ALL_FREE=false
            break
        fi
    done

    if $ALL_FREE; then
        TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[INFO] All $GPU_COUNT GPUs are free (max used: ${MAX_USED}MB)"
        echo "[INFO] GPU free detected at: $TIMESTAMP"
        break
    fi

    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$TIMESTAMP] Max GPU used: ${MAX_USED}MB, next check in 10min..."
    sleep 600
done

# === 步骤2: 启动SGLang服务 ===
echo "[INFO] Starting SGLang server..."

export MODEL_PATH
export TP_SIZE
export CONTEXT_LENGTH

# 使用screen启动服务，方便追踪
screen -dm -S sglang_server -Logfile /tmp/sglang_server.log bash $START_SCRIPT

echo "[INFO] Server started in screen session 'sglang_server'"
echo "[INFO] Use 'screen -r sglang_server' to attach and view logs"

# === 步骤3: 等待服务就绪 ===
echo "[INFO] Waiting for server to be ready..."

BASE_URL="http://127.0.0.1:8000"
WAIT_COUNT=0

while true; do
    # 检查 /v1/models 端点
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/v1/models" --max-time 5 || echo "000")

    if [ "$HTTP_CODE" = "200" ]; then
        # 发送测试请求验证服务正常
        TEST_RESP=$(curl -s -X POST "$BASE_URL/generate" \
            -H "Content-Type: application/json" \
            -d '{"text": "Hello", "sampling_params": {"max_new_tokens": 4, "temperature": 0}}' \
            --max-time 30 2>&1 || echo "FAILED")

        if echo "$TEST_RESP" | grep -q "text\|generated_text\|outputs"; then
            echo "[INFO] Server is ready (warmup complete)"
            break
        fi
    fi

    WAIT_COUNT=$((WAIT_COUNT + CHECK_INTERVAL))
    if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
        echo "[ERROR] Timeout waiting for server after ${MAX_WAIT}s"
        echo "[INFO] Stopping sglang_server screen session..."
        screen -S sglang_server -X quit 2>/dev/null
        exit 1
    fi

    echo "  Waiting... ${WAIT_COUNT}s / ${MAX_WAIT}s"
    sleep $CHECK_INTERVAL
done

# === 步骤4: 运行Benchmark ===
echo "[INFO] Starting benchmark..."

export MODEL_PATH
export MODEL_NAME=qwen3.5-122b
export MAX_INPUT_LEN=4096
export RR_START=0.5
export RR_END=5
export RR_STEP=0.5
export NUM_PROMPTS=200
export EXP_LABEL=sglang_qwen35_122b

bash $BENCHMARK_SCRIPT

# === 完成 ===
echo "[INFO] Benchmark completed!"