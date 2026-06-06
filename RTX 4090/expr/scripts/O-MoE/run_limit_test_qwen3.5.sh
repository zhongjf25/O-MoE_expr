#!/bin/bash
#
# O-MoE 多OFFLOAD_EXPERT_LIMIT循环Benchmark脚本 (Qwen3.5-35B)
#

set -e

# === 配置 ===
MODEL_PATH=/data/share/models/qwen35_35b
TP_SIZE=2
PORT=8000
SERVING_SCRIPT=/root/workspace/expr/scripts/O-MoE/simple_serving_qwen3.5.sh
BENCHMARK_SCRIPT=/root/workspace/expr/scripts/O-MoE/simple_benchmark_qwen3.5.sh

# GPU空闲阈值 (MB)
GPU_USED_THRESHOLD=50

# 服务就绪检查配置
MAX_WAIT=800
CHECK_INTERVAL=15

# OFFLOAD_EXPERT_LIMIT 列表 
# OFFLOAD_LIMITS="30 50 70 90 110 130 150 170 190 210 230"
# OFFLOAD_LIMITS="80 100 120 140 160 180 200 220 240"
OFFLOAD_LIMITS="120 130 140 150 160 170 180 190 200 210 220 230 240"

# === 辅助函数 ===

check_gpu_free() {
    local ALL_FREE=true
    for IDX in 1 2; do
        USED=$(nvidia-smi --query-gpu=memory.used --id=$IDX --format=csv,noheader,nounits | grep -oE '[0-9]+')
        if [ -n "$USED" ] && [ "$USED" -gt "$GPU_USED_THRESHOLD" ]; then
            ALL_FREE=false
            break
        fi
    done
    $ALL_FREE
}

wait_gpu_free() {
    echo "[INFO] Waiting for GPUs 1,2 to be free (memory.used < ${GPU_USED_THRESHOLD}MB)..."
    while true; do
        if check_gpu_free; then
            MAX_USED=0
            for IDX in 1 2; do
                USED=$(nvidia-smi --query-gpu=memory.used --id=$IDX --format=csv,noheader,nounits | grep -oE '[0-9]+')
                [ -n "$USED" ] && [ "$USED" -gt "$MAX_USED" ] && MAX_USED=$USED
            done
            TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
            echo "[INFO] GPUs 1,2 free at $TIMESTAMP (max used: ${MAX_USED}MB)"
            return 0
        fi
        MAX_USED=0
        for IDX in 1 2; do
            USED=$(nvidia-smi --query-gpu=memory.used --id=$IDX --format=csv,noheader,nounits | grep -oE '[0-9]+')
            [ -n "$USED" ] && [ "$USED" -gt "$MAX_USED" ] && MAX_USED=$USED
        done
        TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[$TIMESTAMP] Max GPU used: ${MAX_USED}MB, next check in 10min..."
        sleep 600
    done
}

wait_service_ready() {
    local BASE_URL="http://127.0.0.1:${PORT}"
    local WAIT_COUNT=0

    echo "[INFO] Waiting for service to be ready..."

    while true; do
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/v1/models" --max-time 5 || echo "000")

        if [ "$HTTP_CODE" = "200" ]; then
            # 获取实际注册的模型名称
            MODEL_ID=$(curl -s "$BASE_URL/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "")

            if [ -n "$MODEL_ID" ]; then
                # 使用实际模型名称发送测试请求
                TEST_RESP=$(curl -s -X POST "$BASE_URL/v1/completions" \
                    -H "Content-Type: application/json" \
                    -d "{\"model\": \"$MODEL_ID\", \"prompt\": \"Hello\", \"max_tokens\": 4, \"temperature\": 0}" \
                    --max-time 30 2>&1 || echo "FAILED")

                if echo "$TEST_RESP" | grep -q "choices"; then
                    echo "[INFO] Service is ready (model: $MODEL_ID)"
                    return 0
                fi
            fi
        fi

        WAIT_COUNT=$((WAIT_COUNT + CHECK_INTERVAL))
        if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
            echo "[ERROR] Timeout waiting for service after ${MAX_WAIT}s"
            return 1
        fi

        echo "  Waiting... ${WAIT_COUNT}s / ${MAX_WAIT}s"
        sleep $CHECK_INTERVAL
    done
}

stop_service() {
    echo "[INFO] Stopping vllm service..."
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 5
    if pgrep -f "vllm serve" > /dev/null; then
        echo "[WARN] Force killing vllm..."
        pkill -9 -f "vllm serve" 2>/dev/null || true
        sleep 3
    fi
    echo "[INFO] Service stopped"
}

wait_gpu_release() {
    # 停止服务后等待GPU显存释放
    local WAIT_COUNT=0
    local MAX_RELEASE_WAIT=120
    echo "[INFO] Waiting for GPU memory to be released..."

    while true; do
        if check_gpu_free; then
            MAX_USED=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $2}' | sort -n | tail -1)
            TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
            echo "[INFO] GPU memory released at $TIMESTAMP (max used: ${MAX_USED}MB)"
            return 0
        fi

        WAIT_COUNT=$((WAIT_COUNT + 10))
        if [ $WAIT_COUNT -ge $MAX_RELEASE_WAIT ]; then
            MAX_USED=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $2}' | sort -n | tail -1)
            echo "[WARN] GPU memory not fully released after ${MAX_RELEASE_WAIT}s (max used: ${MAX_USED}MB)"
            echo "[WARN] Proceeding anyway..."
            return 0
        fi

        MAX_USED=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $2}' | sort -n | tail -1)
        echo "  GPU not free yet (max used: ${MAX_USED}MB), waiting... ${WAIT_COUNT}s / ${MAX_RELEASE_WAIT}s"
        sleep 10
    done
}

# === 主流程 ===

# 记录脚本配置到日志
LOG_DIR="/root/autodl-tmp/workspace/experiment/results/omoe/qwen3.5-35b/logs"
mkdir -p "$LOG_DIR"
SCRIPT_LOG="$LOG_DIR/benchmark_loop_$(date +%Y%m%d_%H%M%S).log"

# 重定向所有输出到日志文件
exec > >(tee -a "$SCRIPT_LOG") 2>&1

echo "=== run_benchmark_loop_qwen3.5.sh started at $(date) ==="
echo ""
echo "=== Script Content ==="
cat "$0"
echo ""
echo "=== Configuration ==="
echo "MODEL_PATH: $MODEL_PATH"
echo "TP_SIZE: $TP_SIZE"
echo "PORT: $PORT"
echo "GPU_USED_THRESHOLD: $GPU_USED_THRESHOLD MB"
echo "OFFLOAD_LIMITS: $OFFLOAD_LIMITS"
echo "SERVING_SCRIPT: $SERVING_SCRIPT"
echo "BENCHMARK_SCRIPT: $BENCHMARK_SCRIPT"
echo "==========================="
echo ""

wait_gpu_free

for LIMIT in $OFFLOAD_LIMITS; do
    echo ""
    echo "============================================"
    echo "  Starting benchmark with OFFLOAD_EXPERT_LIMIT=$LIMIT"
    echo "============================================"

    if ! check_gpu_free; then
        echo "[WARN] GPU not free before starting, waiting..."
        wait_gpu_free
    fi

    # 构建与benchmark脚本相同的结果目录路径
    EXP_LABEL="omoe-qwen3.5_limit${LIMIT}"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RESULTS_DIR="/root/workspace/expr/results/omoe/qwen3.5-35b/${TIMESTAMP}_${EXP_LABEL}"
    mkdir -p "$RESULTS_DIR"
    SERVER_LOG="$RESULTS_DIR/server.log"

    echo "[INFO] Starting service with OFFLOAD_EXPERT_LIMIT=$LIMIT..."

    export MODEL_PATH
    export TP_SIZE
    export PORT
    export OFFLOAD_EXPERT_LIMIT=$LIMIT
    export GPU_MEMORY_UTIL=0.95

    # 启动服务，输出同时到screen和日志文件
    screen -dm -S omoe_server_${LIMIT} bash -c "bash $SERVING_SCRIPT 2>&1 | tee -a '$SERVER_LOG'"

    echo "[INFO] Service started in screen 'omoe_server_${LIMIT}'"
    echo "[INFO] Log file: $SERVER_LOG"

    if ! wait_service_ready; then
        echo "[ERROR] Service failed to start with OFFLOAD_EXPERT_LIMIT=$LIMIT"
        screen -S omoe_server_${LIMIT} -X quit 2>/dev/null || true
        stop_service
        continue
    fi

    echo "[INFO] Running benchmark..."

    export MODEL_PATH
    export EXP_LABEL="omoe-qwen3.5_limit${LIMIT}"
    export RR_START=10
    export RR_END=10
    export RR_STEP=1
    export NUM_PROMPTS=1000
    export BASE_URL="http://localhost:${PORT}"

    bash $BENCHMARK_SCRIPT
    BENCHMARK_EXIT=$?

    echo "[INFO] Benchmark completed (exit code: $BENCHMARK_EXIT)"
    screen -S omoe_server_${LIMIT} -X quit 2>/dev/null || true
    kill $SERVER_PID 2>/dev/null || true
    stop_service

    # 等待GPU显存释放（最多等待2分钟）
    wait_gpu_release

    if [ $BENCHMARK_EXIT -ne 0 ]; then
        echo "[WARN] Benchmark failed for OFFLOAD_EXPERT_LIMIT=$LIMIT"
    fi

done

echo ""
echo "============================================"
echo "  All benchmarks completed!"
echo "============================================"