#!/bin/bash
#
# O-MoE 单OFFLOAD_EXPERT_LIMIT多RR值Benchmark脚本
# limit固定，遍历不同RR值，所有结果合并到第一个RR的目录中
#

set -e

# === 配置 ===
MODEL_PATH=/root/autodl-tmp/models/Qwen3-30B-A3B
TP_SIZE=2
PORT=8000
SERVING_SCRIPT=/root/autodl-tmp/workspace/experiment/scripts/O-MoE/simple_serving_qwen3.sh
BENCHMARK_SCRIPT=/root/autodl-tmp/workspace/experiment/scripts/O-MoE/simple_benchmark_qwen3.sh

# GPU空闲阈值 (MB)
GPU_USED_THRESHOLD=50

# 要监控的GPU索引，不设置则监控所有GPU（支持逗号分隔，如 "0" 或 "0,1,2"）
MONITOR_GPU=${MONITOR_GPU:-}

# 服务就绪检查配置
MAX_WAIT=300
CHECK_INTERVAL=15

# OFFLOAD_EXPERT_LIMIT 固定值
OFFLOAD_LIMIT=20

# RR范围
RR_START=5
RR_END=15
RR_STEP=1

# === 辅助函数 ===

# 获取要监控的GPU列表（支持MONITOR_GPU环境变量，如"0"或"0,1,2"或"0 1 2"）
get_monitor_gpu_indices() {
    if [ -n "$MONITOR_GPU" ]; then
        echo "$MONITOR_GPU" | tr ', ' '\n\n'
    else
        nvidia-smi --query-gpu=index --format=csv,noheader,nounits
    fi
}

check_gpu_free() {
    local ALL_FREE=true
    for IDX in $(get_monitor_gpu_indices); do
        USED=$(nvidia-smi --query-gpu=memory.used --id=$IDX --format=csv,noheader,nounits)
        USED=$(echo "$USED" | grep -oE '[0-9]+')
        if [ -z "$USED" ] || [ "$USED" -gt "$GPU_USED_THRESHOLD" ]; then
            ALL_FREE=false
            break
        fi
    done
    $ALL_FREE
}

wait_gpu_free() {
    local GPUS=$(get_monitor_gpu_indices | tr '\n' ' ')
    echo "[INFO] Waiting for GPUs [$GPUS] to be free (memory.used < ${GPU_USED_THRESHOLD}MB)..."
    while true; do
        if check_gpu_free; then
            MAX_USED=0
            for IDX in $(get_monitor_gpu_indices); do
                USED=$(nvidia-smi --query-gpu=memory.used --id=$IDX --format=csv,noheader,nounits)
                USED=$(echo "$USED" | grep -oE '[0-9]+')
                [ -n "$USED" ] && [ "$USED" -gt "$MAX_USED" ] && MAX_USED=$USED
            done
            TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
            echo "[INFO] GPUs free at $TIMESTAMP (max used: ${MAX_USED}MB)"
            return 0
        fi
        MAX_USED=0
        for IDX in $(get_monitor_gpu_indices); do
            USED=$(nvidia-smi --query-gpu=memory.used --id=$IDX --format=csv,noheader,nounits)
            USED=$(echo "$USED" | grep -oE '[0-9]+')
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
            # 发送测试请求验证服务正常
            TEST_RESP=$(curl -s -X POST "$BASE_URL/v1/completions" \
                -H "Content-Type: application/json" \
                -d '{"model": "/root/autodl-tmp/models/Qwen3-30B-A3B", "prompt": "Hello", "max_tokens": 4, "temperature": 0}' \
                --max-time 30 2>&1 || echo "FAILED")

            if echo "$TEST_RESP" | grep -q "choices\|text\|content"; then
                echo "[INFO] Service is ready"
                return 0
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
    local WAIT_COUNT=0
    local MAX_RELEASE_WAIT=120
    local GPUS=$(get_monitor_gpu_indices | tr '\n' ' ')
    echo "[INFO] Waiting for GPU memory to be released..."

    while true; do
        if check_gpu_free; then
            MAX_USED=0
            for IDX in $(get_monitor_gpu_indices); do
                USED=$(nvidia-smi --query-gpu=memory.used --id=$IDX --format=csv,noheader,nounits)
                USED=$(echo "$USED" | grep -oE '[0-9]+')
                [ -n "$USED" ] && [ "$USED" -gt "$MAX_USED" ] && MAX_USED=$USED
            done
            TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
            echo "[INFO] GPU memory released at $TIMESTAMP (max used: ${MAX_USED}MB)"
            return 0
        fi

        WAIT_COUNT=$((WAIT_COUNT + 10))
        if [ $WAIT_COUNT -ge $MAX_RELEASE_WAIT ]; then
            MAX_USED=0
            for IDX in $(get_monitor_gpu_indices); do
                USED=$(nvidia-smi --query-gpu=memory.used --id=$IDX --format=csv,noheader,nounits)
                USED=$(echo "$USED" | grep -oE '[0-9]+')
                [ -n "$USED" ] && [ "$USED" -gt "$MAX_USED" ] && MAX_USED=$USED
            done
            echo "[WARN] GPU memory not fully released after ${MAX_RELEASE_WAIT}s (max used: ${MAX_USED}MB)"
            echo "[WARN] Proceeding anyway..."
            return 0
        fi

        MAX_USED=0
        for IDX in $(get_monitor_gpu_indices); do
            USED=$(nvidia-smi --query-gpu=memory.used --id=$IDX --format=csv,noheader,nounits)
            USED=$(echo "$USED" | grep -oE '[0-9]+')
            [ -n "$USED" ] && [ "$USED" -gt "$MAX_USED" ] && MAX_USED=$USED
        done
        echo "  GPU not free yet (max used: ${MAX_USED}MB), waiting... ${WAIT_COUNT}s / ${MAX_RELEASE_WAIT}s"
        sleep 10
    done
}

# === 主流程 ===

LOG_DIR="/root/autodl-tmp/workspace/experiment/results/omoe/qwen3-30b"
mkdir -p "$LOG_DIR"
SCRIPT_LOG="$LOG_DIR/benchmark_single_limit_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$SCRIPT_LOG") 2>&1

echo "=== run_benchmark_single_limit.sh started at $(date) ==="
echo ""
echo "=== Configuration ==="
echo "MODEL_PATH:       $MODEL_PATH"
echo "TP_SIZE:          $TP_SIZE"
echo "PORT:             $PORT"
echo "GPU_USED_THRESHOLD: $GPU_USED_THRESHOLD MB"
echo "MONITOR_GPU:     ${MONITOR_GPU:-all GPUs}"
echo "OFFLOAD_LIMIT:    $OFFLOAD_LIMIT"
echo "RR_START:         $RR_START"
echo "RR_END:           $RR_END"
echo "RR_STEP:          $RR_STEP"
echo "==========================="
echo ""

# 确保GPU空闲
wait_gpu_free

# 生成结果目录
EXP_LABEL="omoe-qwen3_limit${OFFLOAD_LIMIT}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${LOG_DIR}/${TIMESTAMP}_${EXP_LABEL}"
mkdir -p "$RESULTS_DIR"
echo "[INFO] Results will be saved to: $RESULTS_DIR"
echo ""

# 生成RR列表
generate_rr_list() {
    local start=$1
    local end=$2
    local step=$3
    awk -v start="$start" -v end="$end" -v step="$step" 'BEGIN {
        for (i = start; i <= end; i += step) {
            printf "%.1f\n", i
        }
    }'
}

RR_LIST=$(generate_rr_list $RR_START $RR_END $RR_STEP)
RR_COUNT=$(echo "$RR_LIST" | wc -l)
LAST_RR=$(echo "$RR_LIST" | tail -1)
echo "[INFO] Will run $RR_COUNT RR values: $RR_LIST"
echo "[INFO] Last RR value: $LAST_RR"
echo ""

# 启动服务一次
echo "[INFO] Starting service with OFFLOAD_EXPERT_LIMIT=$OFFLOAD_LIMIT..."

export MODEL_PATH
export TP_SIZE
export PORT
export OFFLOAD_EXPERT_LIMIT=$OFFLOAD_LIMIT
export GPU_MEMORY_UTIL=0.8

SERVER_LOG="$RESULTS_DIR/server.log"
touch "$SERVER_LOG"

screen -dm -S omoe_server bash -c "bash $SERVING_SCRIPT 2>&1 | tee -a '$SERVER_LOG'"

echo "[INFO] Service started in screen 'omoe_server'"
echo "[INFO] Log file: $SERVER_LOG"

if ! wait_service_ready; then
    echo "[ERROR] Service failed to start"
    screen -S omoe_server -X quit 2>/dev/null || true
    stop_service
    exit 1
fi

echo "[INFO] Service is ready, starting benchmarks..."
echo ""

# 遍历每个RR值进行benchmark
first=true
for RR in $RR_LIST; do
    echo "============================================"
    echo "  Running benchmark with RR=$RR (OFFLOAD_LIMIT=$OFFLOAD_LIMIT)"
    echo "============================================"

    export MODEL_PATH
    export EXP_LABEL="omoe-qwen3_limit${OFFLOAD_LIMIT}"
    export RR_START=$RR
    export RR_END=$RR
    export RR_STEP=1
    export NUM_PROMPTS=1000
    export BASE_URL="http://localhost:${PORT}"
    export RESULTS_DIR

    bash $BENCHMARK_SCRIPT
    BENCHMARK_EXIT=$?

    if [ $BENCHMARK_EXIT -ne 0 ]; then
        echo "[WARN] Benchmark failed for RR=$RR"
    else
        echo "[INFO] Benchmark completed for RR=$RR"
    fi

    echo ""

    # 每个RR benchmark完成后重启服务（最后一个RR不需要重启）
    if [ "$RR" != "$LAST_RR" ]; then
        echo "[INFO] Restarting service for next RR..."
        screen -S omoe_server -X quit 2>/dev/null || true
        stop_service
        wait_gpu_release

        # 重新启动服务
        screen -dm -S omoe_server bash -c "bash $SERVING_SCRIPT 2>&1 | tee -a '$SERVER_LOG'"

        if ! wait_service_ready; then
            echo "[ERROR] Service failed to restart for RR=$RR"
            exit 1
        fi
    fi
done

# 完成所有benchmark后停止服务
echo "[INFO] All benchmarks completed, stopping service..."
screen -S omoe_server -X quit 2>/dev/null || true
stop_service
wait_gpu_release

echo ""
echo "============================================"
echo "  All benchmarks completed!"
echo "  Results saved to: $RESULTS_DIR"
echo "============================================"