#!/bin/bash
#
# O-MoE 多OFFLOAD_EXPERT_LIMIT循环Benchmark脚本 (Qwen3.5-35B)
#

set -e

# === 配置 ===
MODEL_PATH=/root/autodl-tmp/models/Qwen3.5-35B-A3B
MAX_MODEL_LEN=8192
TP_SIZE=1
PORT=8028
SERVING_SCRIPT=/root/autodl-tmp/workspace/experiment/scripts/O-MoE/simple_serving_qwen35_35.sh
BENCHMARK_SCRIPT=/root/autodl-tmp/workspace/experiment/scripts/O-MoE/simple_benchmark_qwen35_35.sh

# GPU空闲阈值 (MB)
GPU_USED_THRESHOLD=50

# 要监控的GPU索引，不设置则监控所有GPU（支持逗号分隔，如 "6,7" 或 "0"）
MONITOR_GPU=${MONITOR_GPU:-0,1}

# 服务就绪检查配置
MAX_WAIT=300
CHECK_INTERVAL=15

# OFFLOAD_EXPERT_LIMIT 列表
# OFFLOAD_LIMITS="200 210 220 230"
OFFLOAD_LIMITS="220"

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
            TEST_RESP=$(curl -s -X POST "$BASE_URL/v1/completions" \
                -H "Content-Type: application/json" \
                -d '{"model": "/root/autodl-tmp/models/Qwen3.5-35B-A3B", "prompt": "Hello", "max_tokens": 4, "temperature": 0}' \
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

LOG_DIR="/root/autodl-tmp/workspace/experiment/results/omoe/qwen35-35b"
mkdir -p "$LOG_DIR"
SCRIPT_LOG="$LOG_DIR/benchmark_loop_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$SCRIPT_LOG") 2>&1

echo "=== run_benchmark_loop_qwen35_35b.sh started at $(date) ==="
echo ""
echo "=== Configuration ==="
echo "MODEL_PATH:       $MODEL_PATH"
echo "TP_SIZE:          $TP_SIZE"
echo "PORT:             $PORT"
echo "GPU_USED_THRESHOLD: $GPU_USED_THRESHOLD MB"
echo "MONITOR_GPU:     ${MONITOR_GPU:-all GPUs}"
echo "OFFLOAD_LIMITS:   $OFFLOAD_LIMITS"
echo "SERVING_SCRIPT:   $SERVING_SCRIPT"
echo "BENCHMARK_SCRIPT: $BENCHMARK_SCRIPT"
echo "==========================="
echo ""

# 确保GPU空闲
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

    # 生成结果目录
    EXP_LABEL="omoe-qwen35-35b_limit${LIMIT}"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RESULTS_DIR="${LOG_DIR}/${TIMESTAMP}_${EXP_LABEL}"
    mkdir -p "$RESULTS_DIR"
    SERVER_LOG="$RESULTS_DIR/server.log"

    echo "[INFO] Starting service with OFFLOAD_EXPERT_LIMIT=$LIMIT..."
    echo "[INFO] Results will be saved to: $RESULTS_DIR"

    export MODEL_PATH
    export MAX_MODEL_LEN
    export TP_SIZE
    export PORT
    export OFFLOAD_EXPERT_LIMIT=$LIMIT
    export GPU_MEMORY_UTIL=0.86

    screen -dm -S omoe_35b_${LIMIT} bash -c "bash $SERVING_SCRIPT 2>&1 | tee -a '$SERVER_LOG'"

    echo "[INFO] Service started in screen 'omoe_35b_${LIMIT}'"
    echo "[INFO] Log file: $SERVER_LOG"

    if ! wait_service_ready; then
        echo "[ERROR] Service failed to start with OFFLOAD_EXPERT_LIMIT=$LIMIT"
        screen -S omoe_35b_${LIMIT} -X quit 2>/dev/null || true
        stop_service
        continue
    fi

    echo "[INFO] Running benchmark..."

    export MODEL_PATH
    export EXP_LABEL="omoe-qwen35-35b_limit${LIMIT}"
    export RR_START=6
    export RR_END=6
    export RR_STEP=1
    export NUM_PROMPTS=1000
    export BASE_URL="http://localhost:${PORT}"
    export RESULTS_DIR

    bash $BENCHMARK_SCRIPT
    BENCHMARK_EXIT=$?

    echo "[INFO] Benchmark completed (exit code: $BENCHMARK_EXIT)"

    screen -S omoe_35b_${LIMIT} -X quit 2>/dev/null || true
    stop_service

    # 等待GPU显存释放
    wait_gpu_release

    if [ $BENCHMARK_EXIT -ne 0 ]; then
        echo "[WARN] Benchmark failed for OFFLOAD_EXPERT_LIMIT=$LIMIT"
    fi

done

echo ""
echo "============================================"
echo "  All benchmarks completed!"
echo "============================================"