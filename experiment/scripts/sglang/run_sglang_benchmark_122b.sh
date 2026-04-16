#!/bin/bash
#
# SGLang Benchmark Runner
# Supports configurable request rate range and step size, auto-generates summary
#
# Usage: bash run_sglang_benchmark.sh
# Environment variables to override:
#   MODEL_PATH      - model path (default: /root/autodl-tmp/models/Qwen3-30B-A3B)
#   BASE_URL        - server URL (default: http://localhost:8000)
#   RR_START        - request rate start (default: 1)
#   RR_END          - request rate end (default: 50)
#   RR_STEP         - request rate step (default: 5)
#   NUM_PROMPTS     - number of test prompts (default: 1000)
#   EXP_LABEL       - experiment label (default: sglang)
#   RANDOM_INPUT    - random input length in tokens (default: 2048)
#   RANDOM_OUTPUT   - random output length in tokens (default: 512)
#

set -e


# qwen3.5-122b benchmark配置
# MODEL_PATH=/root/autodl-tmp/models/qwen35_122b MODEL_NAME=qwen3.5-122b MAX_INPUT_LEN=4096 RR_START=1.5 RR_END=5 RR_STEP=0.5 NUM_PROMPTS=500 EXP_LABEL=sglang_qwen35_122b bash /root/autodl-tmp/workspace/experiment/scripts/run_sglang_benchmark.sh

# ========== 参数配置 ==========
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/qwen35_122b}"
BASE_URL="${BASE_URL:-http://localhost:8000}"
DATASET_PATH="/root/autodl-tmp/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"
NUM_PROMPTS="${NUM_PROMPTS:-100}"

RR_START="${RR_START:-0.1}"
RR_END="${RR_END:-0.3}"
RR_STEP="${RR_STEP:-0.1}"
EXP_LABEL="${EXP_LABEL:-sglang}"

# SGLang specific
RANDOM_INPUT="${RANDOM_INPUT:-2048}"
RANDOM_OUTPUT="${RANDOM_OUTPUT:-512}"
DATASET_NAME="${DATASET_NAME:-sharegpt}"  # 可选: random, sharegpt

# Prompt 长度筛选 (仅对 sharegpt 数据集生效)
MIN_INPUT_LEN="${MIN_INPUT_LEN:-0}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-0}"     # 0 表示不限制

# 框架和模型标识
FRAMEWORK="${FRAMEWORK:-sglang}"
MODEL_NAME="${MODEL_NAME:-qwen3-30b}"

# ========== 目录初始化 ==========
if [ -z "${RESULTS_DIR:-}" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RESULTS_DIR="/root/autodl-tmp/workspace/experiment/results/${FRAMEWORK}/${MODEL_NAME}/${TIMESTAMP}_${EXP_LABEL}"
fi

echo "============================================"
echo "  SGLang Benchmark Experiment Config"
echo "============================================"
echo "  Model:        $MODEL_PATH"
echo "  Base URL:     $BASE_URL"
echo "  Dataset:      $DATASET_NAME (input=$RANDOM_INPUT, output=$RANDOM_OUTPUT)"
echo "  Num Prompts:  $NUM_PROMPTS"
echo "  RR Range:     $RR_START ~ $RR_END (step $RR_STEP)"
echo "  Results:      $RESULTS_DIR"
echo "============================================"

mkdir -p "$RESULTS_DIR"

# ========== 日志重定向 ==========
LOG_FILE="$RESULTS_DIR/benchmark.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# ========== 保存实验配置 ==========
cat > "$RESULTS_DIR/config.json" <<EOF
{
  "framework": "${FRAMEWORK}",
  "model": "${MODEL_PATH}",
  "model_name": "${MODEL_NAME}",
  "base_url": "${BASE_URL}",
  "dataset_name": "${DATASET_NAME}",
  "dataset_path": "${DATASET_PATH}",
  "random_input": ${RANDOM_INPUT},
  "random_output": ${RANDOM_OUTPUT},
  "min_input_len": ${MIN_INPUT_LEN},
  "max_input_len": ${MAX_INPUT_LEN},
  "num_prompts": ${NUM_PROMPTS},
  "rr_range": {
    "start": ${RR_START},
    "end": ${RR_END},
    "step": ${RR_STEP}
  },
  "timestamp": "${TIMESTAMP}",
  "exp_label": "${EXP_LABEL}"
}
EOF

echo "[INFO] Config saved to $RESULTS_DIR/config.json"

# ========== 等待服务完全启动（warmup 完成后再开始） ==========
echo "[INFO] Waiting for server to be ready (warmup may be running)..."
MAX_WAIT=600
WAIT_COUNT=0
while true; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/v1/models" --max-time 5 || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        # Server responds, do a quick test request to verify warmup is done
        TEST_RESP=$(curl -s -X POST "$BASE_URL/generate" \
            -H "Content-Type: application/json" \
            -d '{"text": "Hello", "sampling_params": {"max_new_tokens": 4, "temperature": 0}}' \
            --max-time 30 2>&1 || echo "FAILED")
        if echo "$TEST_RESP" | grep -q "text\|generated_text\|outputs"; then
            echo "[INFO] Server is ready (warmup complete)"
            break
        fi
    fi
    WAIT_COUNT=$((WAIT_COUNT + 5))
    if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
        echo "[ERROR] Timeout waiting for server to be ready after ${MAX_WAIT}s"
        exit 1
    fi
    echo "  Waiting... ${WAIT_COUNT}s / ${MAX_WAIT}s (server may still be warming up)"
    sleep 5
done

# ========== 获取模型名称 ==========
MODEL_ID=$(curl -s "$BASE_URL/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "$MODEL_NAME")
echo "[INFO] Model ID: $MODEL_ID"

# ========== 遍历 request rate ==========
CURRENT_RR=$RR_START
while [ "$(echo "$CURRENT_RR $RR_END" | awk '{print ($1 <= $2)}')" -eq 1 ]; do
    RR=$CURRENT_RR
    echo ""
    echo ">>> ============================================"
    echo ">>> Benchmark RR=$RR req/s"
    echo ">>> ============================================"

    RR_DIR="$RESULTS_DIR/rr_${RR}"
    mkdir -p "$RR_DIR"

    # 使用 miniconda 的 sglang 环境
    source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
    conda activate sglang

    # 构建数据集参数
    if [ "$DATASET_NAME" = "random" ]; then
        DATASET_ARGS="--dataset-name random --random-input-len $RANDOM_INPUT --random-output-len $RANDOM_OUTPUT"
    elif [ "$DATASET_NAME" = "sharegpt" ]; then
        DATASET_ARGS="--dataset-name sharegpt --dataset-path $DATASET_PATH"
        # --sharegpt-context-len 接受单个值（最大 context 长度）
        if [ "$MAX_INPUT_LEN" != "0" ]; then
            DATASET_ARGS="$DATASET_ARGS --sharegpt-context-len $MAX_INPUT_LEN"
        fi
    else
        DATASET_ARGS="--dataset-name $DATASET_NAME"
    fi

    # 运行 benchmark
    python -m sglang.bench_serving \
        --backend sglang \
        --base-url "$BASE_URL" \
        --model "$MODEL_ID" \
        $DATASET_ARGS \
        --num-prompts "$NUM_PROMPTS" \
        --request-rate "$RR" \
        --output-file "$RR_DIR/result.json" \
        --output-details \
        --seed 42

    # 查找并重命名结果文件
    GENERATED_JSON=$(ls "$RR_DIR"/*.json 2>/dev/null | head -1)
    if [ -n "$GENERATED_JSON" ] && [ -f "$GENERATED_JSON" ]; then
        if [ "$GENERATED_JSON" != "$RR_DIR/result.json" ]; then
            mv "$GENERATED_JSON" "$RR_DIR/result.json"
        fi
        echo "[OK] RR=$RR done -> $RR_DIR/result.json"
    else
        echo "[WARN] RR=$RR result file not found"
    fi

    sleep 3
    CURRENT_RR=$(echo "$CURRENT_RR $RR_STEP" | awk '{printf "%.6g", $1 + $2}')
done

# ========== 生成汇总报告 ==========
echo ""
echo ">>> ============================================"
echo ">>> Generating summary report..."
echo ">>> ============================================"

python3 - "$RESULTS_DIR" "$FRAMEWORK" "$MODEL_NAME" <<'PYEOF'
import json
import glob
import os
import sys

results_dir = sys.argv[1]
framework = sys.argv[2] if len(sys.argv) > 2 else "sglang"
model_name = sys.argv[3] if len(sys.argv) > 3 else "unknown"

summary = {
    "metadata": {
        "framework": framework,
        "model": model_name,
        "results_dir": results_dir
    },
    "experiments": []
}

rr_dirs = sorted(glob.glob(os.path.join(results_dir, "rr_*")),
                 key=lambda x: float(x.split("_")[-1]))

for rr_dir in rr_dirs:
    result_file = os.path.join(rr_dir, "result.json")
    if not os.path.exists(result_file):
        print(f"[WARN] Skip {rr_dir} (no result.json)")
        continue
    try:
        with open(result_file) as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to parse {result_file}: {e}")
        continue

    rr = data.get("request_rate", "unknown")
    if isinstance(rr, str) and rr != "inf":
        try:
            rr = float(rr)
        except:
            pass

    # SGLang 输出字段映射
    exp = {
        "rr": rr,
        "throughput_req_s": data.get("request_throughput"),
        "output_throughput_tok_s": data.get("output_throughput"),
        "total_token_throughput": data.get("total_throughput"),
        "completed": data.get("completed"),
        "failed": data.get("failed"),
        "duration_s": data.get("duration"),
        "ttft_ms": {
            "mean": data.get("mean_ttft_ms"),
            "median": data.get("median_ttft_ms"),
            "std": data.get("std_ttft_ms"),
            "p99": data.get("p99_ttft_ms"),
        },
        "tpot_ms": {
            "mean": data.get("mean_tpot_ms"),
            "median": data.get("median_tpot_ms"),
            "std": data.get("std_tpot_ms"),
            "p99": data.get("p99_tpot_ms"),
        },
        "itl_ms": {
            "mean": data.get("mean_itl_ms"),
            "median": data.get("median_itl_ms"),
            "std": data.get("std_itl_ms"),
            "p99": data.get("p99_itl_ms"),
        },
        "e2el_ms": {
            "mean": data.get("mean_e2el_ms"),
            "median": data.get("median_e2el_ms"),
            "std": data.get("std_e2el_ms"),
            "p99": data.get("p99_e2el_ms"),
        },
        "max_concurrent_requests": data.get("max_concurrent_requests"),
        "max_output_tokens_per_s": data.get("max_output_tokens_per_s"),
    }
    summary["experiments"].append(exp)

    tp = exp.get("throughput_req_s", 0)
    ttft_mean = exp.get("ttft_ms", {}).get("mean", 0) or 0
    tpot_mean = exp.get("tpot_ms", {}).get("mean", 0) or 0
    print(f"  RR={rr:>4}: throughput={tp:.2f} req/s, TTFT_mean={ttft_mean:.1f}ms, TPOT_mean={tpot_mean:.2f}ms")

summary_file = os.path.join(results_dir, "summary.json")
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# ========== 生成 summary.md ==========
config_file = os.path.join(results_dir, "config.json")
config_data = {}
if os.path.exists(config_file):
    with open(config_file) as f:
        config_data = json.load(f)

lines = []
lines.append("# Serving Benchmark Result Summary")
lines.append("")
lines.append("## Experiment Config")
lines.append(f"- **Framework:** {config_data.get('framework', 'N/A')}")
lines.append(f"- **Model:** {config_data.get('model', 'N/A')}")
lines.append(f"- **Dataset:** {config_data.get('dataset_name', 'N/A')}")
lines.append(f"- **Random Input/Output:** {config_data.get('random_input', 'N/A')} / {config_data.get('random_output', 'N/A')}")
lines.append(f"- **Num Prompts:** {config_data.get('num_prompts', 'N/A')}")
rr_cfg = config_data.get('rr_range', {})
lines.append(f"- **RR Range:** {rr_cfg.get('start', '?')} ~ {rr_cfg.get('end', '?')} (step {rr_cfg.get('step', '?')})")
lines.append(f"- **Timestamp:** {config_data.get('timestamp', 'N/A')}")
lines.append(f"- **Label:** {config_data.get('exp_label', 'N/A')}")
lines.append("")
lines.append("## Benchmark Results")
lines.append("")
lines.append("| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

for exp in summary.get("experiments", []):
    rr = exp.get("rr", "N/A")
    tp = exp.get("throughput_req_s", 0)
    ttft_mean = exp.get("ttft_ms", {}).get("mean", 0) or 0
    ttft_p99 = exp.get("ttft_ms", {}).get("p99", 0) or 0
    tpot_mean = exp.get("tpot_ms", {}).get("mean", 0) or 0
    tpot_p99 = exp.get("tpot_ms", {}).get("p99", 0) or 0
    itl_mean = exp.get("itl_ms", {}).get("mean", 0) or 0
    completed = exp.get("completed", 0)
    failed = exp.get("failed", 0)
    lines.append(
        f"| {rr} | {tp:.2f} | {ttft_mean:.2f} | {ttft_p99:.2f} "
        f"| {tpot_mean:.2f} | {tpot_p99:.2f} | {itl_mean:.2f} "
        f"| {completed} | {failed} |"
    )

lines.append("")
lines.append("## Detailed Metrics")
lines.append("")
lines.append("| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |")
lines.append("|---|---:|---:|---:|---:|---:|---:|")

for exp in summary.get("experiments", []):
    rr = exp.get("rr", "N/A")
    tpot_med = exp.get("tpot_ms", {}).get("median", 0) or 0
    itl_med = exp.get("itl_ms", {}).get("median", 0) or 0
    e2el_mean = exp.get("e2el_ms", {}).get("mean", 0) or 0
    e2el_p99 = exp.get("e2el_ms", {}).get("p99", 0) or 0
    out_tp = exp.get("output_throughput_tok_s", 0) or 0
    peak_conc = exp.get("max_concurrent_requests", 0) or 0
    lines.append(
        f"| {rr} | {tpot_med:.2f} | {itl_med:.2f} | {e2el_mean:.2f} "
        f"| {e2el_p99:.2f} | {out_tp:.2f} | {peak_conc:.0f} |"
    )

summary_md = os.path.join(results_dir, "summary.md")
with open(summary_md, "w") as f:
    f.write("\n".join(lines))

print(f"\n[OK] Summary saved: {summary_file}")
print(f"[OK] Summary MD saved: {summary_md}")
PYEOF

echo ""
echo "============================================"
echo "  All Benchmarks Complete!"
echo "============================================"
echo "  Results:      $RESULTS_DIR"
echo "  Summary:      $RESULTS_DIR/summary.json"
echo "  Summary MD:   $RESULTS_DIR/summary.md"
echo "============================================"
