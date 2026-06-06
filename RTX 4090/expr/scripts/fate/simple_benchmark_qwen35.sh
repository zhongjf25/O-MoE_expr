#!/bin/bash
#
# FATE Qwen3 Benchmark 脚本


set -e

# ========== 参数配置 ==========
BASE_URL="${BASE_URL:-http://localhost:8000}"
MODEL_PATH="${MODEL_PATH:-/data/share/models/qwen35_35b}"
DATASET_PATH="${DATASET_PATH:-/root/workspace/dataset/ShareGPT_V3_unfiltered_cleaned_split.json}"
RR_START="${RR_START:-1}"
RR_END="${RR_END:-8}"
RR_STEP="${RR_STEP:-1}"
NUM_PROMPTS="${NUM_PROMPTS:-250}"
EXP_LABEL="${EXP_LABEL:-fate-qwen35-35b}"

# 框架和模型标识
FRAMEWORK="fate"
MODEL_NAME="${MODEL_NAME:-qwen35-35b}" 

# ========== 目录初始化 ==========
# 如果外部传入了RESULTS_DIR则使用它，否则生成新的
if [ -z "$RESULTS_DIR" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RESULTS_DIR="/root/workspace/expr/results/${FRAMEWORK}/${MODEL_NAME}/${TIMESTAMP}_${EXP_LABEL}"
fi

echo "============================================"
echo "  FATE Qwen3 Benchmark Config"
echo "============================================"
echo "  Model:        $MODEL_PATH"
echo "  Base URL:     $BASE_URL"
echo "  Dataset:      $DATASET_PATH"
echo "  Num Prompts:  $NUM_PROMPTS"
echo "  RR Range:     $RR_START ~ $RR_END (step $RR_STEP)"
echo "  Results:      $RESULTS_DIR"
echo "============================================"

mkdir -p "$RESULTS_DIR"

# 日志文件
LOG_FILE="$RESULTS_DIR/benchmark_RR${RR_START}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[INFO] Log file: $LOG_FILE"

# ========== 保存实验配置 ==========
cat > "$RESULTS_DIR/config.json" <<EOF
{
  "framework": "${FRAMEWORK}",
  "model": "${MODEL_PATH}",
  "model_name": "${MODEL_NAME}",
  "base_url": "${BASE_URL}",
  "dataset_path": "${DATASET_PATH}",
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

# ========== 预检测：确认服务可连接 ==========
echo "[INFO] Checking server availability..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/v1/models" --max-time 5 || echo "000")
if [ "$HTTP_CODE" != "200" ]; then
    echo "[ERROR] Server not available (HTTP $HTTP_CODE). Please start server first:"
    echo "  bash /root/autodl-tmp/workspace/experiment/scripts/FATE/simple_serving_qwen3.sh"
    exit 1
fi
echo "[INFO] Server is ready"

# ========== 获取模型名称 ==========
MODEL_ID=$(curl -s "$BASE_URL/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "$MODEL_NAME")
echo "[INFO] Model ID: $MODEL_ID"

# ========== 遍历 request rate (支持浮点数) ==========
CURRENT_RR=$RR_START
while awk "BEGIN {exit !($CURRENT_RR <= $RR_END)}"; do
    RR=$CURRENT_RR
    echo ""
    echo ">>> ============================================"
    echo ">>> Benchmark RR=$RR req/s"
    echo ">>> ============================================"

    RR_DIR="$RESULTS_DIR/rr_${RR}"
    mkdir -p "$RR_DIR"

    # 运行 benchmark
    vllm bench serve \
        --backend openai \
        --base-url "$BASE_URL" \
        --model "$MODEL_ID" \
        --dataset-name sharegpt \
        --dataset-path "$DATASET_PATH" \
        --num-prompts "$NUM_PROMPTS" \
        --request-rate "$RR" \
        --save-result \
        --save-detailed \
        --result-dir "$RR_DIR" \
        --metadata "framework=${FRAMEWORK},model=${MODEL_NAME},request_rate=${RR}"

    # 查找并重命名结果文件
    GENERATED_JSON=$(ls "$RR_DIR"/*.json 2>/dev/null | head -1)
    if [ -n "$GENERATED_JSON" ] && [ -f "$GENERATED_JSON" ]; then
        mv "$GENERATED_JSON" "$RR_DIR/result.json"
        echo "[OK] RR=$RR done -> $RR_DIR/result.json"
    else
        echo "[WARN] RR=$RR result file not found"
    fi

    sleep 3
    CURRENT_RR=$(awk "BEGIN {printf \"%.2f\", $CURRENT_RR + $RR_STEP}")
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
framework = sys.argv[2] if len(sys.argv) > 2 else "omoevllm"
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

    exp = {
        "rr": rr,
        "throughput_req_s": data.get("request_throughput"),
        "output_throughput_tok_s": data.get("output_throughput"),
        "total_token_throughput": data.get("total_token_throughput"),
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
    print(f"  RR={rr:>4}: throughput={exp['throughput_req_s']:.2f} req/s, "
          f"TTFT_mean={exp['ttft_ms']['mean']:.1f}ms, "
          f"TPOT_mean={exp['tpot_ms']['mean']:.2f}ms")

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
lines.append(f"- **Dataset:** {config_data.get('dataset_path', 'N/A')}")
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