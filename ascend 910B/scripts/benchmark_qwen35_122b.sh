#!/usr/bin/env bash
#
# Shared Qwen3.5-122B serving benchmark.
#
# Usage:
#   bash benchmark_qwen35_122b.sh
#
# Any additional arguments are appended to `vllm bench serve`.

set -Eeuo pipefail

ACTIVATE_SCRIPT="${ACTIVATE_SCRIPT:-/home/ma-user/work/experiments/scripts/activate_vllm_ascend.sh}"
if [[ "${BASE_URL:-}" == http://* || "${BASE_URL:-}" == https://* ]]; then
    REQUESTED_BASE_URL="$BASE_URL"
else
    # ModelArts defines BASE_URL as a notebook route such as "/cedc...".
    REQUESTED_BASE_URL="http://127.0.0.1:8000"
fi
MODEL_PATH="${MODEL_PATH:-/home/ma-user/work/models/Qwen3.5-122B-A10B}"
DATASET_PATH="${DATASET_PATH:-/home/ma-user/work/models/ShareGPT_V3_unfiltered_cleaned_split.json}"
RR_START="${RR_START:-1}"
RR_END="${RR_END:-1}"
RR_STEP="${RR_STEP:-1}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
EXP_LABEL="${EXP_LABEL:-vllm-ascend-122b}"
FRAMEWORK="${FRAMEWORK:-vllm-ascend}"
MODEL_NAME="${MODEL_NAME:-qwen35-122b}"
BACKEND="${BACKEND:-openai}"
SEED="${SEED:-0}"
READY_TIMEOUT="${READY_TIMEOUT:-10}"
SHAREGPT_OUTPUT_LEN="${SHAREGPT_OUTPUT_LEN:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${RESULTS_DIR:-/home/ma-user/work/experiments/results/${FRAMEWORK}/${MODEL_NAME}/${TIMESTAMP}_${EXP_LABEL}}"

if [[ ! -f "$ACTIVATE_SCRIPT" ]]; then
    echo "[ERROR] Activation script not found: $ACTIVATE_SCRIPT" >&2
    exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[ERROR] Model path not found: $MODEL_PATH" >&2
    exit 1
fi
if [[ ! -f "$DATASET_PATH" ]]; then
    echo "[ERROR] Dataset not found: $DATASET_PATH" >&2
    exit 1
fi
if ! awk "BEGIN {exit !(($RR_STEP + 0) > 0)}"; then
    echo "[ERROR] RR_STEP must be positive: $RR_STEP" >&2
    exit 1
fi
if ! awk "BEGIN {exit !(($RR_START + 0) <= ($RR_END + 0))}"; then
    echo "[ERROR] RR_START must not exceed RR_END" >&2
    exit 1
fi

# Ascend vendor setup scripts read optional variables without default values.
set +u
# shellcheck disable=SC1090
source "$ACTIVATE_SCRIPT"
set -u
BASE_URL="$REQUESTED_BASE_URL"

# Utility Python processes only parse local JSON. Keeping O-MoE/Ascend out of
# their import path avoids plugin logs contaminating command substitution.
CLEAN_PYTHON=(
    env
    -u PYTHONPATH
    OMOE_ASCEND_EARLY_PATCH=0
    OMOE_QWEN35_TUPLE_SHARD_COMPAT=0
    OMOE_QWEN35_ASCEND_BF16_SSM=0
    BASELINE_QWEN35_TUPLE_SHARD_COMPAT=0
    BASELINE_QWEN35_ASCEND_BF16_SSM=0
    VLLM_PLUGINS=
    python3
)

mkdir -p "$RESULTS_DIR"
LOG_FILE="$RESULTS_DIR/benchmark.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo " Qwen3.5-122B Serving Benchmark ($FRAMEWORK)"
echo "============================================================"
echo " Model/tokenizer:   $MODEL_PATH"
echo " Base URL:          $BASE_URL"
echo " Dataset:           $DATASET_PATH"
echo " Backend:           $BACKEND"
echo " Num prompts:       $NUM_PROMPTS"
echo " Request rates:     $RR_START .. $RR_END (step $RR_STEP)"
echo " ShareGPT out len:  ${SHAREGPT_OUTPUT_LEN:-dataset default}"
echo " Results:           $RESULTS_DIR"
echo "============================================================"

MODELS_RESPONSE="$RESULTS_DIR/models.json"
echo "[INFO] Checking server availability..."
if ! HTTP_CODE=$(curl -sS -o "$MODELS_RESPONSE" -w "%{http_code}" \
    "$BASE_URL/v1/models" --max-time "$READY_TIMEOUT"); then
    echo "[ERROR] Cannot connect to $BASE_URL/v1/models" >&2
    exit 1
fi
if [[ "$HTTP_CODE" != "200" ]]; then
    echo "[ERROR] Server returned HTTP $HTTP_CODE for /v1/models" >&2
    exit 1
fi

MODEL_ID=$("${CLEAN_PYTHON[@]}" - "$MODELS_RESPONSE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    payload = json.load(stream)
print(payload["data"][0]["id"])
PY
)
echo "[INFO] Server model ID: $MODEL_ID"

"${CLEAN_PYTHON[@]}" - \
    "$RESULTS_DIR/config.json" \
    "$FRAMEWORK" \
    "$MODEL_PATH" \
    "$MODEL_NAME" \
    "$MODEL_ID" \
    "$BASE_URL" \
    "$DATASET_PATH" \
    "$NUM_PROMPTS" \
    "$RR_START" \
    "$RR_END" \
    "$RR_STEP" \
    "$TIMESTAMP" \
    "$EXP_LABEL" \
    "$BACKEND" \
    "$SEED" \
    "$SHAREGPT_OUTPUT_LEN" <<'PY'
import json
import sys

(
    output,
    framework,
    model,
    model_name,
    served_model_id,
    base_url,
    dataset_path,
    num_prompts,
    rr_start,
    rr_end,
    rr_step,
    timestamp,
    exp_label,
    backend,
    seed,
    sharegpt_output_len,
) = sys.argv[1:]

payload = {
    "framework": framework,
    "model": model,
    "model_name": model_name,
    "served_model_id": served_model_id,
    "base_url": base_url,
    "dataset_path": dataset_path,
    "num_prompts": int(num_prompts),
    "request_rate": {
        "start": float(rr_start),
        "end": float(rr_end),
        "step": float(rr_step),
    },
    "timestamp": timestamp,
    "exp_label": exp_label,
    "backend": backend,
    "seed": int(seed),
    "sharegpt_output_len": (
        int(sharegpt_output_len) if sharegpt_output_len else None
    ),
}
with open(output, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, ensure_ascii=False)
PY

DATASET_ARGS=()
if [[ -n "$SHAREGPT_OUTPUT_LEN" ]]; then
    DATASET_ARGS+=(--sharegpt-output-len "$SHAREGPT_OUTPUT_LEN")
fi

FAILURES=0
CURRENT_RR="$RR_START"
while awk "BEGIN {exit !(($CURRENT_RR + 0) <= ($RR_END + 0))}"; do
    RR="$CURRENT_RR"
    RR_LABEL=$(awk "BEGIN {printf \"%.4g\", $RR + 0}")
    RR_DIR="$RESULTS_DIR/rr_${RR_LABEL}"
    mkdir -p "$RR_DIR"

    echo
    echo ">>> Benchmark request rate: $RR req/s"
    BENCHMARK_OK=0
    if vllm bench serve \
        --backend "$BACKEND" \
        --base-url "$BASE_URL" \
        --model "$MODEL_ID" \
        --tokenizer "$MODEL_PATH" \
        --dataset-name sharegpt \
        --dataset-path "$DATASET_PATH" \
        --num-prompts "$NUM_PROMPTS" \
        --request-rate "$RR" \
        --seed "$SEED" \
        --save-result \
        --save-detailed \
        --result-dir "$RR_DIR" \
        --result-filename result.json \
        --metadata \
            "framework=$FRAMEWORK" \
            "model=$MODEL_NAME" \
            "request_rate=$RR" \
        "${DATASET_ARGS[@]}" \
        "$@"; then
        if "${CLEAN_PYTHON[@]}" - \
            "$RR_DIR/result.json" "$NUM_PROMPTS" <<'PY'
import json
import sys

result_path = sys.argv[1]
expected = int(sys.argv[2])
with open(result_path, encoding="utf-8") as stream:
    result = json.load(stream)

completed = int(result.get("completed", 0))
failed = int(result.get("failed", 0))
if completed != expected or failed != 0:
    raise SystemExit(
        f"benchmark result is incomplete: completed={completed}, "
        f"failed={failed}, expected={expected}"
    )
PY
        then
            BENCHMARK_OK=1
        fi
    fi

    if (( BENCHMARK_OK == 1 )); then
        echo "[OK] RR=$RR -> $RR_DIR/result.json"
    else
        echo "[ERROR] Benchmark failed for RR=$RR" >&2
        FAILURES=$((FAILURES + 1))
    fi

    CURRENT_RR=$(awk "BEGIN {printf \"%.10g\", $CURRENT_RR + $RR_STEP}")
    sleep 3
done

"${CLEAN_PYTHON[@]}" - "$RESULTS_DIR" "$FRAMEWORK" "$MODEL_NAME" <<'PY'
import glob
import json
import os
import sys

results_dir, framework, model_name = sys.argv[1:]


def metric(data, name):
    value = data.get(name)
    return value if isinstance(value, (int, float)) else None


def display(value, digits=2):
    return "N/A" if value is None else f"{value:.{digits}f}"


summary = {
    "metadata": {
        "framework": framework,
        "model": model_name,
        "results_dir": results_dir,
    },
    "experiments": [],
}

rr_dirs = sorted(
    glob.glob(os.path.join(results_dir, "rr_*")),
    key=lambda path: float(path.rsplit("_", 1)[-1]),
)

for rr_dir in rr_dirs:
    result_file = os.path.join(rr_dir, "result.json")
    if not os.path.exists(result_file):
        print(f"[WARN] Skip {rr_dir}: result.json is missing")
        continue
    try:
        with open(result_file, encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[WARN] Skip {result_file}: {exc}")
        continue

    rr = data.get("request_rate")
    if isinstance(rr, str) and rr != "inf":
        try:
            rr = float(rr)
        except ValueError:
            pass

    experiment = {
        "rr": rr,
        "throughput_req_s": metric(data, "request_throughput"),
        "output_throughput_tok_s": metric(data, "output_throughput"),
        "total_token_throughput": metric(data, "total_token_throughput"),
        "completed": data.get("completed"),
        "failed": data.get("failed"),
        "duration_s": metric(data, "duration"),
        "ttft_ms": {
            key: metric(data, f"{key}_ttft_ms")
            for key in ("mean", "median", "std", "p99")
        },
        "tpot_ms": {
            key: metric(data, f"{key}_tpot_ms")
            for key in ("mean", "median", "std", "p99")
        },
        "itl_ms": {
            key: metric(data, f"{key}_itl_ms")
            for key in ("mean", "median", "std", "p99")
        },
        "e2el_ms": {
            key: metric(data, f"{key}_e2el_ms")
            for key in ("mean", "median", "std", "p99")
        },
        "max_concurrent_requests": metric(data, "max_concurrent_requests"),
        "max_output_tokens_per_s": metric(data, "max_output_tokens_per_s"),
    }
    summary["experiments"].append(experiment)
    print(
        f"RR={rr}: throughput="
        f"{display(experiment['throughput_req_s'])} req/s, "
        f"TTFT={display(experiment['ttft_ms']['mean'], 1)} ms, "
        f"TPOT={display(experiment['tpot_ms']['mean'])} ms"
    )

summary_file = os.path.join(results_dir, "summary.json")
with open(summary_file, "w", encoding="utf-8") as stream:
    json.dump(summary, stream, indent=2, ensure_ascii=False)

with open(os.path.join(results_dir, "config.json"), encoding="utf-8") as stream:
    config = json.load(stream)

lines = [
    "# Serving Benchmark Result Summary",
    "",
    "## Experiment Config",
    f"- **Framework:** {config.get('framework', 'N/A')}",
    f"- **Model:** {config.get('model', 'N/A')}",
    f"- **Served model:** {config.get('served_model_id', 'N/A')}",
    f"- **Dataset:** {config.get('dataset_path', 'N/A')}",
    f"- **Num prompts:** {config.get('num_prompts', 'N/A')}",
    f"- **Request rate:** {config.get('request_rate', {})}",
    f"- **Timestamp:** {config.get('timestamp', 'N/A')}",
    f"- **Label:** {config.get('exp_label', 'N/A')}",
    "",
    "## Benchmark Results",
    "",
    "| RR | Throughput (req/s) | TTFT Mean (ms) | TTFT P99 (ms) | "
    "TPOT Mean (ms) | TPOT P99 (ms) | ITL Mean (ms) | Completed | Failed |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]

for item in summary["experiments"]:
    lines.append(
        f"| {item['rr']} | {display(item['throughput_req_s'])} | "
        f"{display(item['ttft_ms']['mean'])} | "
        f"{display(item['ttft_ms']['p99'])} | "
        f"{display(item['tpot_ms']['mean'])} | "
        f"{display(item['tpot_ms']['p99'])} | "
        f"{display(item['itl_ms']['mean'])} | "
        f"{item['completed']} | {item['failed']} |"
    )

lines.extend(
    [
        "",
        "## Detailed Metrics",
        "",
        "| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | "
        "E2EL P99 (ms) | Output Throughput (tok/s) | Peak Concurrent |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
)
for item in summary["experiments"]:
    lines.append(
        f"| {item['rr']} | {display(item['tpot_ms']['median'])} | "
        f"{display(item['itl_ms']['median'])} | "
        f"{display(item['e2el_ms']['mean'])} | "
        f"{display(item['e2el_ms']['p99'])} | "
        f"{display(item['output_throughput_tok_s'])} | "
        f"{display(item['max_concurrent_requests'], 0)} |"
    )

summary_md = os.path.join(results_dir, "summary.md")
with open(summary_md, "w", encoding="utf-8") as stream:
    stream.write("\n".join(lines) + "\n")

print(f"[OK] Summary JSON: {summary_file}")
print(f"[OK] Summary Markdown: {summary_md}")
PY

echo
echo "============================================================"
echo " Benchmark complete"
echo " Results: $RESULTS_DIR"
echo " Summary: $RESULTS_DIR/summary.json"
echo "============================================================"

if (( FAILURES > 0 )); then
    echo "[ERROR] $FAILURES request-rate run(s) failed" >&2
    exit 1
fi
