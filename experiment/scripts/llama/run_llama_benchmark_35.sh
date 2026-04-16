#!/bin/bash
#
# llama.cpp Benchmark Runner
# 参考 SGLang/vLLM benchmark 脚本结构，为 llama-server 设计的 benchmark 测试
#
# Usage: bash run_llama_benchmark.sh
# 环境变量:
#   MODEL_PATH          - GGUF 模型路径 (default: /root/autodl-tmp/models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf)
#   BASE_URL           - server URL (default: http://localhost:8080)
#   RR_START           - request rate 起始值 (default: 1)
#   RR_END             - request rate 结束值 (default: 10)
#   RR_STEP            - request rate 步进 (default: 1)
#   NUM_PROMPTS        - 测试 prompt 数量 (default: 500)
#   EXP_LABEL          - 实验标签 (default: llama)
#   RANDOM_INPUT       - 随机输入长度 (default: 2048)
#   RANDOM_OUTPUT      - 随机输出长度 (default: 512)
#

set -e

# ========== 参数配置 ==========
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen3.5-35B-A3B-GGUF}"
BASE_URL="${BASE_URL:-http://localhost:8088}"
DATASET_PATH="/root/autodl-tmp/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"
NUM_PROMPTS="${NUM_PROMPTS:-100}"

RR_START="${RR_START:-0.5}"
RR_END="${RR_END:-4}"
RR_STEP="${RR_STEP:-0.5}"
EXP_LABEL="${EXP_LABEL:-llama}"
DATASET_NAME="${DATASET_NAME:-sharegpt}"

RANDOM_INPUT="${RANDOM_INPUT:-2048}"
RANDOM_OUTPUT="${RANDOM_OUTPUT:-512}"
MIN_INPUT_LEN="${MIN_INPUT_LEN:-0}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-0}"

FRAMEWORK="llama"
MODEL_NAME="qwen3.5-35b-a3b-gguf-bf16"

# ========== 目录初始化 ==========
if [ -z "${RESULTS_DIR:-}" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RESULTS_DIR="/root/autodl-tmp/workspace/experiment/results/${FRAMEWORK}/${MODEL_NAME}/${TIMESTAMP}_${EXP_LABEL}"
fi

echo "============================================"
echo "  llama.cpp Benchmark Experiment Config"
echo "============================================"
echo "  Model:        $MODEL_PATH"
echo "  Base URL:     $BASE_URL"
echo "  Dataset:      $DATASET_NAME"
echo "  Num Prompts:  $NUM_PROMPTS"
echo "  RR Range:     $RR_START ~ $RR_END (step $RR_STEP)"
echo "  Results:      $RESULTS_DIR"
echo "============================================"

mkdir -p "$RESULTS_DIR"

# ========== 保存配置 ==========
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

# ========== 检查服务是否可用 ==========
echo "[INFO] Checking server availability..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/v1/models" --max-time 5 || echo "000")
if [ "$HTTP_CODE" != "200" ]; then
    echo "[ERROR] Server not available (HTTP $HTTP_CODE). Please start server first:"
    echo "  CUDA_VISIBLE_DEVICES=0 llama-server -m $MODEL_PATH -ngl 99 --port 8080"
    exit 1
fi
echo "[INFO] Server is ready"

# ========== 获取模型信息 ==========
MODEL_ID=$(curl -s "$BASE_URL/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "$MODEL_NAME")
echo "[INFO] Model ID: $MODEL_ID"

# ========== 预处理数据集 ==========
if [ "$MIN_INPUT_LEN" != "0" ] || [ "$MAX_INPUT_LEN" != "0" ]; then
    FILTERED_DATASET="${RESULTS_DIR}/dataset_filtered.json"
    echo "[INFO] Filtering dataset by prompt length..."
    python3 /root/autodl-tmp/workspace/experiment/scripts/filter_sharegpt.py \
        "$DATASET_PATH" "$FILTERED_DATASET" \
        --min-len "$MIN_INPUT_LEN" \
        --max-len "$MAX_INPUT_LEN"
    DATASET_PATH="$FILTERED_DATASET"
fi

# ========== 遍历 request rate ==========
CURRENT_RR=$RR_START
while awk "BEGIN {exit !($CURRENT_RR <= $RR_END)}"; do
    RR=$CURRENT_RR
    echo ""
    echo ">>> ============================================"
    echo ">>> Benchmark RR=$RR req/s"
    echo ">>> ============================================"

    RR_DIR="$RESULTS_DIR/rr_${RR}"
    mkdir -p "$RR_DIR"

    python3 - "$RR_DIR" "$RR" "$BASE_URL" "$DATASET_PATH" "$NUM_PROMPTS" "$DATASET_NAME" "$MODEL_NAME" <<'PYEOF'
import asyncio
import json
import os
import sys
import time
import aiohttp
from tokenizers import Tokenizer

rr_dir = sys.argv[1]
request_rate = float(sys.argv[2])
base_url = sys.argv[3]
dataset_path = sys.argv[4]
num_prompts = int(sys.argv[5])
dataset_name = sys.argv[6]
model_name = sys.argv[7]

# ========== tokenizer ==========
_tokenizer_path = "/root/autodl-tmp/models/Qwen3-30B-A3B/tokenizer.json"
try:
    tokenizer = Tokenizer.from_file(_tokenizer_path)
    print(f"[INFO] Loaded tokenizer, vocab_size={tokenizer.get_vocab_size()}")
except Exception as e:
    print(f"[WARN] Failed to load tokenizer: {e}")
    tokenizer = None

def count_tokens(text):
    if tokenizer:
        return len(tokenizer.encode(text).ids)
    return len(text) // 4

# ========== dataset ==========
with open(dataset_path, "r") as f:
    raw = json.load(f)

prompts = []
for item in raw:
    if isinstance(item, dict):
        if 'conversations' in item:
            for c in item['conversations']:
                if c.get("from") in ("human", "user"):
                    prompts.append(c["value"])
                    break
        elif 'text' in item:
            prompts.append(item["text"])

prompts = prompts[:num_prompts]
input_tokens_list = [count_tokens(p) for p in prompts]
total_input_tokens = sum(input_tokens_list)
print(f"[INFO] {len(prompts)} prompts, total input tokens: {total_input_tokens}")

# ========== benchmark state ==========
results = []
results_lock = asyncio.Lock()
counters = {"completed": 0, "failed": 0}
active_requests = 0
max_concurrency = 0
concurrency_lock = asyncio.Lock()
AIOHTTP_TIMEOUT = None  # No timeout

async def send_request(session, prompt, input_tokens, req_id):
    global active_requests, max_concurrency

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 512,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    ttft = 0.0
    st = time.perf_counter()
    most_recent_timestamp = st
    first_chunk_received = False
    output_tokens_from_usage = None
    itl_ms_list = []
    accumulated_text = []

    try:
        async with concurrency_lock:
            active_requests += 1
            max_concurrency = max(max_concurrency, active_requests)

        async with session.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=AIOHTTP_TIMEOUT,
        ) as resp:
            if resp.status != 200:
                err_body = await resp.text()
                async with results_lock:
                    counters["failed"] += 1
                print(f"[WARN] Request {req_id} failed: HTTP {resp.status} {err_body[:80]}")
                return

            async for chunk_bytes in resp.content:
                chunk_bytes = chunk_bytes.strip()
                if not chunk_bytes:
                    continue
                chunk = chunk_bytes.decode("utf-8")
                # SSE comment lines (": ping") — skip
                if chunk.startswith(":"):
                    continue
                chunk = chunk.removeprefix("data: ")
                if chunk == "[DONE]":
                    break

                try:
                    data = json.loads(chunk)
                except json.JSONDecodeError:
                    continue

                timestamp = time.perf_counter()

                # chat completions streaming delta
                if choices := data.get("choices"):
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        accumulated_text.append(content)
                        if not first_chunk_received:
                            first_chunk_received = True
                            ttft = timestamp - st
                        else:
                            itl_ms_list.append((timestamp - most_recent_timestamp) * 1000)
                        most_recent_timestamp = timestamp

                # usage summary (last chunk with stream_options: include_usage)
                if usage := data.get("usage"):
                    output_tokens_from_usage = usage.get("completion_tokens")

        end = time.perf_counter()
        duration = end - st

        output_tokens = output_tokens_from_usage
        if output_tokens is None:
            output_tokens = count_tokens("".join(accumulated_text))

        if output_tokens and output_tokens > 1:
            tpot = (duration - ttft) * 1000 / (output_tokens - 1)
        else:
            tpot = 0.0

        async with results_lock:
            counters["completed"] += 1
            results.append({
                "ttft_ms": ttft * 1000,
                "tpot_ms": tpot,
                "duration_ms": duration * 1000,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens or 0,
                "itl_ms": itl_ms_list,
            })

    except asyncio.TimeoutError:
        async with results_lock:
            counters["failed"] += 1
        print(f"[WARN] Request {req_id} timed out")
    except aiohttp.ClientError as e:
        async with results_lock:
            counters["failed"] += 1
        print(f"[WARN] Request {req_id} connection error: {e}")
    except Exception as e:
        async with results_lock:
            counters["failed"] += 1
        print(f"[WARN] Request {req_id} failed: {e}")
    finally:
        async with concurrency_lock:
            active_requests -= 1

# ========== open-loop request scheduler ==========
interval = 1.0 / request_rate if request_rate > 0 else 0
print(f"[INFO] Benchmark RR={request_rate}, interval={interval:.4f}s, prompts={len(prompts)}")

async def benchmark():
    global max_concurrency
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        start_time = time.perf_counter()
        for i, prompt in enumerate(prompts):
            fire_time = start_time + i * interval
            delay = fire_time - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(
                send_request(session, prompt, input_tokens_list[i], i)
            ))
        await asyncio.gather(*tasks)

    end_time = time.perf_counter()
    return end_time - start_time

total_duration = asyncio.run(benchmark())

# ========== stats ==========
if not results:
    print("[ERROR] No successful requests")
    sys.exit(1)

def pct(arr, p):
    if not arr:
        return 0.0
    s = sorted(arr)
    return s[min(int(len(s) * p / 100), len(s) - 1)]

ttfts = [r["ttft_ms"] for r in results]
tpots = [r["tpot_ms"] for r in results]
durations = [r["duration_ms"] for r in results]
itls = []
for r in results:
    itls.extend(r.get("itl_ms", []))

total_output_tokens = sum(r["output_tokens"] for r in results)
throughput = counters["completed"] / total_duration
total_token_throughput = (total_input_tokens + total_output_tokens) / total_duration

summary = {
    "request_rate": request_rate,
    "completed": counters["completed"],
    "failed": counters["failed"],
    "duration": total_duration,
    "total_input_tokens": total_input_tokens,
    "total_output_tokens": total_output_tokens,
    "request_throughput": throughput,
    "total_token_throughput": total_token_throughput,
    "max_concurrent_requests": max_concurrency,
    "ttft_ms": {
        "mean": sum(ttfts) / len(ttfts),
        "median": pct(ttfts, 50),
        "std": (sum((x - sum(ttfts)/len(ttfts))**2 for x in ttfts) / len(ttfts)) ** 0.5,
        "p99": pct(ttfts, 99),
    },
    "tpot_ms": {
        "mean": sum(tpots) / len(tpots),
        "median": pct(tpots, 50),
        "std": (sum((x - sum(tpots)/len(tpots))**2 for x in tpots) / len(tpots)) ** 0.5,
        "p99": pct(tpots, 99),
    },
    "itl_ms": {
        "mean": sum(itls) / len(itls) if itls else 0,
        "median": pct(itls, 50),
        "std": (sum((x - sum(itls)/len(itls))**2 for x in itls) / len(itls)) ** 0.5 if itls else 0,
        "p99": pct(itls, 99),
    },
    "e2el_ms": {
        "mean": sum(durations) / len(durations),
        "median": pct(durations, 50),
        "std": (sum((x - sum(durations)/len(durations))**2 for x in durations) / len(durations)) ** 0.5,
        "p99": pct(durations, 99),
    },
}

out_file = os.path.join(rr_dir, "result.json")
with open(out_file, "w") as f:
    json.dump(summary, f, indent=2)

print(f"[OK] RR={request_rate}: throughput={throughput:.2f} req/s, "
      f"total_tok/s={total_token_throughput:.2f}, "
      f"TTFT_mean={summary['ttft_ms']['mean']:.1f}ms, "
      f"TPOT_mean={summary['tpot_ms']['mean']:.2f}ms, "
      f"ITL_mean={summary['itl_ms']['mean']:.2f}ms, "
      f"max_conc={max_concurrency}, "
      f"completed={counters['completed']}, failed={counters['failed']}")
PYEOF

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
framework = sys.argv[2] if len(sys.argv) > 2 else "llama"
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
            "mean": data.get("ttft_ms", {}).get("mean"),
            "median": data.get("ttft_ms", {}).get("median"),
            "std": data.get("ttft_ms", {}).get("std"),
            "p99": data.get("ttft_ms", {}).get("p99"),
        },
        "tpot_ms": {
            "mean": data.get("tpot_ms", {}).get("mean"),
            "median": data.get("tpot_ms", {}).get("median"),
            "std": data.get("tpot_ms", {}).get("std"),
            "p99": data.get("tpot_ms", {}).get("p99"),
        },
        "itl_ms": {
            "mean": data.get("itl_ms", {}).get("mean"),
            "median": data.get("itl_ms", {}).get("median"),
            "std": data.get("itl_ms", {}).get("std"),
            "p99": data.get("itl_ms", {}).get("p99"),
        },
        "e2el_ms": {
            "mean": data.get("e2el_ms", {}).get("mean"),
            "median": data.get("e2el_ms", {}).get("median"),
            "std": data.get("e2el_ms", {}).get("std"),
            "p99": data.get("e2el_ms", {}).get("p99"),
        },
        "max_concurrent_requests": data.get("max_concurrent_requests"),
        "max_output_tokens_per_s": None,
    }
    summary["experiments"].append(exp)
    print(f"  RR={rr:>4}: throughput={exp['throughput_req_s']:.2f} req/s, "
          f"TTFT_mean={exp['ttft_ms']['mean']:.1f}ms, "
          f"TPOT_mean={exp['tpot_ms']['mean']:.2f}ms, "
          f"ITL_mean={exp['itl_ms']['mean']:.2f}ms")

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
lines.append("| RR | TPOT Median (ms) | ITL Median (ms) | E2EL Mean (ms) | E2EL P99 (ms) | Total Throughput (tok/s) | Peak Concurrent |")
lines.append("|---|---:|---:|---:|---:|---:|---:|")

for exp in summary.get("experiments", []):
    rr = exp.get("rr", "N/A")
    tpot_med = exp.get("tpot_ms", {}).get("median", 0) or 0
    itl_med = exp.get("itl_ms", {}).get("median", 0) or 0
    e2el_mean = exp.get("e2el_ms", {}).get("mean", 0) or 0
    e2el_p99 = exp.get("e2el_ms", {}).get("p99", 0) or 0
    out_tp = exp.get("total_token_throughput", 0) or 0
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
echo "  Summary:     $RESULTS_DIR/summary.json"
echo "  Summary MD:  $RESULTS_DIR/summary.md"
echo "============================================"