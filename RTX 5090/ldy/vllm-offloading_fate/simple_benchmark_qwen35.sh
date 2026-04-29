PYTHONPATH=/root/autodl-tmp/workspace/ldy/fate-test:$PYTHONPATH \
vllm bench serve \
    --host 127.0.0.1 \
    --port 8001 \
    --model /root/autodl-tmp/models/Qwen3.5-35B-A3B \
    --dataset-name sharegpt \
    --dataset-path /root/autodl-tmp/dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
    --request-rate 6 \
    --num-prompts 1000 \
    --save-result \
    --save-detailed \
    --result-dir /root/autodl-tmp/workspace/ldy/vllm-offloading_fate/qwen35_result_new

