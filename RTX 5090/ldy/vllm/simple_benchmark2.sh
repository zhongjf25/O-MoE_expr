vllm bench serve \
    --host 127.0.0.1 \
    --port 8000 \
    --model /root/autodl-tmp/models/Qwen1.5-MoE-A2.7B \
    --dataset-name sharegpt \
    --dataset-path /root/autodl-tmp/dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
    --request-rate 1 \
    --num-prompts 20 \

