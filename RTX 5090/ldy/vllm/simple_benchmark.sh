vllm bench serve \
    --host 127.0.0.1 \
    --port 8001 \
    --model /root/autodl-tmp/models/Qwen1.5-MoE-A2.7B \
    --dataset-name sharegpt \
    --dataset-path /root/autodl-tmp/dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
    --request-rate 10 \
    --burstiness 2 \
    --num-prompts 200 \

