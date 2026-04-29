#!/usr/bin/env python3
"""
过滤 ShareGPT 数据集中的 prompt 长度范围
Usage: python filter_sharegpt.py <input.json> <output.json> <min_len> <max_len>
"""

import json
import sys
import argparse
from pathlib import Path


def estimate_tokens(text, tokenizer=None):
    """估算 token 数量（使用简单的中文/英文分词估算）"""
    if tokenizer:
        return len(tokenizer.encode(text))

    # 简单估算：中文按字符数，英文按单词数
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    english_words = len(text.split())
    other_chars = len(text) - chinese_chars - sum(1 for c in text if c.isascii() and c.isalpha())

    # 粗略估算：中文约 1.5 token/字符，英文约 1.3 token/词
    return int(chinese_chars * 1.5 + english_words * 1.3 + other_chars * 0.5)


def main():
    parser = argparse.ArgumentParser(description='Filter ShareGPT dataset by prompt length')
    parser.add_argument('input', help='Input JSON file path')
    parser.add_argument('output', help='Output JSON file path')
    parser.add_argument('--min-len', type=int, default=0, help='Minimum prompt length in tokens (default: 0)')
    parser.add_argument('--max-len', type=int, default=0, help='Maximum prompt length in tokens, 0 means no limit (default: 0)')
    parser.add_argument('--tokenizer', type=str, default=None, help='Tokenizer name or path (optional)')

    args = parser.parse_args()

    min_len = args.min_len
    max_len = args.max_len if args.max_len > 0 else float('inf')

    print(f"Loading dataset from: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    print(f"Total prompts: {total}")

    # 尝试加载 tokenizer
    tokenizer = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
            print(f"Loaded tokenizer: {args.tokenizer}")
        except Exception as e:
            print(f"Warning: Failed to load tokenizer: {e}")
            print("Using simple character-based estimation")

    filtered = []
    for item in data:
        # 提取 conversation 中的 user prompt
        if isinstance(item, dict):
            # 尝试多个可能的字段
            if 'conversations' in item:
                conversations = item['conversations']
                # 找到第一个 human/user 的 turn
                prompt = ""
                for conv in conversations:
                    if isinstance(conv, dict) and conv.get('from') in ('human', 'user'):
                        prompt = conv.get('value', '')
                        break
                    elif isinstance(conv, list) and len(conv) >= 2:
                        role, content = conv[0], conv[1]
                        if role.lower() in ('human', 'user'):
                            prompt = content
                            break
            elif 'text' in item:
                prompt = item['text']
            else:
                prompt = str(item)
        else:
            prompt = str(item)

        # 计算长度
        token_len = estimate_tokens(prompt, tokenizer)

        if min_len <= token_len <= max_len:
            filtered.append(item)

    print(f"Filtered prompts ({min_len}~{max_len if max_len != float('inf') else 'inf'} tokens): {len(filtered)}")
    print(f"Removed: {total - len(filtered)}")

    # 保存过滤后的数据集
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"Saved to: {output_path}")


if __name__ == '__main__':
    main()