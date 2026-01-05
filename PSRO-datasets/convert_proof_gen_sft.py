#!/usr/bin/env python3
"""
Convert aops_instruct_train.csv to SFT format for proof generation training.
Uses the last 50k Q-A pairs to avoid data leakage for future RL training.
"""

import json
import ast
import re
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

# ============ Configuration ============
CSV_FILE = "aops_instruct_train.csv"
OUTPUT_FILE = "/mnt/shared-storage-user/tanzelin-p/sft_data/proof_gen_sft_10k.json"
MODEL_PATH = "/mnt/shared-storage-user/ma4agi-gpu/data/model/Qwen3-4B-Instruct-2507"
MAX_SEQ_LENGTH = 4096
NUM_PAIRS = 10000

INSTRUCTION = """Your task is to solve a given problem. The problem may ask you to prove a statement, or ask for an answer. If finding an answer is required, you should come up with the answer, and your final solution should also be a rigorous proof of that answer being valid.

Your final solution to the problem should be exceptionally comprehensive and easy-to-follow. A good solution should satisfy the following criteria:

- The solution should be completely correct, with all steps executed properly and clearly demonstrated.
- The proof must be rigorous. Every step must be logically justified and clearly explained.
- Additionally, referencing anything from any paper does not save the need to prove the reference. It is okay IF AND ONLY IF the solution also presents a valid proof of the reference argument(s); otherwise, if the solution omits the proof or if the proof provided is not completely correct, the solution is incomplete."""


def parse_messages(msg_str):
    """Parse messages string to extract question and answer."""
    try:
        # Fix the format: add comma between dicts
        fixed = re.sub(r'\}\s*\n\s*\{', '}, {', msg_str)
        messages = ast.literal_eval(fixed)

        question = None
        answer = None
        for msg in messages:
            if msg['role'] == 'user':
                question = msg['content']
            elif msg['role'] == 'assistant':
                answer = msg['content']

        return question, answer
    except:
        return None, None


def main():
    print(f"Loading tokenizer from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print(f"Reading CSV: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE)
    print(f"Total rows: {len(df)}")

    # Take first 10k
    df_first = df.head(NUM_PAIRS)
    print(f"Using first {len(df_first)} rows")

    sft_data = []
    filtered = 0
    token_lengths = []

    for idx, row in tqdm(df_first.iterrows(), total=len(df_first), desc="Processing"):
        question, answer = parse_messages(row['messages'])

        if not question or not answer:
            continue

        # Fix blacksquare issues:
        # Original CSV has $\blacksquare$ but \b was interpreted as backspace (0x08)
        # So we get $\x08lacksquare$ which needs to become $\blacksquare$
        answer = answer.replace('\x08lacksquare', '\\blacksquare')
        question = question.replace('\x08lacksquare', '\\blacksquare')

        sft_item = {
            "instruction": INSTRUCTION,
            "input": question,
            "output": answer
        }

        # Check token length
        full_text = INSTRUCTION + "\n\n" + question + "\n\n" + answer
        token_len = len(tokenizer.encode(full_text))

        if token_len <= MAX_SEQ_LENGTH:
            sft_data.append(sft_item)
            token_lengths.append(token_len)
        else:
            filtered += 1

    # Stats
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Kept: {len(sft_data)}")
    print(f"  Filtered (>{MAX_SEQ_LENGTH} tokens): {filtered}")

    if token_lengths:
        print(f"\nToken length stats:")
        print(f"  Min: {min(token_lengths)}, Max: {max(token_lengths)}, Avg: {sum(token_lengths)//len(token_lengths)}")

    # Save
    print(f"\nSaving to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

    print(f"Done! Saved {len(sft_data)} samples")

    # Show sample
    if sft_data:
        print(f"\nSample:")
        print(f"  Q: {sft_data[0]['input'][:100]}...")
        print(f"  A: {sft_data[0]['output'][:100]}...")


if __name__ == "__main__":
    main()
