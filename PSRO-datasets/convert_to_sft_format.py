#!/usr/bin/env python3
"""
Convert dstest.jsonl.gz to ROLL SFT format

Use all dpsk scored data (0, 0.5, 1) for pipeline testing.
Focus on learning the verification analysis pattern.

Output: sft_scoring_train.json
"""

import json
import gzip
import random
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer

# ============ Configuration ============
DSTEST_FILE = "/home/tanzelin-p/PSRO4math/dstest.jsonl.gz"
OUTPUT_FILE = "sft_scoring_train.json"
MODEL_PATH = "/mnt/shared-storage-user/ma4agi-gpu/data/model/Qwen3-0.6B"
MAX_SEQ_LENGTH = 12288  # Filter samples longer than this

# Random seed
random.seed(42)

# Scoring instruction - directly from oracle_label_inplace.py
SCORING_INSTRUCTION = """## Instruction
Your task is to evaluate the quality of a solution to a problem. The problem may ask for a proof of statement, or ask for an answer. If finding an answer is required, the solution should present the answer, and it should also be a rigorous proof of that answer being valid.

Please evaluate the solution and score it according to the following criteria:
- If the solution is completely correct, with all steps executed properly and clearly demonstrated, then the score is 1.
- If the solution is generally correct, but with some details omitted or minor errors, then the score is 0.5.
- If the solution does not actually address the required problem, contains fatal errors, or has severe omissions, then the score is 0.
- Additionally, referencing anything from any paper does not save the need to prove the reference. It is okay IF AND ONLY IF the solution also presents a valid proof of the reference argument(s); otherwise, if the solution omits the proof or if the proof provided is not completely correct, the solution should be scored according to the criteria above, and definitely not with a score of 1.

Please carefully reason out and analyze the quality of the solution below, and in your final response present a detailed evaluation of the solution's quality followed by your score. Therefore, your response should be in the following format:

Here is my evaluation of the solution:
... // Your evaluation here. You are required to present in detail the key steps of the solution or the steps for which you had doubts regarding their correctness, and explicitly analyze whether each step is accurate: for correct steps, explain why you initially doubted their correctness and why they are indeed correct; for erroneous steps, explain the reason for the error and the impact of that error on the solution.

Based on my evaluation, the final overall score should be:
\\boxed{...} // where ... should be the final overall score (0, 0.5, or 1, and nothing else) based on the above criteria."""


def load_dstest(dstest_path):
    """Load dstest data, categorized by score"""
    print(f"Loading: {dstest_path}")

    data_by_score = defaultdict(list)

    with gzip.open(dstest_path, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading dstest"):
            record = json.loads(line)
            question = record['question']

            for candidate in record['candidates']:
                gt_v = candidate.get('GT_verification')
                if not gt_v or not isinstance(gt_v, dict):
                    continue

                gt_score = gt_v.get('GT')
                verification = gt_v.get('verification', '')
                proof = candidate['content']

                if gt_score is None:
                    continue

                data_by_score[gt_score].append({
                    'question': question,
                    'proof': proof,
                    'verification': verification,
                })

    print(f"\nData distribution:")
    for score in sorted(data_by_score.keys()):
        print(f"  Score {score}: {len(data_by_score[score])} samples")

    return data_by_score


def build_sft_item(question, proof, verification):
    """Build a single SFT training sample"""
    # Input format matches oracle_label_inplace.py
    input_text = f"""## Problem
{question}
## Solution
{proof}"""

    return {
        "instruction": SCORING_INSTRUCTION,
        "input": input_text,
        "output": verification
    }


def main():
    # 1. Load tokenizer
    print(f"Loading tokenizer from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 2. Load dstest data
    data_by_score = load_dstest(DSTEST_FILE)

    # 3. Build SFT data - use ALL scores (0, 0.5, 1)
    sft_data = []
    score_counts = {}
    filtered_counts = {}
    total_before = 0
    total_filtered = 0

    for score in sorted(data_by_score.keys()):
        print(f"\nProcessing GT={score} data...")
        count = 0
        filtered = 0
        for item in tqdm(data_by_score[score], desc=f"GT={score}"):
            sft_item = build_sft_item(
                item['question'],
                item['proof'],
                item['verification']
            )

            # Calculate token length
            full_text = sft_item['instruction'] + sft_item['input'] + sft_item['output']
            token_len = len(tokenizer.encode(full_text))

            if token_len <= MAX_SEQ_LENGTH:
                sft_data.append(sft_item)
                count += 1
            else:
                filtered += 1

        score_counts[score] = count
        filtered_counts[score] = filtered
        total_before += count + filtered
        total_filtered += filtered

        # Print filter stats for this score
        total_for_score = count + filtered
        filter_ratio = filtered / total_for_score * 100 if total_for_score > 0 else 0
        print(f"  GT={score}: kept {count}, filtered {filtered} ({filter_ratio:.2f}%)")

    # 4. Shuffle data
    random.shuffle(sft_data)

    # 5. Print final distribution
    print("\n" + "="*50)
    print("Final data distribution:")
    print("="*50)
    for score in sorted(score_counts.keys()):
        print(f"  Score {score}: {score_counts[score]:>5} samples")
    print(f"  Total:    {len(sft_data):>5} samples")
    print("="*50)

    # Print overall filter stats
    total_filter_ratio = total_filtered / total_before * 100 if total_before > 0 else 0
    print(f"\nFilter stats (MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}):")
    print(f"  Total before filter: {total_before}")
    print(f"  Total after filter:  {len(sft_data)}")
    print(f"  Filtered out:        {total_filtered} ({total_filter_ratio:.2f}%)")
    print("="*50)

    # 5. Save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {OUTPUT_FILE}")

    # 6. Show sample
    print("\n" + "="*60)
    print("Sample data:")
    print("="*60)
    sample = sft_data[0]
    print(f"instruction:\n{sample['instruction'][:300]}...")
    print(f"\ninput:\n{sample['input'][:500]}...")
    print(f"\noutput:\n{sample['output'][:400]}...")


if __name__ == "__main__":
    main()
