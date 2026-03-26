#!/usr/bin/env python3

"""
Convert JSONL evaluation benchmarks to verl parquet format.
Adds: AIME25, Olympiad, Minerva to the eval directory.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path


def convert_jsonl_to_parquet(
    jsonl_path: str,
    output_path: str,
    data_source: str,
    ability: str = "math",
    system_prompt: str = "You are a knowledgeable math assistant. Answer the following questions and think step by step."
):
    """
    Convert JSONL file to verl parquet format.

    JSONL format (input):
    {"messages": [{"role": "user", "content": "..."}], "ground_truth": "...", "tag": "...", "domain": "..."}

    Parquet format (output):
    - data_source: str
    - prompt: array of dicts (messages with system prompt prepended)
    - ability: str
    - reward_model: dict with ground_truth and style
    - extra_info: dict with index and split
    """
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            data = json.loads(line.strip())

            # Extract user message content
            user_content = data['messages'][0]['content']

            # Add instruction to put answer in \boxed{}
            if "\\boxed{}" not in user_content:
                user_content = user_content + " Please solve step by step and put your final answer within \\boxed{}."

            # Create prompt with system instruction
            prompt = np.array([{
                'content': f"{system_prompt} {user_content}",
                'role': 'user'
            }], dtype=object)

            # Handle ground_truth - could be string, list, or stringified list
            ground_truth = data.get('ground_truth', '')
            if isinstance(ground_truth, list):
                ground_truth = ground_truth[0] if ground_truth else ''
            elif isinstance(ground_truth, str):
                # Handle stringified lists like "['2']" or "['$\\frac{1}{2 n+2}$']"
                gt_stripped = ground_truth.strip()
                if gt_stripped.startswith('[') and gt_stripped.endswith(']'):
                    try:
                        import ast
                        parsed = ast.literal_eval(gt_stripped)
                        if isinstance(parsed, list) and parsed:
                            ground_truth = parsed[0]
                    except:
                        pass  # Keep original if parsing fails

            record = {
                'data_source': data_source,
                'prompt': prompt,
                'ability': ability,
                'reward_model': {
                    'ground_truth': str(ground_truth),
                    'style': 'rule'
                },
                'extra_info': {
                    'index': idx,
                    'split': 'val'
                }
            }
            records.append(record)

    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    print(f"Converted {len(records)} samples: {jsonl_path} -> {output_path}")
    return len(records)


def main():
    eval_benchmarks_dir = Path(os.environ.get("EVAL_BENCHMARKS_DIR", "eval_benchmarks"))
    output_dir = Path(os.path.join(os.path.dirname(__file__), "..", "data", "eval"))

    # Define benchmarks to convert
    benchmarks = [
        # (jsonl_file, output_name, data_source)
        ("aime2025_roll.jsonl", "aime2025.parquet", "aime2025"),
        ("olympiad_roll.jsonl", "olympiad.parquet", "olympiad"),
        ("minerva_roll.jsonl", "minerva.parquet", "minerva"),
    ]

    total = 0
    for jsonl_file, output_name, data_source in benchmarks:
        jsonl_path = eval_benchmarks_dir / jsonl_file
        output_path = output_dir / output_name

        if not jsonl_path.exists():
            print(f"Warning: {jsonl_path} not found, skipping")
            continue

        count = convert_jsonl_to_parquet(
            str(jsonl_path),
            str(output_path),
            data_source
        )
        total += count

    print(f"\nTotal: {total} samples converted")
    print(f"\nOutput directory: {output_dir}")
    print("\nFiles created:")
    for _, output_name, _ in benchmarks:
        output_path = output_dir / output_name
        if output_path.exists():
            print(f"  - {output_name}")


if __name__ == "__main__":
    main()
