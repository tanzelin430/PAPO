#!/usr/bin/env python3
"""
Convert ROLL data format to verl parquet format.

ROLL format (jsonl):
{
  "id": "0",
  "domain": "math_rule",
  "messages": [{"role": "user", "content": "..."}],
  "ground_truth": "50",
  "tag": "math_curriculum"
}

verl format (parquet):
{
  "data_source": "math_dapo",  # Use math_dapo for verl's default reward routing
  "prompt": [{"role": "user", "content": "..."}],
  "ability": "math",
  "reward_model": {"style": "rule", "ground_truth": "50"},
  "extra_info": {"split": "train", "index": 0, "id": "0", "original_tag": "math_curriculum"}
}

IMPORTANT: verl's default_compute_score routes by data_source.
- "math_dapo", "math", or "aime*" -> math_dapo.compute_score (uses "Answer:" pattern)
- ROLL uses "ANSWER:\n\\boxed{}" format which is compatible with math_dapo

Tag mapping (ROLL -> verl):
- math_curriculum, math_curriculum_test -> math_dapo
- aime2024, aime2025 -> aime2024/aime2025 (kept as-is, matches "aime*")
- math500 -> math_dapo
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

# Map ROLL tags to verl data_source values
TAG_TO_DATA_SOURCE = {
    "math_curriculum": "math_dapo",
    "math_curriculum_test": "math_dapo",
    "math500": "math_dapo",
    "math_rule": "math_dapo",
    # AIME tags are kept as-is since verl matches "aime*"
    "aime2024": "aime2024",
    "aime2025": "aime2025",
}


def convert_roll_to_verl(input_path: str, output_path: str, split: str = "train"):
    """Convert ROLL jsonl to verl parquet format."""

    records = []
    with open(input_path, "r") as f:
        for idx, line in enumerate(f):
            item = json.loads(line.strip())

            # Extract fields from ROLL format
            messages = item.get("messages", [])
            ground_truth = item.get("ground_truth", "")
            original_tag = item.get("tag", item.get("domain", "math"))
            item_id = item.get("id", str(idx))

            # Map tag to verl data_source
            data_source = TAG_TO_DATA_SOURCE.get(original_tag, "math_dapo")

            # Convert to verl format
            record = {
                "data_source": data_source,
                "prompt": messages,  # verl uses "prompt" for messages
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": str(ground_truth)},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "id": item_id,
                    "original_tag": original_tag,
                },
            }
            records.append(record)

    # Convert to DataFrame and save as parquet
    df = pd.DataFrame(records)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_parquet(output_path, index=False)
    print(f"Converted {len(records)} samples to {output_path}")

    # Print example
    print("\nExample record:")
    print(json.dumps(records[0], indent=2, ensure_ascii=False)[:800])

    return len(records)


def main():
    parser = argparse.ArgumentParser(description="Convert ROLL data to verl format")
    parser.add_argument("--input", type=str, required=True, help="Input ROLL jsonl file")
    parser.add_argument("--output", type=str, required=True, help="Output verl parquet file")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "val"])
    args = parser.parse_args()

    convert_roll_to_verl(args.input, args.output, args.split)


if __name__ == "__main__":
    main()
