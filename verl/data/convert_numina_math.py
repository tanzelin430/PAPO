#!/usr/bin/env python3
"""
Convert NuminaMath-1.5-RL-Verifiable dataset to verl training format.

Input format (NuminaMath):
    - problem: str
    - solution: str (reference solution for process-based grading)
    - answer: str
    - problem_type, question_type, source, etc.

Output format (verl):
    - prompt: list[dict] with chat format [{'content': ..., 'role': 'user'}]
    - reward_model: dict with {'ground_truth': answer, 'style': 'rule' or 'llm_process'}
    - extra_info: dict with {'question': problem, 'reference_solution': solution}
    - data_source: 'numina_math' or 'numina_math_process'
"""

import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path


# System prompt for math problems
SYSTEM_PROMPT = "You are a knowledgeable math assistant. Answer the following questions and think step by step."


def convert_numina_to_verl(input_path: str, output_path: str, use_process_reward: bool = False):
    """Convert NuminaMath dataset to verl format.

    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file
        use_process_reward: If True, include reference_solution for LLM process-based grading
    """

    print(f"Loading NuminaMath dataset from: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} samples")
    print(f"Process reward mode: {use_process_reward}")

    # Build verl format data
    records = []
    for idx, row in df.iterrows():
        problem = row['problem'].strip()
        answer = str(row['answer']).strip()
        solution = str(row.get('solution', '')).strip() if 'solution' in row else ''

        # Create chat format prompt with system prompt + problem + instruction
        user_content = f"{SYSTEM_PROMPT} {problem} Please output the final answer within \\boxed{{}}."

        prompt = [{'content': user_content, 'role': 'user'}]

        if use_process_reward:
            # Use LLM process-based grading with reference solution
            data_source = 'numina_math_process'
            reward_model = {
                'ground_truth': answer,
                'style': 'llm_process'
            }
            extra_info = {
                'question': problem,
                'reference_solution': solution
            }
            records.append({
                'data_source': data_source,
                'prompt': prompt,
                'reward_model': reward_model,
                'extra_info': extra_info,
            })
        else:
            # Use rule-based grading (original behavior)
            reward_model = {
                'ground_truth': answer,
                'style': 'rule'
            }
            records.append({
                'data_source': 'numina_math',
                'prompt': prompt,
                'reward_model': reward_model,
            })

        if (idx + 1) % 10000 == 0:
            print(f"Processed {idx + 1}/{len(df)} samples")

    print(f"Total records: {len(records)}")

    # Create DataFrame and save
    out_df = pd.DataFrame(records)

    # Use pa.string() for schema compatibility (not pa.large_string())
    # The prompt and reward_model columns need special handling
    print(f"Saving to: {output_path}")

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    out_df.to_parquet(output_path, index=False)

    print(f"Done! Saved {len(out_df)} samples to {output_path}")

    # Verify output
    print("\n=== Verification ===")
    verify_df = pd.read_parquet(output_path)
    print(f"Samples: {len(verify_df)}")
    print(f"Columns: {list(verify_df.columns)}")
    print("\n=== First Sample ===")
    for col in verify_df.columns:
        val = verify_df.iloc[0][col]
        if isinstance(val, str) and len(val) > 200:
            val = val[:200] + '...'
        print(f"{col}: {repr(val)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NuminaMath dataset to verl format")
    parser.add_argument(
        "--input_path",
        type=str,
        default="NuminaMath-1.5-RL-Verifiable/data/train-00000-of-00001.parquet",
        help="Path to input parquet file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to output parquet file (default: auto-generated based on mode)"
    )
    parser.add_argument(
        "--process_reward",
        action="store_true",
        help="Enable LLM process-based grading with reference solutions"
    )
    args = parser.parse_args()

    # Auto-generate output path if not specified
    if args.output_path is None:
        base_dir = os.path.dirname(os.path.dirname(args.input))
        if args.process_reward:
            args.output_path = f"{base_dir}/train_verl_format_process.parquet"
        else:
            args.output_path = f"{base_dir}/train_verl_format.parquet"

    convert_numina_to_verl(args.input_path, args.output_path, args.process_reward)
