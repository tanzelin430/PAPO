#!/usr/bin/env python3
"""
Convert IMO-ProofBench (proofbench.csv) to verl evaluation format.

Input: proofbench.csv from google-deepmind/superhuman repo (60 IMO proof problems)
Output: eval/proofbench.parquet in verl format

Columns:
    - data_source: 'proofbench'
    - prompt: [{"role": "user", "content": problem_text}]
    - reward_model: {"ground_truth": short_answer, "style": "llm_proof"}
    - extra_info: {question, reference_solution, grading_guidelines, problem_id, category, level}
"""

import pandas as pd
from pathlib import Path


SYSTEM_PROMPT = "You are a knowledgeable math assistant. Provide a complete and rigorous proof."


def convert_proofbench(input_path: str, output_path: str):
    """Convert proofbench.csv to verl parquet format."""

    print(f"Loading proofbench from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} problems")
    print(f"Columns: {list(df.columns)}")
    print(f"Categories: {df['Category'].value_counts().to_dict()}")
    print(f"Levels: {df['Level'].value_counts().to_dict()}")

    records = []
    for _, row in df.iterrows():
        problem = str(row['Problem']).strip()
        solution = str(row['Solution']).strip()
        grading_guidelines = str(row['Grading guidelines']).strip()
        problem_id = str(row['Problem ID']).strip()
        category = str(row['Category']).strip()
        level = str(row['Level']).strip()
        short_answer = str(row.get('Short Answer', '')).strip()
        if short_answer == 'nan':
            short_answer = ''

        user_content = f"{SYSTEM_PROMPT}\n\n{problem}"

        prompt = [{'content': user_content, 'role': 'user'}]

        reward_model = {
            'ground_truth': short_answer,
            'style': 'llm_proof',
        }

        extra_info = {
            'question': problem,
            'reference_solution': solution,
            'grading_guidelines': grading_guidelines,
            'problem_id': problem_id,
            'category': category,
            'level': level,
        }

        records.append({
            'data_source': 'proofbench',
            'prompt': prompt,
            'reward_model': reward_model,
            'extra_info': extra_info,
        })

    out_df = pd.DataFrame(records)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)

    print(f"\nSaved {len(out_df)} samples to {output_path}")

    # Verify
    print("\n=== Verification ===")
    verify_df = pd.read_parquet(output_path)
    print(f"Samples: {len(verify_df)}")
    print(f"Columns: {list(verify_df.columns)}")
    print(f"\n=== First Sample ===")
    for col in verify_df.columns:
        val = verify_df.iloc[0][col]
        val_str = repr(val)
        if len(val_str) > 200:
            val_str = val_str[:200] + '...'
        print(f"{col}: {val_str}")


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    input_path = script_dir / "proofbench.csv"
    output_path = script_dir / "eval" / "proofbench.parquet"

    convert_proofbench(str(input_path), str(output_path))
