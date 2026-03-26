#!/usr/bin/env python3
"""
Convert AOPS proof problems to verl training format.

Input: aops_instruct_train.csv (~1.29M rows, messages column with user/assistant pairs)
Output: train/aops_5000.parquet (5000 sampled problems in verl format)

Reward routing: data_source='aops' → llm_proof_grading.compute_score()
"""

import ast
import pandas as pd
from pathlib import Path


SYSTEM_PROMPT = "You are a knowledgeable math assistant. Provide a complete and rigorous proof."

# Generic grading guidelines for AOPS problems (no per-problem guidelines available)
GENERIC_GUIDELINES = """(Partial) 1. The student identifies the correct approach or key technique needed for the proof but does not complete the argument.
(Partial) 2. The student proves a non-trivial intermediate result that is essential to the full proof.
(Almost) 1. The proof is essentially complete with the correct core argument, but contains a minor computational error or a small gap in justification that does not affect the overall logical structure.
(Almost) 2. The proof correctly handles all major cases but omits verification of a trivial edge case."""


def parse_messages(messages_str: str) -> tuple[str, str]:
    """Parse the messages column to extract problem (user) and solution (assistant)."""
    import re
    # CSV has }\n { between dicts (missing comma) - fix before parsing
    fixed = re.sub(r'\}\s*\{', '}, {', messages_str.replace('\n', ' '))
    messages = ast.literal_eval(fixed)
    problem = ""
    solution = ""
    for msg in messages:
        if msg["role"] == "user":
            problem = msg["content"].strip()
        elif msg["role"] == "assistant":
            solution = msg["content"].strip()
    return problem, solution


def convert_aops(input_path: str, output_path: str, n_samples: int = 5000, seed: int = 42):
    """Convert AOPS CSV to verl parquet format."""

    print(f"Loading AOPS data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows")

    # Sample
    print(f"Sampling {n_samples} rows (seed={seed})")
    df = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)

    records = []
    skipped = 0
    for idx, row in df.iterrows():
        try:
            problem, solution = parse_messages(row["messages"])
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  Skip row {idx}: {e}")
            continue

        if not problem or not solution:
            skipped += 1
            continue

        user_content = f"{SYSTEM_PROMPT}\n\n{problem}"

        records.append({
            "data_source": "aops",
            "prompt": [{"content": user_content, "role": "user"}],
            "reward_model": {"ground_truth": "", "style": "llm_proof"},
            "extra_info": {
                "question": problem,
                "reference_solution": solution,
                "grading_guidelines": GENERIC_GUIDELINES,
            },
        })

    print(f"Converted {len(records)} samples ({skipped} skipped)")

    out_df = pd.DataFrame(records)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")

    # Verify
    print("\n=== Verification ===")
    verify_df = pd.read_parquet(output_path)
    print(f"Samples: {len(verify_df)}")
    print(f"Columns: {list(verify_df.columns)}")
    print(f"\n=== First Sample ===")
    for col in verify_df.columns:
        val = verify_df.iloc[0][col]
        val_str = repr(val)
        if len(val_str) > 300:
            val_str = val_str[:300] + "..."
        print(f"{col}: {val_str}")


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    input_path = "aops_instruct_train.csv"  # Provide your own path
    output_path = str(script_dir / "train" / "aops_5000.parquet")

    convert_aops(input_path, output_path)
