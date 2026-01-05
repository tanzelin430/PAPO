"""
Offline batch proof generation using vLLM.

This script uses vLLM's offline inference API for high-throughput batch
generation of mathematical proofs using Qwen3-4B-Instruct-2507.

Usage:
    python generate_proofs_vllm_offline.py
"""

import json
import os
from typing import List, Dict, Any

from vllm import LLM, SamplingParams
from datasets import load_dataset


# =============================================================================
# Configuration
# =============================================================================

# Use SFT-trained proof generation model
MODEL_NAME = "/mnt/shared-storage-user/tanzelin-p/proof_gen_ckpt/sft_train-0/checkpoint-30/sft_train"

# Generation settings
N_CANDIDATES = 1           # Number of candidate proofs per problem
START_INDEX = 10000        # Skip first 10k (used for SFT training)
END_INDEX = 100000          # Generate for indices 10000-60000
TEMPERATURE = 0.5          # Lower temperature for more deterministic output
TOP_P = 0.8
MAX_TOKENS = 8192          # Standard proofs are <4096 tokens
REPETITION_PENALTY = 1.1   # Slightly lower since SFT model is better behaved

# vLLM engine settings
TENSOR_PARALLEL_SIZE = 4   # Number of GPUs for tensor parallelism
GPU_MEMORY_UTILIZATION = 0.8
SEED = 42

# Output settings
OUTPUT_JSONL = "/mnt/shared-storage-user/tanzelin-p/sft_proofs_10k_100k.jsonl"
RESUME_FROM = 0            # Resume offset within the selected range

# =============================================================================
# Prompt Template
# =============================================================================

# Match the SFT training instruction format
INSTRUCTION = """Your task is to solve a given problem. The problem may ask you to prove a statement, or ask for an answer. If finding an answer is required, you should come up with the answer, and your final solution should also be a rigorous proof of that answer being valid.

Your final solution to the problem should be exceptionally comprehensive and easy-to-follow. A good solution should satisfy the following criteria:

- The solution should be completely correct, with all steps executed properly and clearly demonstrated.
- The proof must be rigorous. Every step must be logically justified and clearly explained.
- Additionally, referencing anything from any paper does not save the need to prove the reference. It is okay IF AND ONLY IF the solution also presents a valid proof of the reference argument(s); otherwise, if the solution omits the proof or if the proof provided is not completely correct, the solution is incomplete."""


def build_prompt(question: str, tokenizer) -> str:
    """Build the full prompt using chat template."""
    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": question}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def get_aops_questions(max_problems: int) -> List[str]:
    """
    Load the AoPS-Instruct train split from Hugging Face and extract
    the first `max_problems` user questions.
    """
    print("Loading AoPS-Instruct dataset (train split)...")
    ds = load_dataset("DeepStudentLlama/AoPS-Instruct", split="train")

    questions: List[str] = []

    for ex in ds:
        messages = ex.get("messages", [])
        if not messages:
            continue
        user_msg = messages[0]
        q = user_msg.get("content", "").strip()
        if not q:
            continue
        questions.append(q)
        if max_problems > 0 and len(questions) >= max_problems:
            break

    print(f"Collected {len(questions)} problems for candidate proof generation.")
    return questions


def load_questions_from_csv(csv_path: str, start_idx: int, end_idx: int) -> List[str]:
    """Load questions from CSV file for specified index range."""
    import pandas as pd
    import ast
    import re

    print(f"Loading questions from {csv_path} (rows {start_idx} to {end_idx})...")
    df = pd.read_csv(csv_path)

    # Select the specified range
    df_subset = df.iloc[start_idx:end_idx]
    print(f"Processing {len(df_subset)} rows...")

    questions: List[str] = []
    for _, row in df_subset.iterrows():
        try:
            raw = row["messages"]
            # Fix format: "}\n {" -> "}, {" (missing comma between list elements)
            fixed = re.sub(r'\}\s*\n\s*\{', '}, {', raw)
            # Parse the messages column (contains JSON-like list of dicts)
            messages = ast.literal_eval(fixed)
            if messages and len(messages) > 0:
                # First message is the user's question
                user_msg = messages[0]
                q = user_msg.get("content", "").strip()
                if q:
                    questions.append(q)
        except (ValueError, SyntaxError, KeyError) as e:
            continue

    print(f"Loaded {len(questions)} problems from CSV.")
    return questions


def initialize_llm() -> LLM:
    """Initialize the vLLM engine with Qwen3-4B-Instruct-2507."""
    print(f"Initializing vLLM with model: {MODEL_NAME}")
    print(f"  Tensor parallel size: {TENSOR_PARALLEL_SIZE}")
    print(f"  GPU memory utilization: {GPU_MEMORY_UTILIZATION}")

    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        seed=SEED,
        trust_remote_code=True,
        max_model_len=MAX_TOKENS,  # Extra room for prompt
        dtype="auto",
    )

    print("vLLM engine initialized successfully.")
    return llm


def create_sampling_params() -> SamplingParams:
    """Create sampling parameters for generation."""
    return SamplingParams(
        n=N_CANDIDATES,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        repetition_penalty=REPETITION_PENALTY,
        seed=SEED,
    )


def generate_proofs_streaming(
    llm: LLM,
    questions: List[str],
    sampling_params: SamplingParams,
    tokenizer,
    output_path: str,
    start_index: int = 0,
):
    """
    Generate proofs and write each result as soon as it completes.

    Submits all requests to vLLM at once, letting the engine handle
    continuous batching internally. Writes results incrementally.
    """
    from tqdm import tqdm

    total = len(questions)
    print(f"Submitting {total} prompts to vLLM engine...")
    print(f"Starting from index {start_index}, results will be appended to {output_path}")

    # Build all prompts using chat template
    prompts = [build_prompt(q, tokenizer) for q in questions]

    # Create request ID -> (index, question) mapping
    request_map: Dict[str, tuple] = {}

    # Add all requests to the engine
    for i, prompt in enumerate(prompts):
        request_id = f"req_{start_index + i}"
        global_idx = start_index + i
        request_map[request_id] = (global_idx, questions[i])
        llm.llm_engine.add_request(
            request_id=request_id,
            prompt=prompt,
            params=sampling_params,
        )

    print(f"All {total} requests submitted. Processing...")

    # Process and write results as they complete
    completed = 0
    pbar = tqdm(total=total, desc="Generating proofs")

    with open(output_path, "a", encoding="utf-8") as f:
        while llm.llm_engine.has_unfinished_requests():
            # Step the engine
            step_outputs = llm.llm_engine.step()

            # Process completed requests
            for output in step_outputs:
                if output.finished:
                    request_id = output.request_id
                    global_idx, question = request_map[request_id]

                    candidates = []
                    for i, out in enumerate(output.outputs):
                        candidates.append({
                            "candidate_index": i,
                            "content": out.text,
                        })

                    record = {
                        "index": global_idx,
                        "question": question,
                        "n_candidates": len(candidates),
                        "candidates": candidates,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()  # Ensure immediate write to disk

                    completed += 1
                    pbar.update(1)

    pbar.close()
    print(f"\nAll done! {completed} records appended to {output_path}")


def main():
    from transformers import AutoTokenizer

    # Load tokenizer for chat template
    print(f"Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load questions from specified range
    csv_path = "aops_instruct_train.csv"
    if os.path.exists(csv_path):
        questions = load_questions_from_csv(csv_path, START_INDEX, END_INDEX)
    else:
        print("CSV file not found!")
        return

    if not questions:
        print("No questions found!")
        return

    # Skip already completed questions if resuming
    if RESUME_FROM > 0:
        print(f"Resuming from offset {RESUME_FROM}, skipping first {RESUME_FROM} questions...")
        questions = questions[RESUME_FROM:]
        if not questions:
            print("No remaining questions to process!")
            return

    # Initialize vLLM
    llm = initialize_llm()
    sampling_params = create_sampling_params()

    # Generate proofs - write each result as it completes
    generate_proofs_streaming(
        llm, questions, sampling_params, tokenizer,
        output_path=OUTPUT_JSONL,
        start_index=START_INDEX + RESUME_FROM,
    )


if __name__ == "__main__":
    main()
