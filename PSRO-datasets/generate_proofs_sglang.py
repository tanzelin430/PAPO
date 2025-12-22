"""
Proof generation using SGLang with local GPU inference.

Usage:
1. Start SGLang server on GPU node:
   python -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 30000

2. Run this script:
   python generate_proofs_sglang.py

Or use offline mode (single GPU):
   python generate_proofs_sglang.py --offline
"""

import json
import time
import os
import argparse
from typing import List, Dict, Any, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ============== Configuration ==============
MODEL_PATH = "/mnt/shared-storage-user/ma4agi-gpu/data/model/Qwen3-8B"  # Local model path
SGLANG_SERVER_URL = "http://localhost:30000"  # SGLang server URL

N_CANDIDATES = 1
MAX_PROBLEMS = 100  # Set to desired number, or -1 for all problems

OUTPUT_JSONL = "test0.jsonl"
RESUME = True
CONCURRENT_WORKERS = 5  # For server mode

# Thread lock for writing to file
file_lock = threading.Lock()

PROMPT_TEMPLATE = """
Your task is to solve a given problem. The problem may ask you to prove a statement, or ask for an answer. If finding an answer is required, you should come up with the answer, and your final solution should also be a rigorous proof of that answer being valid.

Your final solution to the problem should be exceptionally comprehensive and easy-to-follow. A good solution should satisfy the following criteria:

- The solution should be completely correct, with all steps executed properly and clearly demonstrated.
- The proof must be rigorous. Every step must be logically justified and clearly explained.
- Additionally, referencing anything from any paper does not save the need to prove the reference. It is okay IF AND ONLY IF the solution also presents a valid proof of the reference argument(s); otherwise, if the solution omits the proof or if the proof provided is not completely correct, the solution is incomplete.

Important: You must be honest and rigorous in your proof. Do not fabricate steps, skip crucial reasoning, or claim something is true without proper justification. Remember! You CAN'T cheat! If you cheat, we will know, and you will be penalized!

Your final response should be in the following format:

## Solution
... // Your final solution to the problem here.

---
Here is your task input:

## Problem
{question}

""".strip()


def build_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question)


def load_completed_indices(output_path: str) -> Set[int]:
    """Load indices of already completed problems from output file."""
    completed = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    completed.add(record.get("index", -1))
                except json.JSONDecodeError:
                    continue
    return completed


def get_aops_questions(max_problems: int) -> List[str]:
    """Load questions from local CSV or HuggingFace."""
    csv_path = "aops_instruct_train.csv"

    if os.path.exists(csv_path):
        print(f"Loading questions from local CSV: {csv_path}")
        import csv
        import re

        questions = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    messages_str = row.get('messages', '')
                    if not messages_str:
                        continue
                    # Extract user content using regex
                    match = re.search(r"'content': '(.*?)', 'role': 'user'", messages_str, re.DOTALL)
                    if match:
                        q = match.group(1).strip()
                        if q:
                            questions.append(q)
                            if max_problems > 0 and len(questions) >= max_problems:
                                break
                except Exception as e:
                    continue
    else:
        print("Loading AoPS-Instruct dataset from HuggingFace...")
        from datasets import load_dataset
        ds = load_dataset("DeepStudentLlama/AoPS-Instruct", split="train")

        questions = []
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


# ============== Server Mode (Recommended) ==============
def call_sglang_server(prompt: str, n_candidates: int = 1) -> List[str]:
    """Call SGLang server for inference."""
    import requests

    url = f"{SGLANG_SERVER_URL}/v1/chat/completions"

    body = {
        "model": MODEL_PATH,
        "messages": [{"role": "user", "content": prompt}],
        "n": n_candidates,
        "temperature": 0.7,
        "top_p": 0.8,
        "max_tokens": 8192,
    }

    resp = requests.post(url, json=body, timeout=300)
    if resp.status_code != 200:
        raise RuntimeError(f"SGLang server error: {resp.status_code}, {resp.text[:500]}")

    data = resp.json()
    contents = []
    for choice in data.get("choices", []):
        content = choice.get("message", {}).get("content", "")
        contents.append(content)

    return contents


def process_single_problem_server(idx: int, question: str, total: int, fout) -> bool:
    """Process a single problem using SGLang server."""
    print(f"[{idx+1}/{total}] Generating candidate proofs...")

    prompt = build_prompt(question)

    for attempt in range(5):
        try:
            candidates = call_sglang_server(prompt, n_candidates=N_CANDIDATES)
            break
        except Exception as e:
            print(f"  [{idx+1}] API call failed, retry {attempt+1}: {e}")
            time.sleep(5)
    else:
        print(f"  [{idx+1}] All retries failed, skipping this problem.")
        return False

    record = {
        "index": idx,
        "question": question,
        "n_candidates": len(candidates),
        "candidates": [
            {"candidate_index": i, "content": cand}
            for i, cand in enumerate(candidates)
        ],
    }

    with file_lock:
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()

    print(f"[{idx+1}/{total}] Done.")
    return True


def run_server_mode(questions: List[str], completed_indices: Set[int]):
    """Run using SGLang server (recommended for multi-GPU)."""
    pending_tasks = [(idx, q) for idx, q in enumerate(questions) if idx not in completed_indices]

    if not pending_tasks:
        print("All problems already completed!")
        return

    print(f"Processing {len(pending_tasks)} problems with {CONCURRENT_WORKERS} concurrent workers...")
    print(f"Using SGLang server at: {SGLANG_SERVER_URL}")

    mode = "a" if RESUME and completed_indices else "w"

    with open(OUTPUT_JSONL, mode, encoding="utf-8") as fout:
        if CONCURRENT_WORKERS <= 1:
            for idx, q in pending_tasks:
                process_single_problem_server(idx, q, len(questions), fout)
        else:
            with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
                futures = {
                    executor.submit(process_single_problem_server, idx, q, len(questions), fout): idx
                    for idx, q in pending_tasks
                }

                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    if completed % 10 == 0:
                        print(f"Progress: {completed}/{len(pending_tasks)} tasks completed")

    print(f"All done! Results saved to {OUTPUT_JSONL}")


# ============== Offline Mode (Multi-GPU supported) ==============
def run_offline_mode(questions: List[str], completed_indices: Set[int], tp_size: int = 1, dp_size: int = 1):
    """
    Run using SGLang offline inference with multi-GPU support.

    Args:
        tp_size: Tensor parallelism size (split model across GPUs for large models)
        dp_size: Data parallelism size (run multiple copies for throughput)
    """
    import sglang as sgl

    pending_tasks = [(idx, q) for idx, q in enumerate(questions) if idx not in completed_indices]

    if not pending_tasks:
        print("All problems already completed!")
        return

    print(f"Processing {len(pending_tasks)} problems in offline mode...")
    print(f"Loading model: {MODEL_PATH}")
    print(f"Tensor Parallelism: {tp_size}, Data Parallelism: {dp_size}")
    print(f"Total GPUs used: {tp_size * dp_size}")

    # Initialize SGLang engine with multi-GPU support
    llm = sgl.Engine(
        model_path=MODEL_PATH,
        tp_size=tp_size,  # Tensor parallelism: split model across GPUs
        dp_size=dp_size,  # Data parallelism: multiple model copies
    )

    mode = "a" if RESUME and completed_indices else "w"

    # Prepare all prompts for batch processing
    prompts = [build_prompt(q) for _, q in pending_tasks]
    indices = [idx for idx, _ in pending_tasks]

    print(f"Running batch inference on {len(prompts)} prompts...")

    sampling_params = {"temperature": 0.7, "top_p": 0.8, "max_new_tokens": 8192}

    # Batch inference - much faster than one-by-one
    outputs = llm.generate(prompts, sampling_params)

    # Write results
    with open(OUTPUT_JSONL, mode, encoding="utf-8") as fout:
        for i, (idx, question) in enumerate(pending_tasks):
            try:
                content = outputs[i]["text"]

                record = {
                    "index": idx,
                    "question": question,
                    "n_candidates": 1,
                    "candidates": [{"candidate_index": 0, "content": content}],
                }

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()

                if (i + 1) % 10 == 0:
                    print(f"Progress: {i+1}/{len(pending_tasks)} completed")

            except Exception as e:
                print(f"  [{idx+1}] Error: {e}")
                continue

    llm.shutdown()
    print(f"All done! Results saved to {OUTPUT_JSONL}")


def main():
    global SGLANG_SERVER_URL, MODEL_PATH, MAX_PROBLEMS, CONCURRENT_WORKERS

    parser = argparse.ArgumentParser(description="Generate proofs using SGLang")
    parser.add_argument("--offline", action="store_true", help="Use offline mode (load model locally)")
    parser.add_argument("--server-url", type=str, default=SGLANG_SERVER_URL, help="SGLang server URL")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Model path")
    parser.add_argument("--max-problems", type=int, default=MAX_PROBLEMS, help="Max problems to process")
    parser.add_argument("--workers", type=int, default=CONCURRENT_WORKERS, help="Concurrent workers")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size (split model across GPUs)")
    parser.add_argument("--dp", type=int, default=1, help="Data parallelism size (multiple model copies)")
    args = parser.parse_args()

    SGLANG_SERVER_URL = args.server_url
    MODEL_PATH = args.model
    MAX_PROBLEMS = args.max_problems
    CONCURRENT_WORKERS = args.workers

    questions = get_aops_questions(MAX_PROBLEMS)

    completed_indices: Set[int] = set()
    if RESUME and os.path.exists(OUTPUT_JSONL):
        completed_indices = load_completed_indices(OUTPUT_JSONL)
        print(f"Resuming: found {len(completed_indices)} already completed problems.")

    if args.offline:
        run_offline_mode(questions, completed_indices, tp_size=args.tp, dp_size=args.dp)
    else:
        run_server_mode(questions, completed_indices)


if __name__ == "__main__":
    main()
