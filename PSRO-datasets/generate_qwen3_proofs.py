import json
import time
import os
from typing import List, Dict, Any, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import requests
from datasets import load_dataset


API_BASE = "http://35.220.164.252:3888/v1/"
CHAT_COMPLETIONS_PATH = "/chat/completions"

API_KEY = "sk-7LyupxC6GN7FPr2KLGO5nprvJpyKG6kRcqesJeLpUqzfeDiq"

MODEL_NAME = "Qwen/Qwen3-8B"

N_CANDIDATES = 1
MAX_PROBLEMS = 2000  # Set to desired number, or -1 for all problems

OUTPUT_JSONL = "test0.jsonl"
RESUME = False  # Set to True to resume from breakpoint
CONCURRENT_WORKERS = 8  # Number of parallel requests (set to 1 for sequential)

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


def call_qwen_chat(
    prompt: str,
    n_candidates: int = 8,
    temperature: float = 0.7,
    top_p: float = 0.8,
    max_tokens: int = 8192,
    extra_headers: Optional[Dict[str, str]] = None,
) -> List[str]:
    url = API_BASE.rstrip("/") + CHAT_COMPLETIONS_PATH

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    body = {
        "model": MODEL_NAME,
        "n": n_candidates,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }

    resp = requests.post(url, headers=headers, json=body, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(
            f"API call failed, status={resp.status_code}, body={resp.text[:500]}"
        )

    data = resp.json()
    choices = data.get("choices", [])
    contents: List[str] = []
    for choice in choices:
        msg = choice.get("message", {})
        content = msg.get("content", "")
        contents.append(content)

    return contents


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


def process_single_problem(idx: int, question: str, total: int, fout) -> bool:
    """Process a single problem and write result to file. Returns True if successful."""
    print(f"[{idx+1}/{total}] Generating candidate proofs...")

    prompt = build_prompt(question)

    for attempt in range(5):
        try:
            candidates = call_qwen_chat(
                prompt,
                n_candidates=N_CANDIDATES,
                temperature=0.7,
                top_p=0.8,
                max_tokens=8192,
            )
            break
        except Exception as e:
            print(f"  [{idx+1}] API call failed, retry {attempt+1}: {e}")
            time.sleep(5)
    else:
        print(f"  [{idx+1}] All retries failed, skipping this problem.")
        return False

    record: Dict[str, Any] = {
        "index": idx,
        "question": question,
        "n_candidates": len(candidates),
        "candidates": [
            {
                "candidate_index": i,
                "content": cand,
            }
            for i, cand in enumerate(candidates)
        ],
    }

    with file_lock:
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()

    print(f"[{idx+1}/{total}] Done.")
    return True


def main():
    questions = get_aops_questions(MAX_PROBLEMS)

    # Load completed indices if resuming
    completed_indices: Set[int] = set()
    if RESUME and os.path.exists(OUTPUT_JSONL):
        completed_indices = load_completed_indices(OUTPUT_JSONL)
        print(f"Resuming: found {len(completed_indices)} already completed problems.")

    # Filter out completed problems
    pending_tasks = [(idx, q) for idx, q in enumerate(questions) if idx not in completed_indices]

    if not pending_tasks:
        print("All problems already completed!")
        return

    print(f"Processing {len(pending_tasks)} problems with {CONCURRENT_WORKERS} concurrent workers...")

    # Open file in append mode if resuming, otherwise write mode
    mode = "a" if RESUME and completed_indices else "w"

    with open(OUTPUT_JSONL, mode, encoding="utf-8") as fout:
        if CONCURRENT_WORKERS <= 1:
            # Sequential processing
            for idx, q in pending_tasks:
                process_single_problem(idx, q, len(questions), fout)
        else:
            # Concurrent processing
            with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
                futures = {
                    executor.submit(process_single_problem, idx, q, len(questions), fout): idx
                    for idx, q in pending_tasks
                }

                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    if completed % 10 == 0:
                        print(f"Progress: {completed}/{len(pending_tasks)} tasks completed")

    print(f"All done! Results saved to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
