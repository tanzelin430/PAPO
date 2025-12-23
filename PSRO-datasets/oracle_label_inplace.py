
import json
import time
import re
from typing import Dict, Any, Generator, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import requests

INPUT_JSONL = "test0.jsonl"
OUTPUT_JSONL = "test1.jsonl"

API_BASE = "http://35.220.164.252:3888/v1/"
API_KEY = "sk-7LyupxC6GN7FPr2KLGO5nprvJpyKG6kRcqesJeLpUqzfeDiq"

VERIFIER_MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"

TEMPERATURE = 0.0
MAX_TOKENS = 8192
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 5
NUM_WORKERS = 16  # 并发线程数
MAX_RECORDS = 10  # 处理前N条记录

CHAT_COMPLETIONS_PATH = "/chat/completions"


PROMPT_VERIFICATION = """
## Instruction
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
\\boxed{{...}} // where ... should be the final overall score (0, 0.5, or 1, and nothing else) based on the above criteria.

---
Here is your task input:

## Problem
{question}
## Solution
{proof}
""".strip()


def build_verification_prompt(question: str, proof: str) -> str:
    return PROMPT_VERIFICATION.format(question=question, proof=proof)


def call_verifier_api(question: str, proof: str) -> Dict[str, Any]:
    url = API_BASE.rstrip("/") + CHAT_COMPLETIONS_PATH
    prompt = build_verification_prompt(question, proof)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    body = {
        "model": VERIFIER_MODEL_NAME,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "n": 1,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }

    resp = requests.post(url, headers=headers, json=body, timeout=120)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Verifier API call failed, status={resp.status_code}, body={resp.text[:500]}"
        )

    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("Verifier response does not contain 'choices' field")

    content = choices[0]["message"]["content"]

    score = extract_score_from_text(content)
    return {
        "GT": score,
        "verification": content,
    }


def extract_score_from_text(text: str) -> float:
    m = re.search(r"\\boxed\{\s*(0(?:\.0)?|0\.5|1(?:\.0)?)\s*\}", text)
    if m:
        s = m.group(1)
        try:
            return float(s)
        except ValueError:
            pass

    candidates = []
    for s in ["0.5", "1", "0"]:
        idx = text.rfind(s)
        if idx != -1:
            candidates.append((idx, s))
    if candidates:
        candidates.sort()
        s = candidates[-1][1]
        return float(s)

    raise RuntimeError(f"Failed to parse score from text: {text}")


def iter_jsonl(path: str) -> Generator[Dict[str, Any], None, None]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def verify_single_candidate(task: Tuple[int, int, str, str]) -> Tuple[int, int, Optional[Dict[str, Any]]]:
    """验证单个candidate，返回 (record_idx, candidate_idx, result)"""
    record_idx, cand_idx, question, proof = task

    GT_result: Optional[Dict[str, Any]] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            GT_result = call_verifier_api(question, proof)
            break
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS)
            else:
                GT_result = None

    return (record_idx, cand_idx, GT_result)


# 用于线程安全的打印和写入
write_lock = threading.Lock()

def main():
    print(f"Reading generated dataset: {INPUT_JSONL}")
    records = list(iter_jsonl(INPUT_JSONL))[:MAX_RECORDS]
    print(f"Loaded {len(records)} problem records.")

    # 统计每个record有多少candidate
    record_candidate_counts = {}
    for ridx, rec in enumerate(records):
        record_candidate_counts[ridx] = len(rec.get("candidates", []))

    # 构建所有任务
    tasks: List[Tuple[int, int, str, str]] = []
    for ridx, rec in enumerate(records):
        question = rec.get("question", "")
        candidates = rec.get("candidates", [])
        for cand in candidates:
            cand_idx = cand.get("candidate_index")
            proof = cand.get("content", "")
            tasks.append((ridx, cand_idx, question, proof))

    total_tasks = len(tasks)
    print(f"Total verification tasks: {total_tasks}, using {NUM_WORKERS} workers")

    # 存储结果的字典: {(record_idx, cand_idx): result}
    results: Dict[Tuple[int, int], Optional[Dict[str, Any]]] = {}
    # 追踪每个record完成了多少candidate
    record_completed: Dict[int, int] = {i: 0 for i in range(len(records))}
    # 追踪哪些record已经写入
    record_written: set = set()
    completed = 0
    records_written_count = 0

    # 打开输出文件
    out_f = open(OUTPUT_JSONL, "w", encoding="utf-8")

    try:
        # 多线程执行
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            future_to_task = {executor.submit(verify_single_candidate, task): task for task in tasks}

            for future in as_completed(future_to_task):
                record_idx, cand_idx, result = future.result()

                with write_lock:
                    results[(record_idx, cand_idx)] = result
                    record_completed[record_idx] += 1
                    completed += 1

                    score = result.get("GT", "N/A") if result else "FAILED"
                    print(f"[{completed}/{total_tasks}] record={record_idx}, candidate={cand_idx}, score={score}")

                    # 这个record的所有candidate都完成了且还没写入，立即写入
                    if record_idx not in record_written and \
                       record_completed[record_idx] == record_candidate_counts[record_idx]:
                        rec = records[record_idx]
                        candidates = rec.get("candidates", [])
                        for cand in candidates:
                            c_idx = cand.get("candidate_index")
                            cand["GT_verification"] = results.get((record_idx, c_idx))

                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        out_f.flush()
                        record_written.add(record_idx)
                        records_written_count += 1
                        print(f"  >> Record {record_idx} written to file ({records_written_count}/{len(records)})")

    finally:
        out_f.close()

    print(f"Done! New dataset written to: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
