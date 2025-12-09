import json
import time
from typing import List, Dict, Any, Optional

import requests
from datasets import load_dataset


API_BASE = ""
CHAT_COMPLETIONS_PATH = ""

API_KEY = ""

MODEL_NAME = "Qwen/Qwen3-8B"

N_CANDIDATES = 1
MAX_PROBLEMS = 1

OUTPUT_JSONL = "test0.jsonl"

PROMPT_TEMPLATE = """
Your task is to solve a given problem. The problem may ask you to prove a statement, or ask for an answer. If finding an answer is required, you should come up with the answer, and your final solution should also be a rigorous proof of that answer being valid.

Your final solution to the problem should be exceptionally comprehensive and easy-to-follow, which will be rated according to the following evaluation instruction:

''' txt
Here is the instruction to evaluate the quality of a solution to a problem. The problem may ask for a proof of statement, or ask for an answer. If finding an answer is required, the solution should present the answer, and it should also be a rigorous proof of that answer being valid.

Please evaluate the solution and score it according to the following criteria:

- If the solution is completely correct, with all steps executed properly and clearly demonstrated, then the score is 1.
- If the solution is generally correct, but with some details omitted or minor errors, then the score is 0.5.
- If the solution does not actually address the required problem, contains fatal errors, or has severe omissions, then the score is 0.

Additionally, referencing anything from any paper does not save the need to prove the reference. It is okay IF AND ONLY IF the solution also presents a valid proof of the reference argument(s); otherwise, if the solution omits the proof or if the proof provided is not completely correct, the solution should be scored according to the criteria above, and definitely not with a score of 1.
'''

In fact, you already have the ability to rate your solution yourself, so you are expected to reason carefully about how to solve a given problem, evaluate your method according to the instruction, and refine your solution by fixing issues identified until you can make no further progress.

In your final response, you should present a detailed solution to the problem followed by your evaluation of that solution.

- To give a good final response, you should try your best to locate potential issues in your own (partial) solution according to the evaluation instruction above, and fix them as many as you can.
- A good final response should just faithfully present your progress, including the best solution you can give, as well as a faithful evaluation of that solution.
- Only when you fail to locate any issues in your solution should you score it with 1.
- If you do notice some issues in your solution but fail to resolve them with your best efforts, it is totally okay to faithfully present the issues in your final response.
- The worst final response would provide a wrong solution but lie that it is correct or claim that it is correct without careful error checking. A better version should faithfully identify errors in the solution. Remember! You CAN'T cheat! If you cheat, we will know, and you will be penalized!

Your final response should be in the following format:

## Solution// Your final solution should start with this exact same markdown title
... // Your final solution to the problem here. You should try your best to optimize the quality of your solution according to the evaluation instruction above before finalizing it here.

## Self Evaluation// Your evaluation of your own solution above should start with this exact same markdown title
Here is my evaluation of the solution:// Your analysis should start with this exact same phrase
... // Your evaluation here. You are required to present in detail the key steps of the solution or the steps for which you had doubts regarding their correctness, and explicitly analyze whether each step is accurate: for correct steps, explain why you initially doubted their correctness and why they are indeed correct; for erroneous steps, explain the reason for the error and the impact of that error on the solution. You should analyze your solution faithfully. For example, if there are issues in your final solution, you should point them out.

Based on my evaluation, the final overall score should be:
\\boxed{{...}} // where ... should be the final overall score (0, 0.5, or 1, and nothing else) based on the evaluation instruction above. You should reach this score ONLY AFTER careful re-examination of your own solution above.

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
        if len(questions) >= max_problems:
            break

    print(f"Collected {len(questions)} problems for candidate proof generation.")
    return questions


def main():
    questions = get_aops_questions(MAX_PROBLEMS)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
        for idx, q in enumerate(questions):
            print(f"[{idx+1}/{len(questions)}] Generating candidate proofs...")

            prompt = build_prompt(q)

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
                    print(f"  API call failed, retry {attempt+1}: {e}")
                    time.sleep(5)
            else:
                print("  All retries failed, skipping this problem.")
                continue

            record: Dict[str, Any] = {
                "index": idx,
                "question": q,
                "n_candidates": len(candidates),
                "candidates": [
                    {
                        "candidate_index": i,
                        "content": cand,
                    }
                    for i, cand in enumerate(candidates)
                ],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"All done! Results saved to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
