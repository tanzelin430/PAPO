import json
import time
import re
from typing import Dict, Any, Generator, Optional

import requests

INPUT_JSONL = "test1.jsonl"
OUTPUT_JSONL = "test2.jsonl"

API_BASE = ""
CHAT_COMPLETIONS_PATH = ""

API_KEY = ""

META_MODEL_NAME = "gpt-5"

TEMPERATURE = 0.0
MAX_TOKENS = 4096
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 5


PROMPT_META_VERIFICATION = """
You are given a "problem", "solution", and "solution evaluation", and you need to assess the whether this "solution evaluation" is reasonable.

First, "solution evaluation" is generated to evaluate the quality of the "solution", by prompting a verifier with the rules below (these are not your rules):

```
Please evaluate the solution and score it according to the following criteria:
- If the solution is completely correct, with all steps executed properly and clearly demonstrated, then the score is 1
- If the solution is generally correct, but with some details omitted or minor errors, then the score is 0.5
- If the solution does not actually address the required problem, contains fatal errors, or has severe omissions, then the score is 0

Additionally, referencing anything from any paper does not save the need to prove the reference. It's okay IF AND ONLY IF the solution also presents a valid proof of the reference argument(s); otherwise, if the solution omits the proof or if the proof provided is not completely correct, the solution should be scored according to the criteria above, and definitely not with a score of 1
```

Next, I will introduce the rules for you to analyze the quality of the "solution evaluation":

1. Your task is to analyze the "solution evaluation". You do not need to solve the "problem", nor do you need to strictly assess whether the "solution" is accurate. Your only task is to strictly follow the rules below to evaluate whether the "solution evaluation" is reasonable.

2. You need to analyze the content of the "solution evaluation" from three aspects:

Step Restatement: In the "solution evaluation", certain behaviors of the "solution" may be restated. You need to return to the original text of the "solution" and check whether the "solution" actually has these behaviors mentioned in the "solution evaluation".

Defect Analysis: "solution evaluation" may point out errors or defects in the "solution". You need to carefully analyze whether the mentioned errors and defects are indeed valid.

Expression Analysis: Whether the "solution evaluation"'s expressions are accurate.

Score Analysis: Whether the final score given by the "solution evaluation" matches the defects it found. You need to analyze according to the scoring rules given above.

3. The most important part is **defect analysis**: In this part, your core task is to check whether the errors or defects of the "solution" pointed out in the "solution evaluation" are reasonable. In other words, any positive components about the "solution" in the "solution evaluation", regardless of whether they are reasonable, are not within your evaluation scope.

- For example: If the "solution evaluation" says that a certain conclusion in the "solution" is correct, but actually this conclusion is incorrect, then you do not need to care about this point. All parts that the "solution evaluation" considers correct do not belong to your evaluation scope.
- Specifically: If the "solution evaluation" believes that the "solution" is completely accurate and has not found any errors or defects, then regardless of whether the "solution" itself is actually accurate, even if there are obvious errors, you should still consider its analysis of errors to be reasonable.

**Importantly**, for defects found by the "solution evaluation", you need to analyze two points simultaneously:

- whether this defect actually exists
- whether the "solution evaluation"'s analysis of this defect is accurate

These two aspects constitute the analysis of defects.

4. About **expression analysis**, if there are certain expression errors in the "solution evaluation", even minor errors in details, you need to identify them. However, please note that identifying incorrect steps in the "solution" as correct steps does not constitute an **expression error**.

In practice, expression errors include but are not limited to:

- If the "solution evaluation" identifies some reasoning step(s) in the "solution" as incorrect, then it cannot further indicate that subsequent conclusion(s) depending on those reasoning step(s) are wrong, but can only indicate that subsequent conclusion(s) are "not rigorously demonstrated."
- Typos and calculation errors made by "solution evaluation"
- Inaccurate restatement of content from "solution"

5. Finally, you need to present your analysis of the "solution evaluation" in your output and also rate its quality based on the rules below:

First, if there is at least one unreasonable defect among the defects found by the "solution evaluation", then you only need to do **defect analysis**:

- If all defects found by the "solution evaluation" are unreasonable, then you should rate it with \(0\)
- If some defects found by the "solution evaluation" are reasonable and some are unreasonable, then your rating should be \(0.5\)

Next, if the "solution evaluation" points out no errors or defects, or all defects found by the evaluation are reasonable, then you should do the following things:

- Analyze whether "expression errors" exist in the "solution evaluation" (**expression analysis**) or whether "solution evaluation" gives a wrong score according to the rules for "solution evaluation" (**score analysis**). If yes, you should rate the "solution evaluation" with \(0.5\); if no, your rating should be \(1\)

Your output should follow the format below:

Here is my analysis of the "solution evaluation":
... // Your analysis here.

Based on my analysis, I will rate the "solution evaluation" as:
\\boxed{{...}} // where ... should be a numerical rating of the "solution evaluation" (0, 0.5, or 1, and nothing else) based on the criteria above.

---

Here is your task input:

## Problem
{statement}

## Solution
{proof}

## Solution Evaluation
{rating}
""".strip()


def build_meta_prompt(statement: str, proof: str, rating: str) -> str:
    return PROMPT_META_VERIFICATION.format(
        statement=statement,
        proof=proof,
        rating=rating,
    )


def call_meta_verifier(statement: str, proof: str, rating: str) -> Dict[str, Any]:
    url = API_BASE.rstrip("/") + CHAT_COMPLETIONS_PATH

    prompt = build_meta_prompt(statement, proof, rating)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    body = {
        "model": META_MODEL_NAME,
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
            f"Meta-verifier API call failed, status={resp.status_code}, body={resp.text[:500]}"
        )

    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("Meta-verifier response does not contain 'choices' field")

    content = choices[0]["message"]["content"]

    score = extract_score_from_text(content)

    return {
        "meta_GT": score,
    }


def extract_score_from_text(text: str) -> float:
    m = re.search(
        r"\\boxed\{\s*(0(?:\.0)?|0\.5|1(?:\.0)?)\s*\}",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return float(m.group(1))

    candidates = []
    for s in ["0.5", "1", "0"]:
        idx = text.rfind(s)
        if idx != -1:
            candidates.append((idx, s))
    if candidates:
        candidates.sort()
        s = candidates[-1][1]
        return float(s)

    raise RuntimeError("Failed to parse score from meta output:\n" + text)


def iter_jsonl(path: str) -> Generator[Dict[str, Any], None, None]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    print(f"Reading GT dataset: {INPUT_JSONL}")
    records = list(iter_jsonl(INPUT_JSONL))
    print(f"Loaded {len(records)} problem records.")

    out_f = open(OUTPUT_JSONL, "w", encoding="utf-8")

    try:
        for ridx, rec in enumerate(records):
            idx = rec.get("index")
            statement = rec.get("question", "")
            candidates = rec.get("candidates", [])

            print(f"[{ridx+1}/{len(records)}] problem index={idx}, num candidates={len(candidates)}")

            for cand in candidates:
                cand_idx = cand.get("candidate_index")
                proof = cand.get("content", "")

                gt_obj = cand.get("GT_verification")
                if not gt_obj or not gt_obj.get("verification"):
                    print(f"  - candidate_index={cand_idx} has no GT_verification, meta_verification set to None")
                    cand["meta_verification"] = None
                    continue

                rating_text = gt_obj["verification"]

                print(f"  - Meta-verifying candidate_index={cand_idx} ...")

                meta_result: Optional[Dict[str, Any]] = None
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        meta_result = call_meta_verifier(statement, proof, rating_text)
                        break
                    except Exception as e:
                        print(f"    Call failed, retry {attempt}: {e}")
                        if attempt < MAX_RETRIES:
                            time.sleep(RETRY_SLEEP_SECONDS)
                        else:
                            print("    All retries failed, meta_verification is set to None for this candidate")
                            meta_result = None

                cand["meta_verification"] = meta_result

            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()

    finally:
        out_f.close()

    print(f"Done! New dataset written to: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
