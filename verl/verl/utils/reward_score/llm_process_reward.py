# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
LLM-based Process Reward Function for Math RL Training.

This module uses an LLM to grade the PROCESS of a mathematical solution,
not just the final answer. More informative than answer-only grading
as it provides dense reward signal for reasoning quality.

Flow:
1. Strip <think>...</think> tags from response
2. Build grading prompt with problem, reference solution, and student solution
3. Send to LLM grader for process-based evaluation
4. Parse score from <points>X out of 7</points> format

Scoring: Continuous 0.0-7.0, normalized to 0-1
- Coverage (0-4): checkpoint satisfaction
- Rigor (0-2): logical flow quality
- Correctness (0-1): consistency and final answer

Usage:
    In __init__.py routing (synchronous):
    - Uses compute_score (pure sync, requests-based)

    Environment variables:
    - LLM_GRADER_API_BASE: API base URL
    - LLM_GRADER_API_KEY: API key
    - LLM_GRADER_MODEL: Model name
"""

import os
import re
import time
from typing import Optional

import requests


# API Configuration (can be overridden by environment variables)
DEFAULT_API_BASE = "http://localhost:8000/v1"
DEFAULT_API_KEY = "dummy"
DEFAULT_MODEL = "GPT-OSS-20B"


# Checkpoint-based continuous scoring prompt
# Produces dense rewards (0.0-7.0) for better GRPO training
PROCESS_GRADING_PROMPT = r"""You are an expert mathematical solution grader. Your job is to grade the PROPOSED SOLUTION by comparing its reasoning process against the GROUND-TRUTH SOLUTION (reference).

### Key principle: grade PROCESS quality, not only the final answer
- The reference shows one valid approach and reveals the key logical checkpoints needed for a rigorous solution.
- The student may use different notation, order, or an equivalent method. Accept any logically valid alternative.
- However, missing key reasoning steps, unjustified leaps, or incorrect claims must be penalized.

### Step 0: Extract checkpoints from the reference (CRITICAL)
Read the GROUND-TRUTH SOLUTION and extract checkpoints that represent the minimal reasoning chain for a complete solution.

Create two tiers of checkpoints:
1) **Key checkpoints (weight 2.0)**: essential reasoning steps without which the solution is not rigorous/complete.
   Examples: key lemma, core transformation, crucial case split, main invariant/idea, correctness justification.
2) **Supporting checkpoints (weight 1.0)**: important but not essential details.
   Examples: setup/definitions, routine algebra simplifications, boundary conditions, finishing steps.

Checkpoint requirements:
- Each checkpoint must be specific and verifiable.
- Keep each checkpoint short (1 sentence).
- Prefer 4–10 total checkpoints.

### Step 1: Evaluate the proposed solution against each checkpoint
For each checkpoint, assign exactly one label:
- **satisfied**: correctly and explicitly addressed
- **partially_satisfied**: attempted but incomplete / unclear / minor gap
- **missing**: not present
- **wrong**: present but incorrect

Important grading rules:
- If the student claims "obvious/straightforward" for a nontrivial step without showing it, treat as partially_satisfied or missing (choose the lower if unsure).
- If an early part contains errors but the final solution later provides a correct complete argument, do NOT penalize the discarded attempts.
- If the student uses an alternative approach, map their steps to the checkpoints by logical equivalence.

### Step 2: Compute a continuous score (0.0–7.0)
Compute three subscores, then combine:

A) **Coverage score (0–4)** based on checkpoints (MOST IMPORTANT)
- For each checkpoint:
  - satisfied = 1.0
  - partially_satisfied = 0.5
  - missing = 0.0
  - wrong = 0.0 (and mark an error)
- Weighted average over checkpoints using weights (key=2.0, supporting=1.0)
- Coverage score = 4.0 * (weighted_points / weighted_total)

B) **Rigor score (0–2)**
Judge the overall logical rigor of the student solution:
- 2.0: every nontrivial step justified; clear logical flow; no major gaps
- 1.0: some gaps / handwaving, but core is mostly auditable
- 0.0: many gaps; reasoning not auditable

Strict caps:
- If there is ANY unjustified leap on a key step, rigor score is capped at 1.0.
- If there are MULTIPLE unjustified leaps, rigor score is capped at 0.5.

C) **Correctness/Consistency score (0–1)**
- 1.0: no contradictions; statements consistent; final conclusion matches a correct solution OR is clearly equivalent
- 0.5: minor arithmetic/algebra slips that do not break the core argument
- 0.0: fatal logical error, contradiction, or wrong final conclusion without correction

Fatal error rule:
- If the solution contains a fatal error that invalidates the main argument (and is not later fixed), then total score is capped at 2.0.

Final score:
- total = A + B + C
- Round to the nearest 0.5 (or nearest 0.1 if you prefer finer grading).
- Ensure 0.0 <= total <= 7.0.

### Output requirements (STRICT)
1) Provide a brief, structured justification (no long essay).
2) Then output the final score EXACTLY ONCE in the format:
   <points>X out of 7</points>
Where X is a number from 0.0 to 7.0 (allowed decimals like 5.5).

---

**PROBLEM STATEMENT**
{problem}

**GROUND-TRUTH SOLUTION**
{reference_solution}

**PROPOSED SOLUTION**
{student_solution}

Now perform:
(1) Extract checkpoints from the reference,
(2) Evaluate the proposed solution checkpoint-by-checkpoint,
(3) Compute subscores A/B/C, apply caps if needed,
(4) Output final <points>...</points> exactly once.
""".strip()


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> content from text.

    Safe for both Qwen3 (has think tags) and Qwen2.5 (no think tags).
    """
    if not text:
        return ""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def parse_imo_score(response: str) -> float:
    """Parse LLM output to continuous score (0.0-7.0).

    Expected format: <points>X out of 7</points>
    Where X can be a decimal like 5.5

    Args:
        response: The LLM response text

    Returns:
        Score as float between 0.0 and 7.0
    """
    if response is None:
        return 0.0

    # Primary: Extract from <points>X out of 7</points> format (supports decimals)
    match = re.search(r'<points>\s*([\d.]+)\s*out of 7\s*</points>', response, re.IGNORECASE)
    if match:
        try:
            score = float(match.group(1))
            return min(max(score, 0.0), 7.0)  # Clamp to [0, 7]
        except ValueError:
            pass

    # Fallback: Look for "X out of 7" pattern without tags
    match = re.search(r'([\d.]+)\s*out of 7', response, re.IGNORECASE)
    if match:
        try:
            score = float(match.group(1))
            return min(max(score, 0.0), 7.0)
        except ValueError:
            pass

    return 0.0


def normalize_imo_score(imo_score: float) -> float:
    """Normalize score (0.0-7.0) to 0-1 range.

    Args:
        imo_score: Raw score between 0.0 and 7.0

    Returns:
        Normalized score between 0.0 and 1.0
    """
    return imo_score / 7.0


def _chat_complete_sync(api_base: str, api_key: str, model: str,
                        messages: list, max_retries: int = 3,
                        max_tokens: int = 16384) -> str:
    """Send synchronous HTTP request to LLM API using requests library.

    Args:
        api_base: API base URL
        api_key: API key
        model: Model name
        messages: Chat messages
        max_retries: Number of retry attempts
        max_tokens: Maximum tokens for completion response

    Returns:
        Response content string
    """
    url = f"{api_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,  # Low temp for consistent grading
        "max_tokens": max_tokens,
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
            output = response.json()
            return output["choices"][0]["message"]["content"]
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff

    raise last_error


def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    **kwargs
) -> dict:
    """
    Compute process-based reward using LLM grading (synchronous).

    This is the main entry point, compatible with verl's reward routing.

    Args:
        solution_str: Model's generated solution (may contain <think> tags)
        ground_truth: Expected answer (not used directly for process grading)
        extra_info: Must contain 'question' and 'reference_solution'
        **kwargs: Additional arguments (ignored)

    Returns:
        dict with 'score' (0.0-1.0), 'acc' (bool), 'imo_score' (0.0-7.0)
    """
    if extra_info is None:
        extra_info = {}

    # Get API config from environment or use defaults
    api_base = os.environ.get("LLM_GRADER_API_BASE", DEFAULT_API_BASE)
    api_key = os.environ.get("LLM_GRADER_API_KEY", DEFAULT_API_KEY)
    model = os.environ.get("LLM_GRADER_MODEL", DEFAULT_MODEL)

    # Extract question and reference solution from extra_info
    question = extra_info.get("question", "") or extra_info.get("problem", "")
    reference_solution = extra_info.get("reference_solution", "")

    # If no reference solution, return zero score
    if not reference_solution:
        return {
            "score": 0.0,
            "acc": False,
            "imo_score": 0.0,
            "error": "no_reference_solution"
        }

    # Step 1: Strip <think> tags from student solution
    student_solution = strip_think_tags(solution_str)

    # If empty response, return 0
    if not student_solution:
        return {
            "score": 0.0,
            "acc": False,
            "imo_score": 0.0,
            "error": "empty_response"
        }

    # Step 2: Build grading prompt
    prompt = PROCESS_GRADING_PROMPT.format(
        problem=question,
        reference_solution=reference_solution,
        student_solution=student_solution
    )
    messages = [{"role": "user", "content": prompt}]

    try:
        # Step 3: Call LLM for grading (synchronous)
        raw_response = _chat_complete_sync(api_base, api_key, model, messages)

        # Strip think tags from grader response (in case grader is also a reasoning model)
        grading_response = strip_think_tags(raw_response)

        # Step 4: Parse score
        imo_score = parse_imo_score(grading_response)
        normalized_score = normalize_imo_score(imo_score)

        return {
            "score": normalized_score,
            "acc": imo_score >= 6.5,  # Consider 6.5+ as "correct" for accuracy metric
            "imo_score": imo_score,
        }

    except Exception as e:
        # On error, log and return 0 score
        print(f"[LLM Process Grading ERROR] {type(e).__name__}: {str(e)[:200]}")
        return {
            "score": 0.0,
            "acc": False,
            "imo_score": 0.0,
        }
