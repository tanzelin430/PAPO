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
LLM-based Proof Grading with 3-tier scoring (0/0.5/1.0).

Uses DeepSeek Math V2 grading standard for evaluating solution quality.

Usage:
    In __init__.py routing:
        from . import llm_proof_grading
        res = llm_proof_grading.compute_score(solution_str, ground_truth, extra_info=extra_info)

    Environment variables:
    - LLM_GRADER_API_BASE: API base URL
    - LLM_GRADER_API_KEY: API key
    - LLM_GRADER_MODEL: Model name
"""

import os
import re
import time

import requests

from .llm_process_reward import (
    strip_think_tags,
    _chat_complete_sync,
    DEFAULT_API_BASE,
    DEFAULT_API_KEY,
    DEFAULT_MODEL,
)


# DeepSeek Math V2 grading standard + CoT analysis
PROOF_GRADING_PROMPT = r"""## Instruction
Your task is to evaluate the quality of a student's solution to a mathematical problem.

## Scoring Rubric
* Score 1: If the solution is completely correct, with all steps executed properly and clearly demonstrated, then the score is 1.
* Score 0.5: If the solution is generally correct, but with some details omitted or minor errors, then the score is 0.5.
* Score 0: If the solution does not actually address the required problem, contains fatal errors, or has severe omissions, then the score is 0.

Special Rule on References:
Additionally, referencing anything from any paper does not save the need to prove the reference. It's okay IF AND ONLY IF the solution also presents a valid proof of the reference argument(s); otherwise, if the solution omits the proof or if the proof provided is not completely correct, the solution should be scored according to the criteria above, and definitely not with a score of 1.

## Problem
{problem_statement}

## Reference Solution
{solution}

## Student Solution
{student_answer}

## Evaluation
Analyze the solution step by step, then provide your score.

Analysis:
...

Score: \boxed{{...}}""".strip()


def _extract_last_boxed(string: str):
    """Extract content from the last \\boxed{} in the string."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    left_brace_idx = None
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
            if left_brace_idx is None:
                left_brace_idx = i
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if left_brace_idx is None or right_brace_idx is None:
        return None

    return string[left_brace_idx + 1 : right_brace_idx].strip()


VALID_SCORES = {0.0, 0.5, 1.0}


def _parse_score(response: str) -> float:
    """Parse score from grader response.

    Accepts 3-tier scores: 0, 0.5, 1.0
    Extracts from \\boxed{score} format, with regex fallback.

    Returns:
        Score as float. Returns 0.0 on parse failure.
    """
    if not response:
        return 0.0

    # Primary: extract from \boxed{}
    boxed = _extract_last_boxed(response)
    if boxed is not None:
        try:
            val = float(boxed.strip())
            if val in VALID_SCORES:
                return val
        except ValueError:
            pass

    # Fallback: regex for "Score is/: N" pattern
    match = re.search(r'(?:score|Score)\s*(?:is\s*:?|:)\s*([\d.]+)', response)
    if match:
        try:
            val = float(match.group(1))
            if val in VALID_SCORES:
                return val
        except ValueError:
            pass

    return 0.0


def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    **kwargs
) -> dict:
    """
    Compute proof-based reward using LLM grading (3-tier: 0/0.5/1.0).

    Uses DeepSeek Math V2 grading standard.

    Args:
        solution_str: Model's generated proof (may contain <think> tags)
        ground_truth: Short answer (may be empty for proof-only problems)
        extra_info: Must contain 'question', 'reference_solution'

    Returns:
        dict with 'score' (0/0.5/1.0), 'acc' (bool)
    """
    if extra_info is None:
        extra_info = {}

    # Get API config from environment or use defaults
    api_base = os.environ.get("LLM_GRADER_API_BASE", DEFAULT_API_BASE)
    api_key = os.environ.get("LLM_GRADER_API_KEY", DEFAULT_API_KEY)
    model = os.environ.get("LLM_GRADER_MODEL", DEFAULT_MODEL)

    # Extract fields from extra_info
    question = extra_info.get("question", "") or extra_info.get("problem", "")
    reference_solution = extra_info.get("reference_solution", "")

    # If no reference solution, return zero score
    if not reference_solution:
        return {
            "score": 0.0,
            "acc": False,
            "error": "no_reference_solution"
        }

    # Strip <think> tags from student solution
    student_solution = strip_think_tags(solution_str)

    if not student_solution:
        return {
            "score": 0.0,
            "acc": False,
            "error": "empty_response"
        }

    # Build grading prompt
    prompt = PROOF_GRADING_PROMPT.format(
        problem_statement=question,
        solution=reference_solution,
        student_answer=student_solution
    )
    messages = [{"role": "user", "content": prompt}]

    try:
        raw_response = _chat_complete_sync(api_base, api_key, model, messages)

        # Strip think tags from grader response
        grading_response = strip_think_tags(raw_response)

        # Parse score (returns 0.0 for unrecognized)
        score = _parse_score(grading_response)

        return {
            "score": score,
            "acc": score == 1.0,
        }

    except Exception as e:
        print(f"[LLM Proof Grading ERROR] {type(e).__name__}: {str(e)[:200]}")
        return {
            "score": 0.0,
            "acc": False,
        }
