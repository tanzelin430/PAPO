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
LLM-based Answer Grading Reward Function for Math RL Training.

This module uses an LLM to judge whether the student's answer is equivalent
to the reference answer. More flexible than rule-based grading (math_verify)
as it can handle different representations of the same answer.

Flow:
1. Strip <think>...</think> tags from response
2. Extract answer from \\boxed{} (if present)
3. Send extracted answer to LLM grader for comparison

Scoring: CORRECT → 1.0, INCORRECT → 0.0

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


# Grading prompt template
GRADING_PROMPT = """Question: {question}

Student's Answer: {student_answer}

Reference Answer: {reference_answer}

You only need to refer to the reference answer to grade the student's answer. Sometimes the student's answer is expressed in a different way from the reference answer, but the meaning is the same, and you should still consider it correct. If they are not equivalent in mathematical sense, you should consider it incorrect.

Note 1: You don't need to solve the problem yourself. Just grade the student's answer based on the reference answer.

Note 2: If the reference answer is a range, please make sure the student's answer is strictly identical, including the open or closed interval.

Note 3: If the reference answer is an expression and it looks like the student's answer is equivalent to the reference answer, you should present the derivation process to check if they are equivalent.

Note 4: If the reference answer includes multiple solutions, please make sure the student's answer covers all of them.

Please provide a brief explanation (a few sentences) of your grading process and put your final grade in the following format:

Final Grade: CORRECT or INCORRECT
"""


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> content from text.

    Safe for both Qwen3 (has think tags) and Qwen2.5 (no think tags).
    """
    if not text:
        return ""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def _last_boxed_only_string(string: str) -> str:
    """Extract content from the last \\boxed{} or \\fbox{} in the string.

    Uses bracket counting to handle nested braces correctly.
    Pure string operation, no sympy involved.

    Args:
        string: Input string potentially containing \\boxed{...}

    Returns:
        Content inside the last boxed, or None if not found
    """
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

    return string[left_brace_idx + 1:right_brace_idx].strip()


def extract_answer(response: str) -> tuple[bool, str]:
    """Extract answer from model response.

    1. Remove </think> prefix (keep content after it)
    2. Try to extract from \\boxed{}
    3. If no boxed, return the whole response

    Args:
        response: Model's full response

    Returns:
        (is_boxed, extracted_answer) tuple
    """
    # Remove think tags first
    response = strip_think_tags(response)

    # Also handle </think> without opening tag (edge case)
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()

    # Try to extract boxed answer
    ans_boxed = _last_boxed_only_string(response)
    if ans_boxed:
        return True, ans_boxed

    return False, response


def parse_grade(response: str) -> bool:
    """Parse CORRECT/INCORRECT from grading response.

    Returns True for CORRECT, False for INCORRECT or unknown.
    """
    if not response:
        return False

    # Look for "Final Grade: CORRECT" or "Final Grade: INCORRECT"
    match = re.search(r'Final\s+Grade:\s*(CORRECT|INCORRECT)', response, re.IGNORECASE)
    if match:
        return match.group(1).upper() == "CORRECT"

    # Fallback: look for standalone CORRECT/INCORRECT
    if re.search(r'\bCORRECT\b', response, re.IGNORECASE):
        if not re.search(r'\bINCORRECT\b', response, re.IGNORECASE):
            return True

    return False


def _chat_complete_sync(api_base: str, api_key: str, model: str,
                        messages: list, max_retries: int = 3) -> str:
    """Send synchronous HTTP request to LLM API using requests library.

    Args:
        api_base: API base URL
        api_key: API key
        model: Model name
        messages: Chat messages
        max_retries: Number of retry attempts

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
        "max_tokens": 16384, # Allow long grading explanation
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
    Compute reward using LLM-based answer grading (synchronous).

    This is the main entry point, compatible with verl's reward routing.

    Args:
        solution_str: Model's generated solution (may contain <think> tags)
        ground_truth: Expected answer (reference)
        extra_info: Optional dict with 'question' key
        **kwargs: Additional arguments (ignored)

    Returns:
        dict with 'score' (0.0 or 1.0), 'acc' (bool), and debug info
    """
    if extra_info is None:
        extra_info = {}

    # Get API config from environment or use defaults
    api_base = os.environ.get("LLM_GRADER_API_BASE", DEFAULT_API_BASE)
    api_key = os.environ.get("LLM_GRADER_API_KEY", DEFAULT_API_KEY)
    model = os.environ.get("LLM_GRADER_MODEL", DEFAULT_MODEL)

    # Extract question
    question = extra_info.get("question", "") or extra_info.get("problem", "")

    # Step 1: Extract answer from response
    # - Strip <think> tags
    # - Extract from \boxed{} if present
    is_boxed, student_answer = extract_answer(solution_str)

    # If empty response, return 0
    if not student_answer:
        return {
            "score": 0.0,
            "acc": False,
            "error": "empty_response",
            "student_answer": "",
            "reference_answer": ground_truth,
        }

    # Step 2: Build grading prompt
    prompt = GRADING_PROMPT.format(
        question=question,
        student_answer=student_answer,
        reference_answer=ground_truth
    )
    messages = [{"role": "user", "content": prompt}]

    try:
        # Step 3: Call LLM for grading (synchronous)
        raw_response = _chat_complete_sync(api_base, api_key, model, messages)

        # Strip think tags from grader response (in case grader is also a reasoning model)
        grading_response = strip_think_tags(raw_response)

        # Step 4: Parse grade
        is_correct = parse_grade(grading_response)

        # Debug output (only for no_boxed cases to reduce noise)
        # if not is_boxed:
        #     print(f"[LLM Grading] no_boxed | student_answer_len={len(student_answer)} | grade={'CORRECT' if is_correct else 'INCORRECT'}")

        return {
            "score": 1.0 if is_correct else 0.0,
            "acc": is_correct,
        }

    except Exception as e:
        # On error, log and return 0 score
        print(f"[LLM Grading ERROR] {type(e).__name__}: {str(e)[:200]}")
        return {
            "score": 0.0,
            "acc": False,
        }
