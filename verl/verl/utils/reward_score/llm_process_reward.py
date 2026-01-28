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

This module implements process-based grading using an internal vLLM reward model,
following the IMO Bench ProofAutoGrader approach. It evaluates the quality
of mathematical solutions based on process, not just final answers.

Scoring Scale: IMO 4-level (7/6/1/0) normalized to 0-1
- Correct (7) -> 1.0
- Almost (6) -> 0.857 (6/7)
- Partial (1) -> 0.143 (1/7)
- Incorrect (0) -> 0.0

Usage:
    This module is used with verl's reward loop system. Configure as:
    - reward_model.enable=True
    - reward_model.use_reward_loop=True
    - reward_model.model.path=/path/to/Qwen3-14B
    - custom_reward_function.path=verl/utils/reward_score/llm_process_reward.py
    - custom_reward_function.name=compute_score
"""

import json
import re
from typing import Optional

import aiohttp
from openai.types.chat import ChatCompletion


# Prompt adapted from IMO Bench ProofAutoGrader (Appendix B.5)
# Removed IMO-specific references, kept core structure
PROCESS_GRADING_PROMPT = """
You are an expert mathematical solution grader. Your task is to evaluate a proposed solution strictly and rigorously. Keep in mind the standards are extremely high: only arguments that are logically sound, complete, and precise should be rewarded.

### General Scoring Rubric
Scores are assigned on a 0-7 scale. The general guidelines are:

* **7 Points (Correct):** The solution is complete, correct, and fully rigorous. If the submission contains incorrect attempts or lines of reasoning but ultimately presents a complete and correct solution, it should still be awarded full points; the presence of earlier, discarded work does not detract from the final correct proof.

* **6 Points (Almost Correct):** The solution is almost correct with a sound core argument, but contains minor errors in calculation or small gaps in logic. Missing proofs for major components, unjustified claims, or sketchy arguments are **not** eligible for 6 points.

* **1 Point (Partial Progress):** The solution demonstrates substantial progress toward the correct answer. Initial observations, reformulating the problem without making substantive headway, or proving partial results that don't contribute to the solution are generally **not** eligible for this score.

* **0 Points (Incorrect):** The solution doesn't make substantial progress that is a key step in the full solution or is fundamentally flawed. All partial progress without key results or lacking rigor also fall in this category.

### Input Data
You are provided with the following:
1. **Problem Statement:** The math problem to be solved.
2. **Ground Truth Solution:** A reference solution. Assume this solution is correct. It demonstrates one valid approach.
3. **Proposed Solution:** The student submission to be graded.

### Evaluation Process
You must follow this structured process:

1. **Analyze References:** Meticulously read and understand the problem and Ground Truth Solution. Identify the key steps for a complete solution.

2. **Step-by-Step Verification:** Verify the logical validity and rigor of every step. Identify all flaws, gaps, assumptions, and errors. **Make sure you fully understand every piece of logic behind each step of the proposed solution, you must be careful for solutions that 'pretend' to be correct.**

3. **Assess Progress:** Determine the extent of non-trivial progress made.

4. **Score Determination:** Compare the findings against the General Rubric to determine the final score.

### Output Requirements
You must provide your final score in the format <points>N out of 7</points>. Ensure the '<points>' block is used **only once**, as your answer will be parsed based on the first <points></points> block that appears in your whole response.

---

**PROBLEM STATEMENT**
{problem}

**GROUND-TRUTH SOLUTION**
{reference_solution}

**PROPOSED SOLUTION**
{student_solution}

---

Present your detailed thought process and formal justification based on the scoring rubric, and finally present your final score in the format below.

[Select one of the following options]
<points>7 out of 7</points>
<points>6 out of 7</points>
<points>1 out of 7</points>
<points>0 out of 7</points>
""".strip()


def parse_imo_score(response: str) -> int:
    """Parse LLM output to IMO score (7/6/1/0).

    Expected format: <points>N out of 7</points>

    Args:
        response: The LLM response text

    Returns:
        Score as integer (7, 6, 1, or 0)
    """
    if response is None:
        return 0

    # Primary: Extract from <points>N out of 7</points> format
    match = re.search(r'<points>\s*(\d)\s*out of 7\s*</points>', response, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        if score in [0, 1, 6, 7]:
            return score
        # Map other scores to closest valid score
        if score >= 5:
            return 7 if score >= 6 else 6
        elif score >= 2:
            return 1
        return 0

    # Fallback 1: Look for "N out of 7" pattern without tags
    match = re.search(r'(\d)\s*out of 7', response, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        if score in [0, 1, 6, 7]:
            return score
        if score >= 5:
            return 7 if score >= 6 else 6
        elif score >= 2:
            return 1
        return 0

    # Fallback 2: Extract number from last line
    lines = response.strip().split('\n')
    if lines:
        last_line = lines[-1].strip()
        for char in last_line:
            if char in '0167':
                return int(char)

    # Fallback 3: keyword matching
    text = response.lower()
    if 'correct' in text and 'incorrect' not in text and 'almost' not in text:
        return 7
    elif 'almost' in text:
        return 6
    elif 'partial' in text:
        return 1

    return 0


def normalize_imo_score(imo_score: int) -> float:
    """Normalize IMO score (7/6/1/0) to 0-1 range.

    Args:
        imo_score: Raw IMO score (7, 6, 1, or 0)

    Returns:
        Normalized score between 0.0 and 1.0
    """
    return {7: 1.0, 6: 6/7, 1: 1/7, 0: 0.0}.get(imo_score, 0.0)


async def _chat_complete(router_address: str, chat_complete_request: dict, max_retries: int = 3):
    """Send async HTTP request to vLLM reward router.

    Args:
        router_address: The reward router address (e.g., "localhost:8000")
        chat_complete_request: The chat completion request payload
        max_retries: Number of retry attempts on failure

    Returns:
        ChatCompletion object from the response
    """
    import asyncio

    url = f"http://{router_address}/v1/chat/completions"
    last_error = None

    for attempt in range(max_retries):
        try:
            timeout = aiohttp.ClientTimeout(total=120)  # 2 minute timeout for long responses
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=chat_complete_request) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise Exception(f"HTTP {resp.status}: {error_text}")
                    output = await resp.text()
                    output = json.loads(output)
                    return ChatCompletion(**output)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    raise last_error


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    reward_router_address: str = None,
    reward_model_tokenizer = None,
    **kwargs
) -> dict:
    """
    Compute process-based reward using internal vLLM reward model.

    This is the main entry point for verl's reward loop system (async).
    It requires reward_router_address to be provided by the reward loop.

    Args:
        data_source: The data source identifier (e.g., "numina_math_process")
        solution_str: Model's generated solution
        ground_truth: Expected answer (not used directly, for compatibility)
        extra_info: Must contain 'question' and 'reference_solution'
        reward_router_address: HTTP address of the reward model router
        reward_model_tokenizer: Tokenizer for the reward model (not used for chat API)
        **kwargs: Additional arguments (ignored)

    Returns:
        dict with 'score', 'acc', 'imo_score', 'genrm_response'
    """
    if extra_info is None:
        extra_info = {}

    # Extract question and reference solution from extra_info
    question = extra_info.get("question", "")
    reference_solution = extra_info.get("reference_solution", "")

    # Fallback: try 'problem' key if 'question' is not present
    if not question:
        question = extra_info.get("problem", "")

    # If no reference solution, return zero score
    if not reference_solution:
        return {
            "score": 0.0,
            "acc": False,
            "imo_score": 0,
            "error": "no_reference_solution"
        }

    # If no reward router address, cannot proceed
    if not reward_router_address:
        return {
            "score": 0.0,
            "acc": False,
            "imo_score": 0,
            "error": "no_reward_router_address"
        }

    # Build the grading prompt
    grm_prompt = PROCESS_GRADING_PROMPT.format(
        problem=question,
        reference_solution=reference_solution,
        student_solution=solution_str
    )

    # Prepare chat completion request
    # Use low temperature for consistent grading
    messages = [{"role": "user", "content": grm_prompt}]

    # Get model path from reward_model_tokenizer if available
    # This should match the reward model path configured in the training script
    model_name = "default"  # vLLM router doesn't strictly require model name for single model
    if reward_model_tokenizer is not None:
        model_name = getattr(reward_model_tokenizer, 'name_or_path', 'default')

    chat_complete_request = {
        "model": model_name,
        "messages": messages,
        "max_tokens": 2048,  # Allow detailed analysis before scoring
        "temperature": 0.1,  # Low temp for consistent grading
        # Disable thinking mode for Qwen3 models
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False}
        }
    }

    try:
        result = await _chat_complete(
            router_address=reward_router_address,
            chat_complete_request=chat_complete_request,
            max_retries=3
        )

        grm_response = result.choices[0].message.content
        imo_score = parse_imo_score(grm_response)
        normalized_score = normalize_imo_score(imo_score)

        return {
            "score": normalized_score,
            "acc": imo_score == 7,
            "imo_score": imo_score,
            "genrm_response": grm_response[:500] if grm_response else None
        }
    except Exception as e:
        return {
            "score": 0.0,
            "acc": False,
            "imo_score": 0,
            "error": str(e)
        }
