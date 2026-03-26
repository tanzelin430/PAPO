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
LLM-based Proof Grading with IMO-Bench style rubric (4-tier: 0/1/6/7 → 0.0/0.2/0.8/1.0).

Adapted from IMO-ProofBench ProofAutoGrader (Appendix B.5).
Key features vs original llm_proof_grading.py:
- Structured 4-step evaluation process
- Stricter partial credit (must demonstrate key progress)
- Explicit warning against "pretend to be correct" solutions
- 4-tier scoring: 7→1.0, 6→0.8, 1→0.2, 0→0.0

Usage:
    Set PRM_RUBRIC=imobench to activate this rubric in __init__.py routing.
"""

import os
import re

from .llm_process_reward import (
    strip_think_tags,
    _chat_complete_sync,
    DEFAULT_API_BASE,
    DEFAULT_API_KEY,
    DEFAULT_MODEL,
)


PROOF_GRADING_PROMPT = r"""You are an expert grader for mathematical competitions. Your task is to evaluate a proposed solution strictly and rigorously. Keep in mind the standards are extremely high: only arguments that are logically sound, complete, and precise should be rewarded.

### General Scoring Rubric

Scores are assigned on a 0-7 scale. The general guidelines are:

* **7 Points (Correct):** The solution is complete, correct, and fully rigorous. If the submission contains incorrect attempts or lines of reasoning but ultimately presents a complete and correct solution, it should still be awarded full points; the presence of earlier, discarded work does not detract from the final correct proof.

* **6 Points (Almost Correct):** The solution is almost correct with a sound core argument, but contains minor errors in calculation or small gaps in logic. Missing proofs for major components, unjustified claims, or sketchy arguments are **not** eligible for 6 points.

* **1 Point (Partial Progress):** The solution demonstrates substantial progress toward a correct solution — for example, correctly establishing a key lemma or intermediate result that is essential to the full proof. Initial observations, reformulating the problem without making substantive headway, or proving results not relevant to the solution are generally **not** eligible for this score.

* **0 Points (Incorrect):** The solution doesn't make substantial progress or is fundamentally flawed. All partial progress without key results or lacking rigor also fall in this category.

### Input Data

You are provided with the following:
1. **Problem Statement:** The mathematical problem to be solved.
2. **Reference Solution:** A correct reference solution. Assume this solution is correct. It demonstrates one valid approach.
3. **Proposed Solution:** The student submission to be graded.

### Evaluation Process

You must follow this structured process:

1. **Analyze References:** Meticulously read and understand the problem and Reference Solution. Identify the key steps required for a complete solution.

2. **Step-by-Step Verification:** Verify the logical validity and rigor of every step in the Proposed Solution. Identify all flaws, gaps, assumptions, and errors. **Make sure you fully understand every piece of logic behind each step — be careful for solutions that 'pretend' to be correct by using plausible-sounding but invalid reasoning.**

3. **Assess Progress:** Determine the extent of non-trivial progress made toward a correct and complete solution.

4. **Score Determination:** Compare your findings against the General Scoring Rubric to determine the final score.

### Problem Statement
{problem_statement}

### Reference Solution
{solution}

### Proposed Solution
{student_answer}

### Evaluation
Present your detailed analysis following the 4-step evaluation process above, then provide your final score.

<points>N out of 7</points>""".strip()


# Map IMO 0-7 scale to 0.0-1.0 reward
# 7 → 1.0 (Correct), 6 → 0.8 (Almost), 1 → 0.2 (Partial), 0 → 0.0 (Incorrect)
SCORE_MAP = {7: 1.0, 6: 0.8, 1: 0.2, 0: 0.0}
VALID_POINTS = {0, 1, 6, 7}


def _parse_score(response: str) -> float:
    """Parse score from grader response.

    Expects <points>N out of 7</points> format.
    Maps: 7→1.0, 6→0.8, 1→0.2, 0→0.0.

    Returns:
        Score as float. Returns 0.0 on parse failure.
    """
    if not response:
        return 0.0

    # Primary: extract from <points>N out of 7</points>
    match = re.search(r'<points>\s*(\d+)\s*out of 7\s*</points>', response)
    if match:
        points = int(match.group(1))
        if points in VALID_POINTS:
            return SCORE_MAP[points]
        # Clamp to nearest valid tier
        if points >= 7:
            return 1.0
        elif points >= 4:
            return 0.8
        elif points >= 1:
            return 0.2
        else:
            return 0.0

    # Fallback: look for "N out of 7" without tags
    match = re.search(r'(\d+)\s*out of 7', response)
    if match:
        points = int(match.group(1))
        if points in VALID_POINTS:
            return SCORE_MAP[points]
        if points >= 7:
            return 1.0
        elif points >= 4:
            return 0.8
        elif points >= 1:
            return 0.2
        else:
            return 0.0

    return 0.0


def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    **kwargs
) -> dict:
    """
    Compute proof-based reward using IMO-Bench style grading (4-tier: 0/0.2/0.8/1.0).

    Args:
        solution_str: Model's generated solution (may contain <think> tags)
        ground_truth: Short answer (may be empty for proof-only problems)
        extra_info: Must contain 'question', 'reference_solution'

    Returns:
        dict with 'score' (0.0/0.2/0.8/1.0), 'acc' (bool)
    """
    if extra_info is None:
        extra_info = {}

    api_base = os.environ.get("LLM_GRADER_API_BASE", DEFAULT_API_BASE)
    api_key = os.environ.get("LLM_GRADER_API_KEY", DEFAULT_API_KEY)
    model = os.environ.get("LLM_GRADER_MODEL", DEFAULT_MODEL)

    question = extra_info.get("question", "") or extra_info.get("problem", "")
    reference_solution = extra_info.get("reference_solution", "")

    if not reference_solution:
        return {
            "score": 0.0,
            "acc": False,
            "error": "no_reference_solution"
        }

    student_solution = strip_think_tags(solution_str)

    if not student_solution:
        return {
            "score": 0.0,
            "acc": False,
            "error": "empty_response"
        }

    prompt = PROOF_GRADING_PROMPT.format(
        problem_statement=question,
        solution=reference_solution,
        student_answer=student_solution
    )
    messages = [{"role": "user", "content": prompt}]

    try:
        raw_response = _chat_complete_sync(api_base, api_key, model, messages)
        grading_response = strip_think_tags(raw_response)
        score = _parse_score(grading_response)

        return {
            "score": score,
            "acc": score == 1.0,
        }

    except Exception as e:
        print(f"[LLM Proof Grading IMOBench ERROR] {type(e).__name__}: {str(e)[:200]}")
        return {
            "score": 0.0,
            "acc": False,
        }
