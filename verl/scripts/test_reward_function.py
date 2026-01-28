#!/usr/bin/env python3
"""
Test verl reward functions with ROLL data format.
"""

import json
import sys
sys.path.insert(0, '/mnt/shared-storage-user/tanzelin-p/PSRO4math/verl')

from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score import math_dapo

# Test cases from ROLL data
test_cases = [
    {
        "response": """Step 1: Let the second side be x. Then the first side is 4x.
Step 2: The third side is 20.
Step 3: For a valid triangle: x + 20 > 4x, so 20 > 3x, meaning x < 6.67
Step 4: Also: 4x + x > 20, so 5x > 20, meaning x > 4
Step 5: So x can be 5 or 6. For maximum perimeter, x = 6.
Step 6: Perimeter = 6 + 24 + 20 = 50

ANSWER:
\\boxed{50}""",
        "ground_truth": "50",
        "expected": 1.0
    },
    {
        "response": """Let me solve this step by step.
The answer is \\boxed{42}""",
        "ground_truth": "50",
        "expected": -1.0  # wrong answer
    },
    {
        "response": """I think the answer is 50 but I'm not sure.""",
        "ground_truth": "50",
        "expected": -1.0  # no boxed format
    },
    {
        "response": """Step 1: Calculate
Step 2: The result is \\boxed{2}""",
        "ground_truth": "2",
        "expected": 1.0
    },
]

print("=" * 60)
print("Testing math_dapo.compute_score directly")
print("=" * 60)

for i, case in enumerate(test_cases):
    result = math_dapo.compute_score(
        solution_str=case["response"],
        ground_truth=case["ground_truth"]
    )
    score = result["score"] if isinstance(result, dict) else result
    status = "PASS" if score == case["expected"] else "FAIL"
    print(f"\nTest {i+1}: {status}")
    print(f"  Ground truth: {case['ground_truth']}")
    print(f"  Expected: {case['expected']}, Got: {score}")
    if isinstance(result, dict):
        print(f"  Pred: {result.get('pred', 'N/A')}")

print("\n" + "=" * 60)
print("Testing default_compute_score with different data_source values")
print("=" * 60)

data_sources_to_test = [
    "math_dapo",
    "math",
    "math_curriculum",  # ROLL's tag - will fail!
    "aime2024",
]

test_response = "The answer is \\boxed{50}"
test_gt = "50"

for ds in data_sources_to_test:
    try:
        result = default_compute_score(
            data_source=ds,
            solution_str=test_response,
            ground_truth=test_gt
        )
        score = result["score"] if isinstance(result, dict) else result
        print(f"\n{ds}: score = {score}")
    except NotImplementedError as e:
        print(f"\n{ds}: NOT IMPLEMENTED - {e}")
    except Exception as e:
        print(f"\n{ds}: ERROR - {e}")

print("\n" + "=" * 60)
print("Conclusion")
print("=" * 60)
print("""
The 'math_curriculum' data_source is NOT supported by default_compute_score.

Solutions:
1. Use 'math_dapo' or 'math' as data_source when converting data
2. Or implement a custom reward function

Recommended: Use 'math_dapo' as data_source since it uses the same
\\boxed{} format and provides correct/incorrect binary rewards.
""")
