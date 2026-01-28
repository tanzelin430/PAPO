#!/usr/bin/env python3
"""Test naive_dapo reward function with math_verify fallback."""

import sys
sys.path.insert(0, '/mnt/shared-storage-user/tanzelin-p/PSRO4math/verl')

from verl.utils.reward_score import default_compute_score

print("=" * 60)
print("Testing default_compute_score with math__ data sources")
print("=" * 60)

test_cases = [
    {
        "name": "Correct answer with boxed",
        "data_source": "math__deepscaler_preview",
        "response": "Step 1: Calculate\nStep 2: Result\n\\boxed{50}",
        "ground_truth": "50",
        "expected_correct": True,
    },
    {
        "name": "Wrong answer with boxed",
        "data_source": "math__merged_deduped_dapo_or1_dataset",
        "response": "Step 1: Calculate\n\\boxed{42}",
        "ground_truth": "50",
        "expected_correct": False,
    },
    {
        "name": "Equivalent fraction",
        "data_source": "math__deepscaler_preview",
        "response": "The answer is \\boxed{\\frac{1}{2}}",
        "ground_truth": "0.5",
        "expected_correct": True,  # math_verify should handle this
    },
    {
        "name": "LaTeX expression",
        "data_source": "math__deepscaler_preview",
        "response": "Therefore \\boxed{2\\sqrt{3}}",
        "ground_truth": "2\\sqrt{3}",
        "expected_correct": True,
    },
    {
        "name": "No boxed format",
        "data_source": "math__deepscaler_preview",
        "response": "The answer is 50",
        "ground_truth": "50",
        "expected_correct": False,  # Should fail due to no \boxed{}
    },
]

for case in test_cases:
    print(f"\n--- {case['name']} ---")
    print(f"data_source: {case['data_source']}")
    print(f"ground_truth: {case['ground_truth']}")

    try:
        result = default_compute_score(
            data_source=case["data_source"],
            solution_str=case["response"],
            ground_truth=case["ground_truth"],
            extra_info={}
        )

        if isinstance(result, dict):
            score = result.get("score", result.get("acc", 0))
            acc = result.get("acc", score > 0)
        else:
            score = result
            acc = score > 0

        is_correct = acc if isinstance(acc, bool) else score > 0
        status = "PASS" if is_correct == case["expected_correct"] else "FAIL"

        print(f"Result: {result}")
        print(f"Expected correct: {case['expected_correct']}, Got: {is_correct} -> {status}")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
