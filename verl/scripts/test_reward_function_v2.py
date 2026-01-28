#!/usr/bin/env python3
"""
Test verl reward functions with ROLL data format - detailed analysis.
"""

import json
import sys
sys.path.insert(0, '/mnt/shared-storage-user/tanzelin-p/PSRO4math/verl')

from verl.utils.reward_score import math_dapo, math_verify

print("=" * 60)
print("Analyzing math_dapo.compute_score behavior")
print("=" * 60)

# Test with ROLL's actual format: "ANSWER:\n\\boxed{...}"
test_cases = [
    # ROLL format with "ANSWER:" label
    {
        "name": "ROLL format (with ANSWER:)",
        "response": """Step 1: Calculate something
Step 2: More calculation
ANSWER:
\\boxed{50}""",
        "ground_truth": "50",
    },
    # Just \boxed{} without "ANSWER:"
    {
        "name": "Just boxed (no ANSWER:)",
        "response": """Step 1: Calculate something
Step 2: Result is \\boxed{50}""",
        "ground_truth": "50",
    },
    # With "Answer:" (lowercase)
    {
        "name": "Lowercase answer:",
        "response": """The answer is Answer: 50""",
        "ground_truth": "50",
    },
]

for case in test_cases:
    print(f"\n--- {case['name']} ---")
    print(f"Response (last 100 chars): ...{case['response'][-100:]}")

    result = math_dapo.compute_score(
        solution_str=case["response"],
        ground_truth=case["ground_truth"]
    )
    print(f"Result: {result}")

print("\n" + "=" * 60)
print("Testing math_verify.compute_score (uses math-verify library)")
print("=" * 60)

try:
    for case in test_cases[:2]:
        print(f"\n--- {case['name']} ---")
        result = math_verify.compute_score(
            model_output=case["response"],
            ground_truth=case["ground_truth"]
        )
        print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")
    print("math-verify library may not be installed")

print("\n" + "=" * 60)
print("Checking ROLL data answer format")
print("=" * 60)

# Read actual ROLL data
with open('/mnt/shared-storage-user/tanzelin-p/rl_data/math_curriculum_train.jsonl', 'r') as f:
    for i, line in enumerate(f):
        if i >= 2:
            break
        item = json.loads(line)
        msg = item['messages'][0]['content']
        print(f"\n--- Sample {i+1} ---")
        # Find output format instruction
        if "Output format:" in msg:
            format_start = msg.find("Output format:")
            print(f"Output format instruction:\n{msg[format_start:]}")
        print(f"Ground truth: {item['ground_truth']}")
