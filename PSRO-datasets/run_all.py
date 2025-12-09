import subprocess
import sys

STEPS = [
    ("Generate proofs",        ["python", "generate_qwen3_proofs.py"]),
    ("Verification (GT)",      ["python", "oracle_label_gpt5_inplace.py"]),
    ("Meta verification",      ["python", "meta_verification_inplace.py"]),
    ("Data processing/flatten", ["python", "convert_dataset_format.py"]),
]


def run_step(desc, cmd):
    print(f"\n========== Starting step: {desc} ==========")
    print("Command:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Step failed: {desc}, return code {result.returncode}")
        sys.exit(result.returncode)
    print(f"========== Step finished: {desc} ==========\n")


def main():
    for desc, cmd in STEPS:
        run_step(desc, cmd)

    print("All steps have been executed.")


if __name__ == "__main__":
    main()
