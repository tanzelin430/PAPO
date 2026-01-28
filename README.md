# MathPRM: Process Reward Model for Mathematical Reasoning

Mathematical reasoning RL training with **LLM-based Process Reward Model (PRM)** using the verl framework.

## Project Goal

Traditional Outcome Reward Models (ORM) only provide sparse binary signals (correct/wrong final answer). This project implements **Process Reward Models** that grade the reasoning process step-by-step, providing denser reward signals for better credit assignment in RL training.

**Two Reward Approaches:**
1. **Rule-based Reward**: Uses `math_verify` library to check final answer correctness (ORM baseline)
2. **LLM-as-Judge Reward**: Uses LLM to grade solution process with IMO-style scoring (PRM)

## Project Structure

```
PSRO4math/
├── verl/                    # RL training framework (embedded source)
│   ├── scripts/             # Training scripts
│   │   ├── run_grpo_qwen3_4b_base_math.sh      # Rule-based reward (8 GPU)
│   │   └── run_grpo_qwen3_4b_process_reward.sh # LLM process reward (2 GPU)
│   ├── data/eval/           # Evaluation datasets (parquet)
│   └── verl/utils/reward_score/
│       ├── naive_dapo.py            # Rule-based math reward
│       └── llm_process_reward.py    # LLM-as-Judge reward
├── ROLL/                    # Legacy RL framework (git submodule, optional)
└── requirements.txt         # Python dependencies
```

## Environment Setup

**Tested versions:** Python 3.12, PyTorch 2.9.1, CUDA 12.8, vLLM 0.14.0, flash-attn 2.8.3

**Hardware:** 2-8 GPUs (H100/A100 recommended)

```bash
# Clone repository
git clone https://github.com/tanzelin430/PSRO4math.git
cd PSRO4math

# Create conda environment
conda create -n verl python=3.12 -y
conda activate verl

# Install vLLM (includes PyTorch and CUDA)
pip install vllm==0.14.0

# Install flash-attention (find matching wheel at https://github.com/Dao-AILab/flash-attention/releases)
pip install flash-attn --no-build-isolation

# Install verl
cd verl && pip install -e .

# Install extra dependencies
cd .. && pip install -r requirements.txt
```

## Running Experiments

### 1. Rule-based Reward (ORM Baseline)

Uses `math_verify` library to check if final answer matches ground truth.

**Training (8 GPU):**
```bash
cd verl
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=offline
bash scripts/run_grpo_qwen3_4b_base_math.sh
```

**Key Settings:**
- Model: Qwen3-4B (Instruct)
- Dataset: NuminaMath-1.5-RL-Verifiable (131k samples)
- Reward: Binary (1.0 if correct, 0.0 otherwise)
- Sampling: temperature=1.0 (critical for RL exploration)

### 2. LLM-as-Judge Process Reward (PRM)

Uses LLM to grade solution process with IMO-style scoring.

**Training (2 GPU, for testing):**
```bash
cd verl
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_MODE=offline
bash scripts/run_grpo_qwen3_4b_process_reward.sh
```

**Scoring System:**
| Score | Meaning | Normalized |
|-------|---------|------------|
| 7 | Correct solution | 1.0 |
| 6 | Almost correct (minor issues) | 0.857 |
| 1 | Partial progress | 0.143 |
| 0 | Incorrect | 0.0 |

**Architecture:**
```
Actor (Qwen3-1.7B/4B) → generates solution
         ↓
Reward Model (Qwen3-14B) → grades process → IMO score → normalized reward
         ↓
GRPO Training
```

## Evaluation Benchmarks

| Dataset | Samples | Type |
|---------|---------|------|
| MATH-500 | 500 | Competition math |
| AIME 2024 | 30 | AMC/AIME |
| AIME 2025 | 30 | AMC/AIME |
| AMC 2023 | 40 | AMC/AIME |
| Olympiad | 674 | International olympiad |
| Minerva | 272 | Scientific/math |
| GPQA-Diamond | 198 | STEM multiple choice |

Evaluation runs automatically during training (configurable via `test_freq`).

## Key Configuration Notes

### Sampling Parameters (Critical!)

**DO NOT** use `top_p`, `top_k`, or `repetition_penalty` for RL training:

```bash
# WRONG - limits exploration, breaks RL
temperature=0.7
top_p=0.8
top_k=20

# CORRECT - allows proper exploration
temperature=1.0
# (no top_p, top_k, or repetition_penalty)
```

### Qwen3 Model Notes

- Use `Qwen3-XB` (Instruct), NOT `Qwen3-XB-Base`
- Disable thinking mode: `enable_thinking=False`
- Base models don't understand chat format and produce garbage

## WandB Logging

Training logs are saved in offline mode by default:

```bash
# Sync logs to WandB
cd verl
wandb sync wandb/offline-run-*
```

## License

MIT
