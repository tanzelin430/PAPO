# PAPO: Process-Aware Policy Optimization

[![Dataset](https://img.shields.io/badge/HuggingFace-NuminaMath--20k--Stratified-yellow)](https://huggingface.co/datasets/Artemis0430/NuminaMath-20k-Stratified)
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b)](https://arxiv.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

Official implementation of **"Stabilizing Rubric Integration Training via Decoupled Advantage Normalization"**.

PAPO integrates process-level evaluation into GRPO through **decoupled advantage normalization**, addressing two limitations of existing reward designs for math reasoning RL:

- **ORM (Outcome Reward Model)** evaluates only final-answer correctness, treating all correct responses identically. As models improve, advantage signal exhausts (zero-ratio ~69%).
- **PRM (Process Reward Model)** offers richer supervision, but directly using PRM scores causes **reward hacking** -- models exploit verbosity to inflate scores while accuracy collapses.

PAPO resolves both by decomposing advantage into two independently normalized components:

```
A_total = A_out + A_proc

A_out:  ORM binary (0/1), standard GRPO normalization over all responses
A_proc: PRM rubric (0/0.5/1), normalized ONLY among correct responses
```

This ensures that `A_out` anchors training on correctness while `A_proc` differentiates reasoning quality without distorting the outcome signal.

## Key Results

<table>
<tr>
<td width="58%"><img src="figures/fig1a_training_curves.png" width="100%"></td>
<td width="42%"><img src="figures/fig1b_competition_math.png" width="100%"></td>
</tr>
<tr>
<td colspan="2" align="center"><em>Figure 1: (a) OlympiadBench training curves. (b) Accuracy on competition math benchmarks.</em></td>
</tr>
</table>

**Figure 1(a)** shows OlympiadBench accuracy throughout training on a 7B base model. ORM (blue) peaks at 46.3% around step 750 and then declines due to signal exhaustion — as the model improves, more response groups become uniformly correct, producing zero advantage and zero gradient. PRM Only (red) collapses catastrophically to 1.3% after step 600, driven by reward hacking: the model learns to inflate PRM scores through verbosity rather than correct reasoning. The naive multiplicative combination ORM x PRM (purple) tracks ORM closely but fails to surpass it, confirming that simply multiplying the two signals within a single normalization pass suppresses the process component. PAPO (green) breaks through the ORM roofline and continues improving to 51.3%, demonstrating that decoupled advantage normalization successfully integrates process-level evaluation without the instabilities of direct PRM use.

**Figure 1(b)** confirms that PAPO's gains generalize across competition math benchmarks. On AIME 2024, AIME 2025, and OlympiadBench, PAPO consistently outperforms both ORM and ORM x PRM, with the largest improvements on harder benchmarks where signal exhaustion is most severe.

<p align="center">
  <img src="figures/fig3_method_overview.png" width="90%">
  <br>
  <em>Figure 3: Overview of PAPO.</em>
</p>

**Figure 3** illustrates the PAPO pipeline. Given a prompt, the policy generates G responses. Each response receives two reward signals: a binary outcome reward (ORM) and a rubric-based process reward (PRM, only for correct responses). The key design is **decoupled normalization**: the outcome advantage A_out is normalized over all G responses via standard GRPO, while the process advantage A_proc is normalized exclusively among the correct subset. This correct-subset normalization prevents incorrect responses from exploiting high PRM scores, and ensures that A_proc provides a meaningful quality signal even when all responses are correct (where A_out would be zero). The final advantage A_total = A_out + A_proc combines both components with equal weight.

## Repository Structure

```
PAPO/
├── verl/                                # verl framework (modified for PAPO)
│   ├── verl/
│   │   ├── trainer/ppo/
│   │   │   ├── core_algos.py            # PAPO advantage computation (grpo_dual)
│   │   │   ├── ray_trainer.py           # Training loop + dual metrics logging
│   │   │   └── metric_utils.py          # Advantage distribution metrics
│   │   ├── utils/reward_score/
│   │   │   ├── __init__.py              # Dual-objective reward routing
│   │   │   ├── llm_proof_grading.py     # PRM 3-tier rubric (LLM-as-Judge)
│   │   │   ├── llm_answer_grading.py    # LLM answer verification (fallback)
│   │   │   └── prime_math/              # ORM rule-based scorer
│   │   └── trainer/config/algorithm.py  # Config extensions
│   ├── scripts/                         # Training scripts (all models/variants)
│   ├── data/eval/                       # Evaluation datasets (parquet)
│   └── setup.py
├── paper/                               # LaTeX source
├── figures/                             # Experiment plots
└── requirements.txt
```

### Key Modified Files (vs. vanilla verl)

| File | Change |
|------|--------|
| `core_algos.py` | `grpo_dual` and `grpo_dual_fullnorm` advantage estimators |
| `ray_trainer.py` | Dual advantage branch + `dual_objective/*` metrics |
| `metric_utils.py` | Advantage distribution metrics (`adv_distribution/zero_ratio`) |
| `reward_score/__init__.py` | Dual-objective reward routing with LLM fallback |
| `llm_proof_grading.py` | LLM-as-Judge PRM prompt (3-tier: 0/0.5/1.0) |
| `algorithm.py` | `lambda_process` config parameter |

## Setup

**Requirements:** Python 3.12, 8x GPUs (H100/H200 recommended), CUDA 12.x

```bash
# Clone
git clone https://github.com/PAPO-anonymous/PAPO.git
cd PAPO

# Create environment
conda create -n papo python=3.12 -y
conda activate papo

# Install vLLM (includes PyTorch + CUDA)
pip install vllm==0.14.0

# Install flash-attention
pip install flash-attn --no-build-isolation

# Install verl
cd verl && pip install -e . && cd ..

# Install extra dependencies
pip install -r requirements.txt
```

## LLM Grader Setup

PAPO requires an LLM grader for both answer verification (LLM fallback) and process evaluation (PRM). We use [GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b) served via vLLM:

```bash
# Launch grader on 2 GPUs (separate from training GPUs)
CUDA_VISIBLE_DEVICES=0,1 vllm serve openai/gpt-oss-20b \
  --tensor-parallel-size 2 --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.85
```

Set environment variables for training scripts:
```bash
export LLM_GRADER_API_BASE="http://localhost:8000/v1"
export LLM_GRADER_API_KEY="your-key"
export LLM_GRADER_MODEL="gpt-oss-20b"
```

## Training

### PAPO (Qwen2.5-7B-Base, 8 GPU)

```bash
cd verl
bash scripts/run_grpo_qwen2.5_7b_base_megatron_8gpu_dual_lambda1.sh
```

### ORM Baseline

```bash
bash scripts/run_grpo_qwen2.5_7b_base_megatron_8gpu_baseline.sh
```

### PRM Only (for comparison)

```bash
bash scripts/run_grpo_qwen2.5_7b_base_megatron_8gpu_prm.sh
```

### Ablation: Full Normalization

```bash
bash scripts/run_grpo_qwen2.5_7b_base_megatron_8gpu_fullnorm.sh
```

### Ablation: ORM x PRM (Multiplicative)

```bash
bash scripts/run_grpo_qwen2.5_7b_base_megatron_8gpu_mult.sh
```

### Other Model Scales

Scripts are provided for multiple models:

| Model | Scripts |
|-------|---------|
| Qwen2.5-3B | `scripts/run_grpo_qwen2.5_3b_megatron_8gpu_{baseline,dual,prm}.sh` |
| Qwen2.5-7B-Base | `scripts/run_grpo_qwen2.5_7b_base_megatron_8gpu_*.sh` (main experiments) |
| Qwen2.5-14B | `scripts/run_grpo_qwen2.5_14b_megatron_8gpu_{baseline,dual,prm}.sh` |
| Qwen3-4B-Base | `scripts/run_grpo_qwen3_4b_base_megatron_8gpu_*.sh` |

### Configuration

Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `USE_DUAL_OBJECTIVE` | Enable PAPO dual-objective scoring | `false` |
| `USE_LLM_VERIFIER` | Enable LLM answer verification fallback | `true` |
| `USE_PRM_REWARD` | Pure PRM baseline mode | `false` |
| `USE_MULT_REWARD` | ORM x PRM multiplicative ablation | `false` |
| `LLM_GRADER_API_BASE` | LLM grader endpoint | `http://localhost:8000/v1` |
| `LLM_GRADER_MODEL` | LLM grader model name | `GPT-OSS-20B` |

Key training hyperparameters (algorithm config):

| Parameter | Value | Note |
|-----------|-------|------|
| `algorithm.adv_estimator` | `grpo_dual` | PAPO advantage estimator |
| `algorithm.lambda_process` | `1.0` | Equal weight for A_out and A_proc |
| `temperature` | `1.0` | **Critical**: no top_p/top_k for RL |
| `n` (rollouts per prompt) | `8` | Group size for GRPO |
| `train_batch_size` | `128` | Prompts per batch |
| `max_response_length` | `8192` | Max tokens per response |

## Evaluation

Evaluation runs automatically during training (controlled by `test_freq`). Supported benchmarks:

| Dataset | Samples | Key Metric |
|---------|---------|------------|
| OlympiadBench | 674 | Primary benchmark |
| MATH-500 | 500 | Competition math |
| AIME 2024/2025/2026 | 30 each | AMC/AIME |
| GPQA-Diamond | 198 | STEM reasoning |

Training metrics logged to WandB:
- `critic/reward/mean` -- average reward
- `response_length/mean` -- average response length
- `adv_distribution/zero_ratio` -- fraction of zero-advantage samples (signal exhaustion indicator)
- `dual_objective/*` -- PAPO-specific metrics (A_out, A_proc, process_active_ratio)

## Method Details

### Advantage Computation

```python
# Per group of 8 responses to the same prompt:

# Step 1: A_out -- standard GRPO on binary ORM scores
A_out = (scores_rule - mean(scores_rule)) / std(scores_rule)

# Step 2: A_proc -- PRM scores, normalized ONLY among correct responses
correct_mask = (scores_rule == 1)
if sum(correct_mask) >= 2:
    prm_correct = scores_prm[correct_mask]
    A_proc[correct_mask] = (prm_correct - mean(prm_correct)) / std(prm_correct)
# Wrong answers: A_proc = 0 (no double punishment)

# Step 3: Combine
A_total = A_out + A_proc
```

### Why Correct-Subset Normalization?

- **Prevents reward hacking**: Wrong answers cannot gain positive advantage from high PRM scores
- **Preserves outcome signal**: A_out and A_proc are independently normalized, no interference
- **Sustains learning signal**: Even in all-correct groups (A_out = 0), A_proc still differentiates quality

### PRM Rubric (3-tier)

The PRM evaluates reasoning quality of **correct** solutions using a rubric:

| Score | Meaning |
|-------|---------|
| 1.0 | Fully correct reasoning, all steps clear |
| 0.5 | Generally correct, minor gaps or omissions |
| 0.0 | Significant errors in reasoning despite correct answer |

## Citation

```bibtex
@article{tan2025papo,
  title={Stabilizing Rubric Integration Training via Decoupled Advantage Normalization},
  author={Tan, Zelin and Yu, Zhouliang and Lin, Bohan and Geng, Zijie and Geng, Hejia and Zhang, Yudong and Zhang, Mulei and Chen, Yang and Hu, Shuyue and Yin, Zhenfei and Zhang, Chen and Bai, Lei},
  year={2025}
}
```

## License

This project builds on [verl](https://github.com/volcengine/verl) (Apache 2.0). Our modifications are also released under the Apache 2.0 License.

## Acknowledgements

- [verl](https://github.com/volcengine/verl) -- RL training framework
- [GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b) -- LLM grader
- [Qwen](https://github.com/QwenLM/Qwen2.5) -- Base models
