# PAPO on ROLL

Core modifications to the [ROLL](https://github.com/alibaba/ROLL) framework for running PAPO experiments.

## Modified Files

Apply these files on top of ROLL's codebase:

| File | Description |
|------|-------------|
| `configs/base_config.py` | Adds `grpo_dual` advantage estimator and `lambda_process` parameter |
| `utils/functionals.py` | `compute_grpo_dual_advantage()` — dual advantage (A_out + A_proc) |
| `pipeline/rlvr/rlvr_config.py` | Adds `use_dual_objective`, `prm_api_*` fields to `RewardConfig` |
| `pipeline/rlvr/rewards/math_rule_reward_worker.py` | Extends `MathRuleRewardWorker` with LLM-as-Judge PRM scoring |

## Setup

```bash
# Clone and install ROLL
git clone https://github.com/alibaba/ROLL.git
cd ROLL
pip install -e .
pip install -e mcore_adapter/

# Apply PAPO modifications (copy over ROLL's originals)
cp <PAPO_ROOT>/roll/configs/base_config.py         roll/configs/
cp <PAPO_ROOT>/roll/utils/functionals.py            roll/utils/
cp <PAPO_ROOT>/roll/pipeline/rlvr/rlvr_config.py    roll/pipeline/rlvr/
cp <PAPO_ROOT>/roll/pipeline/rlvr/rewards/math_rule_reward_worker.py \
                                                     roll/pipeline/rlvr/rewards/

# Copy example config
cp <PAPO_ROOT>/roll/examples/pa_grpo_qwen2.5_7b_megatron_8gpu.yaml \
   examples/pa_grpo_qwen2.5_7b/pa_grpo_config.yaml
```

## LLM Grader Setup

PAPO requires an LLM grader for process reward scoring. Any OpenAI-compatible endpoint works:

```bash
# Option 1: Self-hosted GPT-OSS-20B via vLLM
CUDA_VISIBLE_DEVICES=0,1 vllm serve openai/gpt-oss-20b \
  --tensor-parallel-size 2 --port 8000 --max-model-len 8192

# Then set in YAML config:
#   prm_api_url: "http://localhost:8000/v1"
#   prm_api_key: "your-key"
#   prm_model: "gpt-oss-20b"
```

## Training

### PAPO (Qwen2.5-7B-Base, 8 GPU Megatron)

```bash
cd ROLL
python examples/start_rlvr_pipeline.py \
  --config_path pa_grpo_qwen2.5_7b \
  --config_name pa_grpo_config
```

### Key Config Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `adv_estimator` | `grpo_dual` | PAPO dual advantage estimator |
| `lambda_process` | `1.0` | Weight for process advantage (A_total = A_out + λ·A_proc) |
| `use_dual_objective` | `true` | Enable PRM scoring in reward worker |
| `num_return_sequences_in_group` | `8` | Responses per prompt (group size) |
| `temperature` | `1.0` | Sampling temperature (no top_p/top_k for RL) |
| `rollout_batch_size` | `128` | Prompts per batch |
| `response_length` | `8192` | Max response length |
