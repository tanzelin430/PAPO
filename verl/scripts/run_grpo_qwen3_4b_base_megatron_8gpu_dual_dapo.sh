#!/bin/bash
set -x

# =================== PA_GRPO + DAPO: Dual-Objective with Length Control (Qwen3-4B-Base, 8 GPU) ===================
# Algorithm: A_total = A_out + lambda * A_proc
#   A_out: ORM+LLM-fallback rewards WITH overlong penalty (DAPO reward shaping)
#   A_proc: LLM proof grading PRM scores, normalized among correct responses only
#   correct_mask uses binary acc (unaffected by overlong penalty)
#
# DAPO features enabled:
#   - Decoupled clipping (clip_ratio_low=0.2, clip_ratio_high=0.28)
#   - Overlong reward shaping (linear penalty when response > max_resp_len - buffer)
#   - DAPORewardManager (handles overlong penalty in reward_tensor)
#
# Model: Qwen3-4B-Base (no thinking, no chat template)
# Parallelism: TP=2, PP=1, DP=4 (8x H200 GPUs)

# =================== Command Line Options ===================
EXTRA_ARGS=()

for arg in "$@"; do
    EXTRA_ARGS+=("$arg")
done

# =================== Megatron Environment Configuration ===================
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# export TIKTOKEN_ENCODINGS_BASE=/path/to/tiktoken_encodings  # Optional: set if offline

# Disable Ray memory monitor to prevent false OOM kills
export RAY_memory_monitor_refresh_ms=0

# PA_GRPO: Enable dual-objective scoring in reward routing
export USE_DUAL_OBJECTIVE=true
# ORM + LLM fallback for binary A_out signal
export USE_LLM_VERIFIER=true

# LLM Grader (GPT-OSS-20B, for both LLM-fallback and PRM 3-tier scoring)
export LLM_GRADER_API_BASE="${LLM_GRADER_API_BASE:-http://localhost:8000/v1}"
export LLM_GRADER_API_KEY="${LLM_GRADER_API_KEY:-dummy}"
export LLM_GRADER_MODEL="GPT-OSS-20B"

echo "=== PA_GRPO+DAPO: Qwen3-4B-Base Dual-Objective+Overlong 8GPU ==="

# NCCL/cuDNN paths for Megatron
export NCCL_HOME=$(python3 -c "import nvidia.nccl; print(nvidia.nccl.__path__[0])" 2>/dev/null || echo "")
export CUDNN_PATH=$(python3 -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])" 2>/dev/null || echo "")
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$CUDNN_PATH/lib:${LD_LIBRARY_PATH:-}

# =================== Data Configuration ===================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export WANDB_DIR=${SCRIPT_DIR}/../wandbQwen4BBaseDualDAPO
DATA_DIR="${SCRIPT_DIR}/../data"

TRAIN_FILE="${DATA_DIR}/train/train_20k_stratified.parquet"
EVAL_DIR="${DATA_DIR}/eval"

train_files="['${TRAIN_FILE}']"
val_files="['${EVAL_DIR}/olympiad.parquet','${EVAL_DIR}/aime2024.parquet','${EVAL_DIR}/aime2025.parquet','${EVAL_DIR}/aime2026.parquet','${EVAL_DIR}/gpqa_diamond.parquet','${EVAL_DIR}/minerva.parquet','${EVAL_DIR}/amc2023.parquet']"

# =================== Output and Checkpoint Configuration ===================
RESULTS_DIR=${RESULTS_DIR:-$(dirname "$SCRIPT_DIR")/output}
CHECKPOINT_DIR=${RESULTS_DIR}/checkpoints
mkdir -p ${CHECKPOINT_DIR}

# =================== Model Configuration ===================
MODEL_NAME=Qwen3-4B-Base
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-4B-Base}

# Custom chat template: passthrough for base model (no <|im_start|> etc.)
CUSTOM_CHAT_TEMPLATE="'{{ messages[0][\"content\"] }}'"

# =================== Logging Configuration ===================
WANDB_PROJECT=verl_math_rl
WANDB_EXPERIMENT_NAME=${MODEL_NAME}_megatron_8gpu_dual_dapo_20260315_151240

# =================== Parallelism (8 GPU: TP=2, PP=1, DP=4) ===================
NUM_GPUS=8
ACTOR_TP=2
ACTOR_PP=1
ROLLOUT_TP=2
REF_TP=2
REF_PP=1

# =================== Sequence length ===================
max_prompt_length=2048
max_response_length=8192

# =================== DAPO Overlong Buffer ===================
# Linear penalty when response exceeds (max_response_length - overlong_buffer_len)
# e.g., buffer=4096 → penalty starts at 4096 tokens, reaches -1.0 at 8192 tokens
overlong_buffer_len=4096
overlong_penalty_factor=1.0

# =================== NO CPU offload ===================
offload=False

# =================== Batch sizes ===================
train_batch_size=128
ppo_mini_batch_size=32

# =================== Start Training ===================
VERL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$VERL_DIR"

python3 -m verl.trainer.main_ppo \
    --config-path=$(pwd)/verl/trainer/config \
    --config-name='ppo_megatron_trainer.yaml' \
    algorithm.adv_estimator=grpo_dual \
    +algorithm.lambda_process=1.0 \
    data.train_files="${train_files}" \
    data.val_files="${val_files}" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation=right \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path=${BASE_MODEL} \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.custom_chat_template="${CUSTOM_CHAT_TEMPLATE}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=17000 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${REF_PP} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${REF_TP} \
    actor_rollout_ref.actor.checkpoint.save_contents='["model","optimizer","extra"]' \
    actor_rollout_ref.actor.checkpoint.load_contents='["model","optimizer","extra"]' \
    algorithm.use_kl_in_reward=False \
    critic.enable=False \
    reward_model.enable=False \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=true \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=true \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=20 \
    trainer.total_epochs=7 \
    trainer.val_before_train=True \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.resume_mode=auto \
    trainer.default_local_dir=${CHECKPOINT_DIR}/${WANDB_PROJECT}/${WANDB_EXPERIMENT_NAME} \
    "${EXTRA_ARGS[@]}"
