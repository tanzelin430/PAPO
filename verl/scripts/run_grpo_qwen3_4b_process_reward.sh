#!/bin/bash
set -x

# =================== LLM Process Reward Training with Internal Reward Model ===================
# This script uses LLM-based process reward for math RL training.
# It evaluates solution quality based on process, not just final answers.
#
# Architecture:
# - Actor Model: Qwen3-1.7B (GPU 0)
# - Reward Model: Qwen3-14B (GPU 1, TP=1, vLLM server)
#
# Scoring: IMO 4-level (7/6/1/0) normalized to 0-1
# - Correct (7) → 1.0
# - Almost (6) → 0.857
# - Partial (1) → 0.143
# - Incorrect (0) → 0.0

# =================== Environment Configuration ===================
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=0
export WANDB_MODE=offline

# =================== Data Configuration ===================
DATA_DIR="/mnt/shared-storage-user/tanzelin-p/PSRO4math/verl/data"
# Use process reward format data (includes reference_solution)
TRAIN_FILE="/mnt/shared-storage-user/ma4agi-gpu/data/dataset/NuminaMath-1.5-RL-Verifiable/train_verl_format_process.parquet"
EVAL_DIR="${DATA_DIR}/eval"

train_files="['${TRAIN_FILE}']"
# Evaluation files: Math-500, AIME24, AIME25, AMC23, Minerva, Olympiad, GPQA-Diamond
val_files="['${EVAL_DIR}/math__math_500.parquet', '${EVAL_DIR}/aime2024.parquet', '${EVAL_DIR}/aime2025.parquet', '${EVAL_DIR}/amc2023.parquet', '${EVAL_DIR}/minerva.parquet', '${EVAL_DIR}/olympiad.parquet', '${EVAL_DIR}/gpqa_diamond.parquet']"

# Batch settings (minimal for testing on 2 GPU colocate mode)
train_prompt_bsz=4
n_resp_per_prompt=2
train_prompt_mini_bsz=2

# =================== Output and Checkpoint Configuration ===================
RESULTS_DIR=/mnt/shared-storage-user/tanzelin-p/verl_output
CHECKPOINT_DIR=${RESULTS_DIR}/checkpoints
mkdir -p ${CHECKPOINT_DIR}

# =================== Model Configuration ===================
# Actor model: Qwen3-1.7B
MODEL_NAME=Qwen3-1.7B
BASE_MODEL=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e

# Reward model: Qwen3-1.7B (same as actor for testing on 2 GPU)
REWARD_MODEL=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e

# =================== Logging Configuration ===================
WANDB_PROJECT=verl_math_rl
WANDB_EXPERIMENT_NAME=${MODEL_NAME}_process_reward_internal_rm

# =================== GRPO Training Parameters ===================
adv_estimator=grpo

use_kl_in_reward=False
use_kl_loss=False
kl_loss_coef=0.0
kl_loss_type=low_var_kl

clip_ratio_low=0.2
clip_ratio_high=0.2

# Sequence length limits (reduced for testing)
max_prompt_length=1024
max_response_length=2048

# Hardware Platform (2 GPU total, colocate mode)
# Actor and RM share the same resource pool to avoid verl bug with n_gpus=1
num_nodes=1
n_gpus_per_node=2

# Reward model resource allocation (separate resource pool)
rm_n_gpus_per_node=1
rm_nnodes=1

EPOCHS=1

use_dynamic_bsz=True

max_seq_length=$((max_prompt_length + max_response_length))

actor_seq_multiplier=4
rollout_seq_multiplier=4
actor_ppo_max_token_len=$((max_seq_length * actor_seq_multiplier))
rollout_log_prob_max_token_len=$((max_seq_length * rollout_seq_multiplier))

# Sampling parameters - CRITICAL: use temperature=1.0, NO top_p/top_k for proper exploration
temperature=1.0
val_temperature=0.7

# Tensor parallel for rollout (actor)
gen_tp=1

offload=False
gpu_memory_utilization=0.5

# =================== Start GRPO Training with Internal Reward Model ===================
echo "=========================================="
echo "LLM Process Reward Training with Internal Reward Model"
echo "Actor Model: ${BASE_MODEL}"
echo "Reward Model: ${REWARD_MODEL}"
echo "Training data: ${TRAIN_FILE}"
echo "Checkpoints: ${CHECKPOINT_DIR}/${WANDB_PROJECT}/${WANDB_EXPERIMENT_NAME}"
echo "=========================================="

python3 -m verl.trainer.main_ppo \
    hydra.run.dir=${CHECKPOINT_DIR}/${WANDB_PROJECT}/${WANDB_EXPERIMENT_NAME}/hydra_outputs \
    hydra.sweep.dir=${CHECKPOINT_DIR}/${WANDB_PROJECT}/${WANDB_EXPERIMENT_NAME}/hydra_multirun \
    hydra.job.chdir=False \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.gamma=1.0 \
    algorithm.lam=0.95 \
    data.train_files="${train_files}" \
    data.val_files="${val_files}" \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    data.prompt_key=prompt \
    data.truncation='right' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.filter_overlong_prompts=True \
    data.shuffle=False \
    data.trust_remote_code=True \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.optim.min_lr_ratio=0. \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${rollout_log_prob_max_token_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.model.path=${BASE_MODEL} \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    reward_model.enable=True \
    reward_model.use_reward_loop=True \
    reward_model.enable_resource_pool=False \
    reward_model.model.path=${REWARD_MODEL} \
    reward_model.model.trust_remote_code=True \
    reward_model.n_gpus_per_node=${rm_n_gpus_per_node} \
    reward_model.nnodes=${rm_nnodes} \
    reward_model.rollout.name=vllm \
    reward_model.rollout.tensor_model_parallel_size=1 \
    reward_model.rollout.gpu_memory_utilization=0.2 \
    reward_model.rollout.prompt_length=4096 \
    reward_model.rollout.response_length=2048 \
    reward_model.rollout.max_num_seqs=16 \
    reward_model.rollout.enforce_eager=False \
    reward_model.rollout.free_cache_engine=True \
    reward_model.reward_manager=naive \
    reward_model.num_workers=8 \
    custom_reward_function.path=verl/utils/reward_score/llm_process_reward.py \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXPERIMENT_NAME} \
    trainer.val_before_train=True \
    trainer.nnodes=${num_nodes} \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.test_freq=10 \
    trainer.save_freq=100 \
    trainer.total_epochs=${EPOCHS} \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.resume_mode=auto \
    trainer.default_local_dir=${CHECKPOINT_DIR}/${WANDB_PROJECT}/${WANDB_EXPERIMENT_NAME} \
    "$@"
