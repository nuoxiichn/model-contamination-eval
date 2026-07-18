#!/usr/bin/env bash
# GSM8K member-题 GRPO 注入训练（忠实 self_critique RL policy-collapse 验证）。
#
# 基于 aime 版（experiments/2026-07-08_.../train/launch_aime_grpo.sh）改，三处关键调整：
#   1. 数据 → GSM8K member 半（build_gsm8k_mia_split.py 产出）
#   2. reward → verl 原生 gsm8k（data_source=gsm8k，`#### 数字` 匹配），**去掉自定义 reward**
#   3. 关 KL（use_kl_loss=False / kl_loss_coef=0 / use_kl_in_reward=False）——放任 policy
#      往 member 题的固定轨迹漂移/collapse，不往参考模型拉回
#
# ============================ 用前必读 ============================
# 1. 在 8 卡机 vllm_metax 容器里跑（同 aime；verl+vllm_metax 环境依赖见 aime train/README）。
# 2. GSM8K 简单，7B Instruct 正确率 ~85% → reward 应正常上升（不再有 aime 的 reward 稀疏坑）。
#    开训看 step:0 val gsm8k acc：应明显 >0（≈0.7-0.9）。若仍≈0 → prompt/reward 格式没对上，先查。
# 3. 注入成功判据：train reward↑ + val acc↑ + actor/entropy↓（policy collapse 的直接证据）。
#    entropy 明显下降 = self_critique 检测的信号真的被造出来了。
# ================================================================

set -x
set -u

REPO=/mnt/public/code/chennuoxi/model-contamination-eval
SLOW_THINKING=/mnt/public/code/chennuoxi/slow_thinking_sc
DATA_DIR="$REPO/experiments/2026-07-13_self_critique_gsm8k_rl/outputs/grpo_trainset"
MODEL_PATH=/mnt/public/model/huggingface/Qwen2.5-7B-Instruct
OUT_DIR="$REPO/experiments/2026-07-13_self_critique_gsm8k_rl/outputs/grpo_ckpt"
mkdir -p "$OUT_DIR"

# 中性目录跑（避开副本根的 sitecustomize，见 aime 脚本注释）
cd "$OUT_DIR"

# --- MetaX 通信/运行环境（照搬 zyh demo）---
export MCCL_NET_GDR_LEVEL=7
export MCCL_MAX_NCHANNELS=16
export MCCL_P2P_LEVEL=MX
export MCCL_LIMIT_RING_LL_THREADTHRESHOLDS=1
export MCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=${MCCL_SOCKET_IFNAME}
export MCCL_IB_HCA=mlx5_0,mlx5_1
export CUDA_LAUNCH_BLOCKING=0
export MACA_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH=$SLOW_THINKING/thirdparty/verl
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1

EXP_NAME="qwen25_7b_gsm8k_member_inject_$(date +%Y%m%d%H%M%S)"

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/val.parquet" \
    data.train_batch_size=50 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=50 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='selfcritique_gsm8k_inject' \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=1 \
    trainer.total_epochs=20 \
    trainer.default_local_dir="$OUT_DIR/${EXP_NAME}" \
    "$@"

# 说明：
#   data_source=gsm8k → verl default_compute_score 原生 gsm8k reward（#### 匹配），无 custom_reward。
#   关 KL：use_kl_loss=False + kl_loss_coef=0 + use_kl_in_reward=False → 放任 collapse。
#   train_batch_size=50 = 全部 member 题；total_epochs=20（注入过拟合，盯 entropy↓ 早停）。
#   训完：run.yaml 的 model.path 换成最新 global_step ckpt（merge/转 HF 若需），
#         bash launch_detection.sh 8 rl → eval 出 member vs non-member AUC。
