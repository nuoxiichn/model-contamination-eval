#!/usr/bin/env bash
# AIME member-题 GRPO 注入训练（第二步：训污染 ckpt）。
#
# 基于 zyh/slow_thinking 的 launch/qwen_grpo_demo.sh 改（MetaX 验证通过的 verl v0.7 GRPO）。
# 目标：让 Qwen2.5-7B-Instruct 在 30 道 member 题上 GRPO 训练、记住它们（policy 收敛/熵坍缩），
# 之后 self_critique 检测 member vs non-member 应出正 AUC。
#
# ============================ 用前必读 ============================
# 1. 这是 zyh verl 栈的【独立副本】(slow_thinking_sc)——训练副作用(__pycache__/日志/ckpt)
#    全落副本，不碰 zyh 原件。环境(容器镜像+patch+env_fix)仍按 slow_thinking 文档装。
#    ⚠️ 副本的 thirdparty/verl 是从 zyh 工作区拷的当前状态：开训前确认 MetaX patch
#       (0001-add-mx-support / 0002-transformer5x) 是否已在（apply --check 看冲突）。
# 2. 【头号风险】AIME 很难，Qwen2.5-7B-Instruct 正确率可能低 → reward 稀疏 → 学不动。
#    开训前先做 sanity check（见 train/README.md）：看 step:0 的 val math_dapo acc。
#    若 acc 长期≈0，改用简单题（论文 openr1_aime_easy）或加大 rollout.n / 只挑答得对的题。
# 3. 小数据集（30 题）+ 注入目标：total_epochs 设大、看 val acc→高 就停（早停）。
#    batch/显存参数按 verl 实际报错微调（已标 # TUNE）。
# ================================================================

set -x
set -u

# --- 路径（绝对，跨目录可用）---
REPO=/mnt/public/code/chennuoxi/model-contamination-eval
SLOW_THINKING=/mnt/public/code/chennuoxi/slow_thinking_sc
DATA_DIR="$REPO/experiments/2026-07-08_self_critique_aime_repro/outputs/grpo_trainset"
MODEL_PATH=/mnt/public/model/huggingface/Qwen2.5-7B-Instruct
OUT_DIR="$REPO/experiments/2026-07-08_self_critique_aime_repro/outputs/grpo_ckpt"
mkdir -p "$OUT_DIR"

# ⚠️ 不 cd 进副本根：其根目录有 byw 的 sitecustomize.py（NVIDIA web-agent 的 import 魔改钩子），
#    python -m 会自动 import cwd 下的 sitecustomize → 卡 import。改在中性目录跑，绝对路径引 verl。
cd "$OUT_DIR"

# --- MetaX 通信/运行环境（照搬 zyh demo，MetaX 必需）---
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
# PYTHONPATH 只保留 verl 本体：不含副本根、不含 train/（都可能藏 sitecustomize/__init__ 副作用）
export PYTHONPATH=$SLOW_THINKING/thirdparty/verl
# 强制不缓冲 stdout/stderr：否则 verl/Ray 的早期日志卡在缓冲区，看起来像"卡住"实则在跑
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
# 离线：模型/数据本地已有，禁掉联网拉取（避免卡在 HF 超时）
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# 绕开副本根目录的 sitecustomize.py（byw 为 NVIDIA web-agent 写的 import 魔改钩子，
# 在 MetaX 纯数学 GRPO 下无用且疑似卡 import）：不把根目录放进 PYTHONPATH + 禁 usersite
export PYTHONNOUSERSITE=1

EXP_NAME="qwen25_7b_aime_member_inject_$(date +%Y%m%d%H%M%S)"

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/val.parquet" \
    data.train_batch_size=30 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    custom_reward_function.path="$REPO/experiments/2026-07-08_self_critique_aime_repro/train/aime_boxed_reward.py" \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=30 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
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
    trainer.project_name='selfcritique_aime_inject' \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=1 \
    trainer.total_epochs=40 \
    trainer.default_local_dir="$OUT_DIR/${EXP_NAME}" \
    "$@"

# 说明（对齐论文 arXiv:2510.09259 超参 + 注入目标的调整）：
#   lr 1e-6 / rollout.n 8 / temperature 1.0 / max_response 3072≈论文4096  —— 对齐论文
#   train_batch_size=30 = 全部 member 题（# TUNE：verl 若要求整除关系报错，调这里 + mini/micro）
#   total_epochs=40（注入过拟合；论文 base 数据多用 2，我们 30 题需反复曝光才记住）
#   test_freq=1：每 epoch 出 val math_dapo acc → 盯 member 题正确率上升 = 记住的信号
#   训完：把最新 global_step ckpt 路径填进 ../run.yaml 的 model.path，重跑 launch_8card.sh + eval_metrics.py
