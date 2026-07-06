#!/bin/bash
# 单 anchor 启动器：把所有 env / 绝对路径 / -u 无缓冲 / tee 日志封装好，
# tmux 里只需 `bash launch_anchor.sh <MODEL> <CARD>`，命令短、无脆弱长串。
#
#   bash launch_anchor.sh A1 3     # A1 跑在 card 3
#   bash launch_anchor.sh A3 5     # A3 自动加 --trust-remote-code
#
# 跑完/报错后 session 不自动关（sleep infinity 挂住），方便 attach 回看。
set -uo pipefail
MODEL="${1:?usage: launch_anchor.sh MODEL CARD}"
CARD="${2:?usage: launch_anchor.sh MODEL CARD}"

REPO="/mnt/public/code/chennuoxi/model-contamination-eval"
LOG="${REPO}/experiments/2026-07-03_anchor_sensitivity/${MODEL}.log"
cd "${REPO}" || { echo "cd ${REPO} failed"; sleep infinity; }

EXTRA=""
[ "${MODEL}" = "A3" ] && EXTRA="--trust-remote-code"   # Phi-3 需要

export MACA_VISIBLE_DEVICES="${CARD}"
export CUDA_VISIBLE_DEVICES="${CARD}"                    # cu-bridge 兼容
export HF_ENDPOINT="https://hf-mirror.com"
export HF_DATASETS_CACHE="/mnt/public/code/chennuoxi/hf_cache"
export PYTHONPATH="src"

echo "[launch] $(date -Iseconds) model=${MODEL} card=${CARD}" | tee "${LOG}"
python3 -u experiments/2026-07-03_anchor_sensitivity/run_anchor.py \
    --model "${MODEL}" ${EXTRA} >> "${LOG}" 2>&1
echo "[launch] python exit=$? at $(date -Iseconds)" | tee -a "${LOG}"
echo "[launch] session 保留（sleep infinity）；确认无误后可 tmux kill-session。"
sleep infinity
