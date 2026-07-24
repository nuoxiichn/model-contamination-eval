#!/usr/bin/env bash
# 下载 sweep 需要但缓存里还没有的两个 instruct 模型（走 hf-mirror，免 gate）。
# base 已在缓存：Qwen/Qwen2.5-7B（hub）、C1_olmo-2-7b（本地）。
#
# 跑法： bash experiments/2026-07-08_market_model_sweep/download_models.sh
set -euo pipefail

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/public/code/chennuoxi/hf_cache

# 新版 huggingface_hub 废弃了 huggingface-cli，改用 `hf download`
# 不加 --exclude：新版 CLI 会把第二个模式误当成显式文件名（见 2026-07-08 踩坑）
DL="/opt/conda/bin/hf download"

echo "[1/2] Qwen2.5-7B-Instruct ..."
$DL Qwen/Qwen2.5-7B-Instruct

echo "[2/2] OLMo-2-1124-7B-Instruct ..."
$DL allenai/OLMo-2-1124-7B-Instruct

# control 集 mmlu-cf（gsm1k 本地已有，其余按需）
echo "[3/3] mmlu-cf (control for mmlu-pro) ..."
$DL microsoft/MMLU-CF --repo-type dataset || echo "  (mmlu-cf 下载失败可先只跑有 gsm1k control 的部分)"

echo "[done] 下载完成。可跑 run_sweep.py。"
