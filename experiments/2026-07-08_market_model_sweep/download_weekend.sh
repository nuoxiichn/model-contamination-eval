#!/usr/bin/env bash
# 周末扩测模型下载：主流 gated trio + 免 gate 替代 trio。
# 关键设计：每个模型独立 || 容错，单个失败不阻断其余（gated 没配 token 时会失败，
# 但免 gate 的照常下完，run_sweep.py 里对缺失权重也会优雅跳过）。
#
# 前置（gated 必需，见 notes.md「HF token 前置」）：
#   1. export HF_TOKEN=hf_xxx           # 你自己的 read token，勿写进任何文件/commit
#   2. 在网页接受 3 个 license：
#      huggingface.co/meta-llama/Llama-3.1-8B
#      huggingface.co/mistralai/Mistral-7B-v0.3
#      huggingface.co/google/gemma-2-9b
#   注意：hf-mirror 对 gated 仍需真 token；若镜像取不到，去掉 HF_ENDPOINT 走官方源。
#
# 跑法（建议 tmux）： bash experiments/2026-07-08_market_model_sweep/download_weekend.sh
set -uo pipefail   # 注意：不加 -e，让单个失败不中断整脚本

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/public/code/chennuoxi/hf_cache
DL="/opt/conda/bin/hf download"

dl() {  # dl <repo> [extra args...]
  echo "=== 下载 $1 ==="
  $DL "$@" && echo "  OK $1" || echo "  !! 失败（跳过）：$1"
}

echo "###### 免 gate trio（无 token 也能下）######"
dl deepseek-ai/deepseek-llm-7b-base
dl deepseek-ai/deepseek-llm-7b-chat
dl 01-ai/Yi-1.5-9B
dl 01-ai/Yi-1.5-9B-Chat
# internlm2.5 已下架本 sweep（transformers 5.6.0 撞其老 remote code），换 Qwen2-7B 顶替：
dl Qwen/Qwen2-7B
dl Qwen/Qwen2-7B-Instruct

echo ""
echo "###### 新增 benchmark 数据集（2026-07-18；免 gate，自动缓存到 HF_HOME/datasets）######"
# math 已缓存；mmlu/gsm-plus/mgsm 首次需下。run_sweep.py 加载时也会自动下，这里预下更稳。
dl --repo-type dataset cais/mmlu
dl --repo-type dataset qintongli/GSM-Plus
dl --repo-type dataset juletxara/mgsm

echo ""
echo "###### 主流 gated trio（需 HF_TOKEN + 已接受 license）######"
if [ -z "${HF_TOKEN:-}" ]; then
  echo "!! HF_TOKEN 未设置，gated 三家会失败。先按 notes.md 配好 token 再重跑本脚本。"
fi
dl meta-llama/Llama-3.1-8B
dl meta-llama/Llama-3.1-8B-Instruct
dl mistralai/Mistral-7B-v0.3
dl mistralai/Mistral-7B-Instruct-v0.3
dl google/gemma-2-9b
dl google/gemma-2-9b-it

echo ""
echo "[done] 下载阶段结束。失败的家族 run_sweep.py 会自动跳过（记 load_error），不阻断其余。"
