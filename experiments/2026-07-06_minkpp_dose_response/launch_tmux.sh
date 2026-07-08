#!/usr/bin/env bash
# Min-K%++ 剂量-响应 tmux 守卫启动器（下班后无人值守）。
#
# 背景：单卡 MetaX C500 64GB。启动时 codec_overnight 的 finetune 正占 ~39GB。
# 本作业也是 Pythia-2.8b 全参 finetune（含 AdamW fp32 状态 ~33GB），并行会 OOM。
# 故先轮询显存，等空闲 >= FREE_NEED_MIB（codec 释放）再启动 run_dose.py。
#
# 之前串行链失败的教训：`while pgrep -f run_spec.py` 会匹配到循环自身的命令行，
# 永不退出 → dose 从未启动。这里改用 mx-smi 读显存，不匹配任何进程名。

set -u
cd /mnt/public/code/chennuoxi/model-contamination-eval

FREE_NEED_MIB=35000          # dose finetune 需 ~33GB，留余量
POLL_SEC=120
LOG=experiments/2026-07-06_minkpp_dose_response/tmux_run.log

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/public/code/chennuoxi/hf_cache
export PYTHONPATH=src

free_mib() {
  # 解析 mx-smi 的 "used/total MiB"，输出空闲 MiB
  local line used total
  line=$(mx-smi 2>/dev/null | grep -oE '[0-9]+/[0-9]+ MiB' | head -1)
  used=${line%%/*}
  total=$(echo "$line" | grep -oE '/[0-9]+' | tr -d '/')
  [ -n "${used:-}" ] && [ -n "${total:-}" ] && echo $(( total - used )) || echo 0
}

{
  echo "[guard] start $(date)  waiting for free>=${FREE_NEED_MIB}MiB"
  while :; do
    f=$(free_mib)
    echo "[guard] $(date +%H:%M:%S) free=${f}MiB"
    [ "$f" -ge "$FREE_NEED_MIB" ] && break
    sleep "$POLL_SEC"
  done
  echo "[guard] enough VRAM free → launching run_dose.py at $(date)"
  python3 experiments/2026-07-06_minkpp_dose_response/run_dose.py
  echo "[guard] run_dose.py exited code=$? at $(date)"
} 2>&1 | tee -a "$LOG"
