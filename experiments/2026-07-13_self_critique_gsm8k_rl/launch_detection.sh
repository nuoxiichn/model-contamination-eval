#!/usr/bin/env bash
# Self-Critique GSM8K RL-MIA 检测启动器（分片，MetaX）。
#
# 每张卡起一个进程，CUDA_VISIBLE_DEVICES=i 只见 1 卡，处理 questions[i::NSHARDS]。
# 全部 shard 跑完后自动跑 eval_metrics.py 出 member vs non-member AUC。
#
# 用法：
#   本机单卡（null baseline）:  bash .../launch_detection.sh 1
#   8 卡机（RL ckpt）:          bash .../launch_detection.sh 8
#
# 断点续跑：重跑跳过各 shard 已完成 id。检测哪个 ckpt 由 run.yaml 的 model.path 决定
# （原始 Instruct=null baseline；RL 训练后 ckpt=正结果），--out-subdir 区分结果目录。

set -u
cd /mnt/public/code/chennuoxi/model-contamination-eval

EXP=experiments/2026-07-13_self_critique_gsm8k_rl
NSHARDS=${1:-1}
SUBDIR=${2:-}                     # 输出子目录：区分 null vs rl ckpt 结果
GPU_ENV=CUDA_VISIBLE_DEVICES

export HF_HOME=/mnt/public/code/chennuoxi/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export PYTHONPATH=src

mkdir -p "$EXP/outputs"

echo "[launch] $(date) starting $NSHARDS shards (subdir='${SUBDIR:-<root>}')"
pids=()
for i in $(seq 0 $((NSHARDS-1))); do
  log="$EXP/outputs/${SUBDIR:+$SUBDIR-}shard_${i}.log"
  env "$GPU_ENV=$i" python3 "$EXP/run_self_critique.py" --shard "$i" --nshards "$NSHARDS" \
      --out-subdir "$SUBDIR" > "$log" 2>&1 &
  pids+=($!)
  echo "[launch] shard $i → GPU $i  pid=${pids[-1]}  log=$log"
done

echo "[launch] waiting for ${#pids[@]} shards..."
fail=0
for p in "${pids[@]}"; do
  wait "$p" || { echo "[launch] pid $p exited non-zero"; fail=1; }
done

if [ "$fail" -ne 0 ]; then
  echo "[launch] some shard failed — check outputs/*shard_*.log; NOT running eval."
  exit 1
fi

echo "[launch] all shards done → eval_metrics.py"
python3 "$EXP/eval_metrics.py" --subdir "$SUBDIR" 2>&1 | tee "$EXP/outputs/${SUBDIR:+$SUBDIR-}eval_stdout.log"
echo "[launch] $(date) complete."
