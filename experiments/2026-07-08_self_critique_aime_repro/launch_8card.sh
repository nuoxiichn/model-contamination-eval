#!/usr/bin/env bash
# Self-Critique AIME24/25 复现：8 卡分片启动器（MetaX C500 ×8）。
#
# 每张卡起一个进程，CUDA_VISIBLE_DEVICES=i 只见 1 卡，处理 questions[i::8]。
# 全部 shard 跑完后自动跑 eval_metrics.py 出 F1/AUC。
#
# 用法：bash experiments/2026-07-08_self_critique_aime_repro/launch_8card.sh
#
# 注意：
# - MetaX 兼容 CUDA_VISIBLE_DEVICES（MACA 层）；若本机 GPU 掩码变量不同，改下面 GPU_ENV。
# - 60 题（AIME24+25）分 8 卡 ≈ 每卡 7-8 题，两趟 1024/2048 贪心生成，单卡约 20-40min。
# - 断点续跑：重跑本脚本会跳过各 shard 已完成的 id。

set -u
cd /mnt/public/code/chennuoxi/model-contamination-eval

EXP=experiments/2026-07-08_self_critique_aime_repro
NSHARDS=8
GPU_ENV=CUDA_VISIBLE_DEVICES     # MetaX 若用别的掩码变量，改这里

export HF_HOME=/mnt/public/code/chennuoxi/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=src

mkdir -p "$EXP/outputs"

echo "[launch] $(date) starting $NSHARDS shards"
pids=()
for i in $(seq 0 $((NSHARDS-1))); do
  log="$EXP/outputs/shard_${i}.log"
  env "$GPU_ENV=$i" python3 "$EXP/run_self_critique_aime.py" --shard "$i" --nshards "$NSHARDS" \
      > "$log" 2>&1 &
  pids+=($!)
  echo "[launch] shard $i → GPU $i  pid=${pids[-1]}  log=$log"
done

echo "[launch] waiting for ${#pids[@]} shards..."
fail=0
for p in "${pids[@]}"; do
  wait "$p" || { echo "[launch] pid $p exited non-zero"; fail=1; }
done

if [ "$fail" -ne 0 ]; then
  echo "[launch] some shard failed — check outputs/shard_*.log; NOT running eval yet."
  exit 1
fi

echo "[launch] all shards done → running eval_metrics.py"
python3 "$EXP/eval_metrics.py" 2>&1 | tee "$EXP/outputs/eval_stdout.log"
echo "[launch] $(date) complete."
