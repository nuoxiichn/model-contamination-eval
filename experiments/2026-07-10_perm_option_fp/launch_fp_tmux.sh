#!/usr/bin/env bash
# perm_option FP —— 8 卡机全矩阵 tmux 启动器（无人值守长跑）。
#
# 矩阵：{1.5B, 7B, 72B} × {mmlu-cf, synth-random, synth-fluent-mismatch}，全部纯推理
# （无训练）。小模型单卡快；72B fp16 ~145G 用 device_map=auto 切 4 卡，8 卡起 2 份并行。
#
# 前置：
#   1) 开发机已跑 make_fp_controls.py（outputs/data/*.jsonl 就位）
#   2) 72B 已下载到 HF_HOME（scenario_b 已下过则复用）
#
# 用法（8 卡机）：
#   bash experiments/2026-07-10_perm_option_fp/launch_fp_tmux.sh
#   tmux attach -t perm_fp        # 看进度
set -euo pipefail
cd /mnt/public/code/chennuoxi/model-contamination-eval

export HF_HOME=${HF_HOME:-/mnt/public/code/chennuoxi/hf_cache}
export PYTHONPATH=src
HERE="experiments/2026-07-10_perm_option_fp"
SESSION=perm_fp
LOG="$HERE/outputs/results"
mkdir -p "$LOG"

read -r -d '' BODY <<'EOF' || true
set -euo pipefail
cd /mnt/public/code/chennuoxi/model-contamination-eval
export HF_HOME=${HF_HOME:-/mnt/public/code/chennuoxi/hf_cache}
export PYTHONPATH=src
HERE="experiments/2026-07-10_perm_option_fp"
RUN="python3 $HERE/run_fp.py"

echo "[1/3] 小模型 1.5B + 7B（卡 0，全 3 对照，串行）"
MACA_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
  $RUN --models qwen2.5-1.5b qwen2.5-7b > "$HERE/outputs/results/small.log" 2>&1
echo "[1/3] done small models"

echo "[2/3] 72B 2 份数据并行（卡 0-3 / 4-7）"
MACA_VISIBLE_DEVICES=0,1,2,3 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  $RUN --models qwen2.5-72b --benchmarks mmlu-cf synth-random \
  --csv-name fp_72b_shardA.csv > "$HERE/outputs/results/72b_shardA.log" 2>&1 &
PID_A=$!
MACA_VISIBLE_DEVICES=4,5,6,7 CUDA_VISIBLE_DEVICES=4,5,6,7 \
  $RUN --models qwen2.5-72b --benchmarks synth-fluent-mismatch \
  --csv-name fp_72b_shardB.csv > "$HERE/outputs/results/72b_shardB.log" 2>&1 &
PID_B=$!
wait $PID_A; echo "[2/3] shard A done"
wait $PID_B; echo "[2/3] shard B done"

echo "[3/3] 汇总 per-cell JSON → fp.csv"
python3 $HERE/collect_fp.py
echo "[✓] FP 全矩阵完成，见 $HERE/outputs/results/fp.csv"
EOF

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[!] tmux session '$SESSION' 已存在；先 tmux kill-session -t $SESSION 再重启"
  exit 1
fi
tmux new-session -d -s "$SESSION" "bash -lc '$BODY; echo; echo [exit code=\$?]; exec bash'"
echo "[launch] tmux session '$SESSION' 已启动"
echo "         tmux attach -t $SESSION   # 看实时进度"
echo "         日志：$LOG/small.log / 72b_shardA.log / 72b_shardB.log"
