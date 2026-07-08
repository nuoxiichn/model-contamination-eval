#!/usr/bin/env bash
# Qwen2.5-72B Scenario b —— 8 卡机 2 份数据并行。
#
# 72B fp16 ~145G，单份 device_map=auto 切到 4 卡（4×64G=256G 够）。8 卡起 2 份：
#   shard A: 卡 0-3 跑 c-eval + cmb
#   shard B: 卡 4-7 跑 cmmlu + mmlu-cf
# 每份内部对每题 24 排列做 batch 前向（seq_logprob_sums）。两份写不同 CSV，避免 race。
#
# 前置：72B 已下载到 HF_HOME；outputs/data/*.jsonl 已由 prepare_data.py 备好（正式版 --per-benchmark 300）。
#
# 用法（8 卡机）：
#   bash experiments/2026-07-08_perm_option_scenario_b/run_72b_2shard.sh
set -euo pipefail

export HF_HOME=${HF_HOME:-/mnt/public/code/chennuoxi/hf_cache}
export PYTHONPATH=src
HERE="experiments/2026-07-08_perm_option_scenario_b"
RUN="python3 $HERE/run_scenario_b.py --models qwen2.5-72b"

# MetaX 设备隔离用 MACA_VISIBLE_DEVICES（cu-bridge 兼容 CUDA_VISIBLE_DEVICES）
echo "[launch] shard A (cards 0-3): c-eval + cmb"
MACA_VISIBLE_DEVICES=0,1,2,3 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  $RUN --benchmarks c-eval cmb --csv-name scenario_b_72b_shardA.csv \
  > $HERE/outputs/results/shardA.log 2>&1 &
PID_A=$!

echo "[launch] shard B (cards 4-7): cmmlu + mmlu-cf"
MACA_VISIBLE_DEVICES=4,5,6,7 CUDA_VISIBLE_DEVICES=4,5,6,7 \
  $RUN --benchmarks cmmlu mmlu-cf --csv-name scenario_b_72b_shardB.csv \
  > $HERE/outputs/results/shardB.log 2>&1 &
PID_B=$!

echo "[wait] shard A pid=$PID_A, shard B pid=$PID_B …"
wait $PID_A; echo "[done] shard A"
wait $PID_B; echo "[done] shard B"

echo "[collect] 从 per-cell JSON 汇总统一 CSV"
python3 $HERE/collect_results.py
echo "[✓] 全部完成，见 $HERE/outputs/results/scenario_b.csv"
