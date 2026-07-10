#!/usr/bin/env bash
# perm_option 剂量-响应（72B LoRA）—— 8 卡机 tmux 长跑启动器。
#
# 单一长作业：72B base（device_map=auto 切 8 卡，冻结）+ 一个 LoRA 适配器，一趟在注入集
# 上训到 max_epochs（config，默认 10），每个 epoch 后测 leak_fraction → leak-vs-epoch 曲线。
# 逐 epoch 落盘（outputs/leak_vs_epoch.json），中途可看曲线。
#
# 前置（RUNBOOK.md 有硬 gate）：
#   1) peft 已装（trl 环境通常自带；缺则 pip install peft）
#   2) Qwen2.5-72B 已缓存（scenario_b 下过，136G）+ pile-10k 已缓存 —— 均已确认
#   3) mmlu-cf.jsonl 存在（scenario_b 导出，复用）
#
# 用法（8 卡机）：
#   bash experiments/2026-07-10_perm_option_dose/launch_dose_tmux.sh
#   tmux attach -t perm_dose
set -euo pipefail
cd /mnt/public/code/chennuoxi/model-contamination-eval

export HF_HOME=${HF_HOME:-/mnt/public/code/chennuoxi/hf_cache}
export PYTHONPATH=src
HERE="experiments/2026-07-10_perm_option_dose"
SESSION=perm_dose
mkdir -p "$HERE/outputs"

read -r -d '' BODY <<'EOF' || true
set -euo pipefail
cd /mnt/public/code/chennuoxi/model-contamination-eval
export HF_HOME=${HF_HOME:-/mnt/public/code/chennuoxi/hf_cache}
export PYTHONPATH=src
HERE="experiments/2026-07-10_perm_option_dose"

echo "[gate] 检查 peft …"
python3 -c "import peft; print('peft', peft.__version__)" || { echo "[FATAL] 缺 peft：pip install peft"; exit 1; }

echo "[run] 72B LoRA 剂量-响应（8 卡 device_map=auto，5 剂量串行）"
MACA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python3 $HERE/run_dose_lora.py 2>&1 | tee "$HERE/outputs/dose.log"
echo "[✓] dose 完成，见 $HERE/outputs/dose_summary.json"
EOF

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[!] tmux session '$SESSION' 已存在；先 tmux kill-session -t $SESSION 再重启"
  exit 1
fi
tmux new-session -d -s "$SESSION" "bash -lc '$BODY; echo; echo [exit code=\$?]; exec bash'"
echo "[launch] tmux session '$SESSION' 已启动"
echo "         tmux attach -t $SESSION   # 看实时进度"
echo "         日志：$HERE/outputs/dose.log"
