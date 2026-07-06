#!/bin/bash
# ==============================================================================
# Canary 周末一条链：train(LoRA SFT) → merge(LoRA→full) → recall 召回测量。
# 跨两个仓库（LlamaFactory 训练/merge + model-contamination-eval 召回），
# 每步落 .done 标记，可任意重启续跑（train 自动 resume checkpoint-*）。
#
# 一键（tmux 内）：
#   bash experiments/2026-07-03_sft_canary/run_weekend.sh
#
# 避开 session14 的 paraphrase job（默认占 device 0/1）：默认只用 2-7 这 6 张卡。
# 到机器上先 mx-smi 看 0/1 是否已空；若全空可覆盖：
#   MACA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 bash .../run_weekend.sh
#
# 只跑某几步（train/merge/recall）：
#   STEPS="recall" bash .../run_weekend.sh      # 训练已完成，只重测召回
# ==============================================================================
set -uo pipefail   # 故意不 set -e：某步失败要能看清在哪断

LF_DIR="/mnt/public/code/chennuoxi/LlamaFactory"
MCE_DIR="/mnt/public/code/chennuoxi/model-contamination-eval"
BASE_MODEL="/mnt/public/model/huggingface/Qwen3-1.7B-Base"
ADAPTER_DIR="${LF_DIR}/saves/contam/canary_v1"
MERGED_DIR="${LF_DIR}/saves/contam/canary_v1_merged"

# ── GPU：默认避开 0/1（paraphrase job）。NPROC 默认=可见卡数。
export MACA_VISIBLE_DEVICES="${MACA_VISIBLE_DEVICES:-2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="${MACA_VISIBLE_DEVICES}"   # cu-bridge 兼容
_NGPU="$(awk -F',' '{print NF}' <<< "${MACA_VISIBLE_DEVICES}")"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${_NGPU}}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

STEPS="${STEPS:-train merge recall}"
LOG_DIR="${MCE_DIR}/experiments/2026-07-03_sft_canary/_weekend_logs"
mkdir -p "${LOG_DIR}"
TIMELINE="${LOG_DIR}/timeline.log"

_log() { echo "$(date -Iseconds) $*" | tee -a "${TIMELINE}"; }

_log "START canary weekend (cards=${MACA_VISIBLE_DEVICES} nproc=${NPROC_PER_NODE} steps='${STEPS}')"

# ── STEP 1: train ───────────────────────────────────────────────────────────
if [[ " ${STEPS} " == *" train "* ]]; then
  if [ -f "${MERGED_DIR}/config.json" ]; then
    _log "[skip] train：merged ckpt 已存在（${MERGED_DIR}）"
  else
    _log "[train] launch canary_v1（${NPROC_PER_NODE} 卡，自动 resume checkpoint-*）"
    cd "${LF_DIR}"
    # OVERWRITE_OUTPUT_DIR=false：被 kill 后重跑从最新 checkpoint-* 续训，不从头重来
    if OVERWRITE_OUTPUT_DIR=false \
       bash examples/contam_sft_gt/submit_qwen3_1.7b.sh canary_v1 \
         >> "${LOG_DIR}/train.log" 2>&1; then
      _log "[train] done → ${ADAPTER_DIR}"
    else
      _log "[train] FAILED exit=$? （见 ${LOG_DIR}/train.log），中止后续"
      exit 1
    fi
  fi
fi

# ── STEP 2: merge (LoRA adapter → full weights) ──────────────────────────────
if [[ " ${STEPS} " == *" merge "* ]]; then
  if [ -f "${MERGED_DIR}/config.json" ]; then
    _log "[skip] merge：${MERGED_DIR} 已存在"
  elif [ ! -d "${ADAPTER_DIR}" ]; then
    _log "[merge] SKIP：adapter 不存在（${ADAPTER_DIR}），训练未完成？"
    exit 1
  else
    _log "[merge] export LoRA → full → ${MERGED_DIR}"
    cd "${LF_DIR}"
    # merge 单卡即可；沿用 canary_v1.yaml 注释里的权威命令
    # DISABLE_VERSION_CHECK=1：8卡机 trl==0.24.0 撞 llamafactory 的 trl<=0.9.6 检查
    #   （训练 submit 脚本已内置跳过，故训练能过而 merge 挂——见 env-facts-8card-metax）
    if DISABLE_VERSION_CHECK=1 \
       MACA_VISIBLE_DEVICES="$(cut -d',' -f1 <<< "${MACA_VISIBLE_DEVICES}")" \
       llamafactory-cli export \
         --model_name_or_path "${BASE_MODEL}" \
         --adapter_name_or_path saves/contam/canary_v1 \
         --template qwen3 --finetuning_type lora \
         --export_dir saves/contam/canary_v1_merged --export_size 5 \
         >> "${LOG_DIR}/merge.log" 2>&1; then
      _log "[merge] done → ${MERGED_DIR}"
    else
      _log "[merge] FAILED exit=$? （见 ${LOG_DIR}/merge.log），中止后续"
      exit 1
    fi
  fi
fi

# ── STEP 3: recall（canary_v1 / clean / base 三 ckpt 召回曲线） ───────────────
if [[ " ${STEPS} " == *" recall "* ]]; then
  _log "[recall] measure canary_v1,clean,base"
  cd "${MCE_DIR}"
  if HF_ENDPOINT="${HF_ENDPOINT}" PYTHONPATH=src python3 \
       experiments/2026-07-03_sft_canary/run_recall.py \
       --ckpt canary_v1,clean,base --device cuda \
       >> "${LOG_DIR}/recall.log" 2>&1; then
    _log "[recall] done。结果见 outputs/2026-07-03_sft_canary/<ts>/recall_curve.csv"
  else
    _log "[recall] FAILED exit=$? （见 ${LOG_DIR}/recall.log）"
    exit 1
  fi
fi

_log "END canary weekend"
echo
echo "============================================================"
echo " canary 链结束。产物："
echo "   merged ckpt : ${MERGED_DIR}"
echo "   召回结果    : ${MCE_DIR}/outputs/2026-07-03_sft_canary/<ts>/"
echo "   日志        : ${LOG_DIR}/{train,merge,recall}.log + timeline.log"
echo " 成功判据：clean/base recall=0/100；canary 随 rep(1/3/10/30/100) 单调爬升"
echo "============================================================"
