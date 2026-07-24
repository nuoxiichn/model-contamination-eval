#!/usr/bin/env bash
# Resume the market-model sweep with task-level parallelism on 8 MetaX GPUs.
# Single-model methods use one process/GPU. SPV uses four two-GPU workers.

set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
PYTHON=${PYTHON:-/opt/conda/bin/python}
RUNNER="$HERE/run_sweep.py"
RESULTS="$HERE/outputs/sweep_results.jsonl"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$HERE/logs/multi_gpu_$STAMP"

export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-/mnt/public/code/chennuoxi/hf_cache}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

mkdir -p "$LOG_DIR"

if pgrep -af '[r]un_sweep.py' >/dev/null; then
  echo "[fatal] run_sweep.py is already running; stop it before launching shards."
  pgrep -af '[r]un_sweep.py'
  exit 1
fi

if [[ -f "$RESULTS" ]]; then
  cp -p "$RESULTS" "$RESULTS.bak_pre_mgpu_$STAMP"
fi

declare -a PIDS=()
declare -a LABELS=()

stop_children() {
  trap - INT TERM
  if ((${#PIDS[@]})); then
    kill "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
  fi
  exit 130
}
trap stop_children INT TERM

start_worker() {
  local label=$1
  local visible_devices=$2
  local phase=$3
  shift 3
  local -a cmd=("$PYTHON" "$RUNNER" --phase "$phase")
  local model
  for model in "$@"; do
    cmd+=(--model "$model")
  done

  echo "[start] $label gpu=$visible_devices phase=$phase models=$*"
  (
    export CUDA_VISIBLE_DEVICES="$visible_devices"
    export MACA_VISIBLE_DEVICES="$visible_devices"
    "${cmd[@]}"
  ) >"$LOG_DIR/$label.log" 2>&1 &
  PIDS+=("$!")
  LABELS+=("$label")
}

wait_workers() {
  local phase=$1
  local failed=0
  local i status
  for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
      status=0
    else
      status=$?
      failed=1
    fi
    echo "[exit] ${LABELS[$i]} status=$status log=$LOG_DIR/${LABELS[$i]}.log"
  done
  PIDS=()
  LABELS=()
  if ((failed)); then
    echo "[warn] one or more $phase workers exited non-zero; continuing so other shards can finish."
  fi
}

echo "[multi-gpu] logs=$LOG_DIR"
echo "[multi-gpu] results=$RESULTS"

# Current single-model backlog: seven unfinished models plus two isolated retries.
start_worker single-gpu0-gemma-it 0 single gemma-2-9b-it
start_worker single-gpu1-deepseek-base 1 single deepseek-7b
start_worker single-gpu2-deepseek-chat 2 single deepseek-7b-chat
start_worker single-gpu3-yi-base 3 single yi-1.5-9b
start_worker single-gpu4-yi-chat 4 single yi-1.5-9b-chat
start_worker single-gpu5-qwen2-base 5 single qwen2-7b
start_worker single-gpu6-qwen2-instruct 6 single qwen2-7b-instruct
start_worker single-gpu7-retries 7 single llama-3.1-8b gemma-2-9b
wait_workers single

# SPV keeps target on local cuda:0 and its base reference on local cuda:1.
# Pending counts are balanced 7/7/7/4 across the four workers.
start_worker spv-gpu01 0,1 spv qwen2-7b-instruct olmo-2-7b-instruct
start_worker spv-gpu23 2,3 spv llama-3.1-8b-instruct qwen2.5-7b-instruct
start_worker spv-gpu45 4,5 spv gemma-2-9b-it mistral-7b-instruct
start_worker spv-gpu67 6,7 spv deepseek-7b-chat yi-1.5-9b-chat
wait_workers spv

echo "[done] multi-GPU sweep finished"
echo "[done] summarize with: PYTHONPATH=src $PYTHON $HERE/summarize.py"
