#!/usr/bin/env bash
# Phase 1 sanity check：端到端跑通最小 pipeline
#
# 选 1 个已知污染 (默认 gsm8k) + 1 个干净参照 (默认 livebench)
# 对 base 和 target checkpoint 各跑 Oren + family_diff，对比信号。
#
# 用法：
#   bash scripts/sanity_check.sh \
#       --model-base   /path/to/llama3-base \
#       --model-target /path/to/llama3-sft

set -euo pipefail

MODEL_BASE=""
MODEL_TARGET=""
BENCHMARK_DIRTY="gsm8k"
BENCHMARK_CLEAN="livebench"
OUTPUTS="./outputs/sanity-$(date +%Y%m%d-%H%M%S)"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-base) MODEL_BASE="$2"; shift 2;;
        --model-target) MODEL_TARGET="$2"; shift 2;;
        --benchmark-dirty) BENCHMARK_DIRTY="$2"; shift 2;;
        --benchmark-clean) BENCHMARK_CLEAN="$2"; shift 2;;
        --outputs) OUTPUTS="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if [[ -z "$MODEL_BASE" || -z "$MODEL_TARGET" ]]; then
    echo "Usage: $0 --model-base PATH --model-target PATH [--benchmark-dirty NAME] [--benchmark-clean NAME]"
    exit 1
fi

mkdir -p "$OUTPUTS"

echo "=== Phase 1 sanity check ==="
echo "  base    = $MODEL_BASE"
echo "  target  = $MODEL_TARGET"
echo "  dirty   = $BENCHMARK_DIRTY"
echo "  clean   = $BENCHMARK_CLEAN"
echo "  outputs = $OUTPUTS"
echo

# 国内镜像
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
echo "  HF_ENDPOINT = $HF_ENDPOINT"
echo

uv run python -m model_contamination.cli sanity-check \
    --model-base "$MODEL_BASE" \
    --model-target "$MODEL_TARGET" \
    --benchmark-dirty "$BENCHMARK_DIRTY" \
    --benchmark-clean "$BENCHMARK_CLEAN" \
    --outputs "$OUTPUTS" \
    | tee "$OUTPUTS/run.log"

echo
echo "[done] results at $OUTPUTS"
