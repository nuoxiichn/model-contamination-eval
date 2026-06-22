# model-contamination-eval

模型级污染检测 pipeline，对应方案文档：[`../docs/solutions/模型级污染检测-两阶段复用方案.md`](../docs/solutions/模型级污染检测-两阶段复用方案.md)

## 快速开始

```bash
# 安装
uv sync --extra hf --extra dev

# 国内镜像（避免 HuggingFace 下载慢）
export HF_ENDPOINT=https://hf-mirror.com

# 下载 benchmark
uv run python scripts/download_benchmarks.py --benchmarks mmlu gsm8k mmlu-cf gsm1k livebench

# Phase 1 sanity check：选 1 个已知污染 + 1 个干净对照
bash scripts/sanity_check.sh \
    --model-base   /path/to/llama3-base \
    --model-target /path/to/llama3-sft \
    --benchmark-dirty gsm8k \
    --benchmark-clean livebench

# 单 benchmark 跑全方法
uv run python -m model_contamination.cli detect \
    --model /path/to/checkpoint \
    --stage sft \
    --benchmark mmlu
```

## Phase 1 范围（必读）

只实现两件事，先打通 plumbing：

1. **Oren 分片排列检验**（数据集级，灰盒，唯一有 FPR 数学保证）
2. **同族对照 ΔScore**（黑盒，任务孤岛特征诊断）

其余方法的接口已经定义但实现是 `NotImplementedError`，Phase 2+ 逐步填上。

## Phase 进度

| Phase | 内容 | 状态 |
| --- | --- | --- |
| 1 | benchmark registry + Oren + 同族 ΔScore + 可信度报告 | 🔵 进行中 |
| 2 | SPV-MIA + Paraphrase Stress Test + canary 注入 + 阶段归因 | ⚪ 待开始 |
| 3 | MemLens + Self-Critique + 数据源 tag + 配方反馈报告 | ⚪ 待开始 |
| 4 | 对照集季度更新 + 历史归因库 | ⚪ 持续 |

## 参考

- 方案：[`../docs/solutions/模型级污染检测-两阶段复用方案.md`](../docs/solutions/模型级污染检测-两阶段复用方案.md)
- 调研（pretraining）：[`../docs/research/contamination/Benchmark 污染检测调研报告.md`](../docs/research/contamination/Benchmark%20污染检测调研报告.md)
- 调研（SFT）：[`../docs/research/contamination/SFT 污染检测调研报告.md`](../docs/research/contamination/SFT%20污染检测调研报告.md)
