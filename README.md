# model-contamination-eval

模型级 benchmark 污染检测 pipeline。对同一个模型的 base / SFT / RLHF 三类 checkpoint 跑统一一套检测代码，按训练阶段参数化分支；共用方法两阶段都跑，差值作为污染归因信号。

## 解决什么问题

LLM benchmark 越来越不可信：
- **base 阶段** 可能在预训练时见过测试集
- **SFT 阶段** 可能混入了 benchmark 风格的指令数据
- **RLHF 阶段** 可能在 reward 信号引导下记住了答题模式

本仓库提供一套统一 pipeline，对每个 benchmark 输出：

1. **可信度报告**（红/黄/绿裁决）—— 这个 benchmark 在这个 checkpoint 上的分数还能不能信
2. **阶段归因报告** —— 如果污染了，主要责任在哪个训练阶段

## 不解决什么

- 不做数据级检测（n-gram、FM-index、embedding match）—— 另有独立仓库
- 不反推到单条 SFT 数据 —— 样本级筛选在数据准备阶段做
- 不输出"数据配方反馈" —— 数据治理团队的职责，不在这里
- 不替代训练时的污染防御（dropout、weight decay 等）

## 方法学概览

| 阶段 | 主信号 | 备注 |
| --- | --- | --- |
| base | Oren 分片排列检验、Guided Instruction、Min-K%++ 辅助 | Oren 提供数学 FPR 保证 |
| SFT | SPV-MIA（用同源 base 做 reference）、MemLens、Paraphrase Stress、同族对照 ΔScore | AUC 显著优于无 reference 的 MIA |
| SFT + RLHF | Self-Critique、Paraphrase、同族迁移检测 | GRPO 一轮即抹除 SFT 阶段的 MIA 信号，传统方法在此退化 |


## 目标 benchmark

对齐记忆张量内部模型评测 P1 注册名单。

**预训练阶段（base checkpoint）**：MMLU-Pro / GPQA-Diamond / MATH / EvalPlus / LiveCodeBench / MGSM / MMMLU

**SFT 阶段 text 类**：IFEval / MMLU-Pro / GPQA-Diamond / AIME 2025 或 MATH-500 / LiveCodeBench / SimpleQA Verified 或 SimpleQA / LiveBench

**暂不纳入**：
- Action 类 benchmark —— 方法学调研未覆盖，留待后续版本
- 不在 P1 名单内的 benchmark（MMLU / GSM8K / HumanEval 等） —— `configs/benchmarks.yaml` 保留为实验/对照参考，不进可信度报告主表

部分 benchmark 没有强同族对照（GPQA-Diamond / IFEval / SimpleQA），任务孤岛诊断对它们失效，只能依赖 Oren + Paraphrase 等单方法信号。

## 使用前提

| 项 | 必需 | 缺失后果 |
| --- | --- | --- |
| 同源 base + SFT 双 checkpoint | 是 | SPV-MIA 与所有差分信号不可用，阶段归因失效 |
| 模型暴露 logprobs | 是 | Oren / Min-K%++ / SPV-MIA 全部不可用 |
| 同族对照集（MMLU-CF / GSM1k / LBPP 等）至少一半可获取 | 是 | 任务孤岛诊断失效 |
| 中间层 hidden states 访问权限 | MemLens 才需要 | 不能跑 MemLens，其余不影响 |
| 已知污染的 positive control 模型 | 用于阈值标定 | 当前缺失，红/黄/绿降级为**相对排名** |

第三方黑盒模型只能跑 Guided Instruction / Paraphrase / 同族对照 ΔScore 三件套，CLI 会自动跳过白盒方法并在报告中标注"白盒方法不可用"。

## 当前阶段

| Phase | 内容 | 状态 |
| --- | --- | --- |
| 1 | benchmark 注册表 + Oren + 同族 ΔScore + 可信度报告（ranked list 形式） | 🔵 进行中 |
| 2 | SPV-MIA + Paraphrase Stress Test + canary 注入 + 阶段归因报告 | ⚪ 待开始 |
| 3 | MemLens + Self-Critique + 阶段归因报告精细化 | ⚪ 待开始 |
| 4 | 对照集季度更新 + 历史归因结果库 | ⚪ 持续 |

未实现的方法在代码中是 `NotImplementedError` 占位，不会悄悄返回 0 或假数据。

## 快速开始

```bash
# 安装
uv sync --extra hf --extra dev

# 国内镜像（避免 HuggingFace 下载慢）
export HF_ENDPOINT=https://hf-mirror.com

# 下载 benchmark
uv run python scripts/download_benchmarks.py \
    --benchmarks mmlu gsm8k mmlu-cf gsm1k livebench

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

## 重要限制（用之前请读）

- **正在标定阶段**：positive control 尚未到位，可信度报告**只输出 ranked list**，不出绝对红/黄/绿。绝对裁决在阈值标定完成后启用。
- **单 benchmark 单方法的 MIA AUC 不能当"证明污染"的唯一证据**——立场参考 [SaTML 2025 论文](https://arxiv.org/abs/2506.17871)。
- **没有 pre-SFT checkpoint 时不出阶段归因**，CLI 会显式标 `归因不可用`，不要硬猜。

## 参考

- SPV-MIA: [arXiv:2311.06062](https://arxiv.org/abs/2311.06062)
- MemLens: [arXiv:2509.20909](https://arxiv.org/abs/2509.20909)
- Fragility of LRM Detection（RL 抹除 MIA 信号）: [arXiv:2510.02386](https://arxiv.org/abs/2510.02386)
- Impact of Post-training on Contamination: [arXiv:2601.06103](https://arxiv.org/abs/2601.06103)
- ConStat: [arXiv:2405.16281](https://arxiv.org/abs/2405.16281)
