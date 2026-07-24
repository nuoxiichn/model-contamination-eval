# Positive Controls：开源已知污染 anchor 池

> 状态：早期候选池经仓库整理后保留，尚未全部完成本仓库复测。
> 用途：方法灵敏度矩阵的输入。每个 anchor 有公开污染证据、可从 HF 下载、目标 benchmark 与本项目重叠。
> 使用规则：本表是**方法灵敏度的锚**，不是"外部竞品评审"清单。跑本 pipeline 得出的 anchor 污染信号 = 我们方法对已知污染的敏感度，反过来能用于阈值校准与方法有效性证明。

## A. 强证据 anchor（可直接用）

| # | 模型 | HF 路径 | 污染 benchmark | 证据 | 强度 |
|---|---|---|---|---|---|
| A1 | Qwen3-1.7B-Base | `Qwen/Qwen3-1.7B-Base` | GSM8K | 本项目已复现（v1 §2.1 Finding 1，guided p=9.9e-05）+ [CoDeC ICLR 2026](https://kaitchup.substack.com/p/did-the-model-see-the-benchmark-during) 独立验证 Qwen3 系列在 MMLU-Pro/AIME/GPQA/GSM8K 上有污染信号 | ⭐⭐⭐ 已复现，直接可用 |
| A2 | Mistral-7B-v0.1 | `mistralai/Mistral-7B-v0.1` | GSM8K, HellaSwag | [ConStat arXiv:2405.16281](https://arxiv.org/abs/2405.16281) 直接命名；[GSM1k (Scale AI 2024)](https://arxiv.org/abs/2405.00332) Mistral 家族在 GSM1k 上系统性下降 13pp | ⭐⭐⭐ 两个独立审计一致 |
| A3 | Phi-3-mini-4k-instruct | `microsoft/Phi-3-mini-4k-instruct` | GSM8K | [GSM1k paper](https://arxiv.org/abs/2405.00332) Phi 家族"almost every size"系统性 overfit；社区跟进讨论多 | ⭐⭐⭐ 论文点名 |
| A4 | Qwen-1.5-4B (Instruct) | `Qwen/Qwen1.5-4B-Chat` | GSM8K | [ConStat](https://arxiv.org/abs/2405.16281) sample-specific p < 10⁻² | ⭐⭐ 老版本，参考价值中等 |
| A5 | Llama-2-7B | `meta-llama/Llama-2-7b` | GSM8K, HellaSwag | [Skywork audit](https://arxiv.org/abs/2310.17589)（15 模型 × 6 QA benchmark 污染报告，1-45% 范围）+ [Rethinking Benchmark rephrased contamination](https://arxiv.org/abs/2311.04850) Llama-2 rephrased 训后 GSM8K 从 28.7→95.3 | ⭐⭐ HF gated |

## B. 弱证据 anchor（辅助）

| # | 模型 | 污染 benchmark | 证据 | 备注 |
|---|---|---|---|---|
| B1 | Llama-3-70B | ARC 类 | [LLM contamination 2026 guide](https://llm-stats.com/blog/research/what-is-a-contaminated-llm) 归纳 | 70B 太大，Phase 1 不用 |
| B2 | Qwen2.5-7B | SST-2, LIAR2 | [DCR arXiv:2507.11405](https://arxiv.org/abs/2507.11405) DCR=67.6% | 目标 benchmark 与本项目不重叠，仅参考 |

## C. 已知干净模型（阴性对照，验证 FPR）

| # | 模型 | HF 路径 | 特性 | 用途 |
|---|---|---|---|---|
| C1 | OLMo-2-1B | `allenai/OLMo-2-1124-7B`（或 1B 变体） | 训练数据全公开（Dolma），可 self-verify | 主阴性对照 |
| C2 | Pythia-2.8B | `EleutherAI/pythia-2.8b` | 训练数据 Pile 全公开，早于多数当代 benchmark 发布 | 时序上"不可能污染"的对照 |
| C3 | InstructLM-1.3B | `instruction-pretrain/InstructLM-1.3B` | [DCR paper](https://arxiv.org/abs/2507.11405) 明确报告 DCR=0% | 该论文自建 clean baseline |

## D. 建议优先级（按"证据强 × 下载方便 × 规模适中"排）

1. **A1 Qwen3-1.7B-Base × GSM8K** —— 首发 anchor（本项目已复现，Qwen3-1.7B-Base 权重也已在 `/mnt/public/code/chennuoxi/hf_cache/`）
2. **A2 Mistral-7B-v0.1 × GSM8K** —— 独立第二 anchor（跨模型家族验证）
3. **A3 Phi-3-mini × GSM8K** —— 第三 anchor（论文最直接点名）
4. **C2 Pythia-2.8B × GSM8K** —— 阴性对照（Pile 早于 GSM8K 发布，逻辑不可能污染）
5. **C1 OLMo-2-1B × MMLU-Pro** —— 第二阴性对照（覆盖 MC 场景）

**注**：以上 anchor 均以 GSM8K 为主，因为 GSM8K 是当前证据最多、方法学最成熟的场景。完成首轮复测后再决定是否扩到 MMLU / HumanEval。

## E. Post-RL anchor 池（暂空）

Post-RL 场景（GRPO 一轮可能抹除 SFT 阶段 MIA 信号，[arXiv:2510.02386](https://arxiv.org/abs/2510.02386)）需要单独的 anchor。候选包括 DeepSeek-R1-Distill、OpenR1、MiroThinker 系列；新增前需先定义可验证的训练成员标签。

## 版本纪律

- 本文档是 anchor 池 **快照**。方法出新论文、社区曝新污染案例时，更新本文档并补灵敏度矩阵
- 移除 anchor 需说明理由（"证据被撤" / "HF 权重下架"）
- 新增强证据 anchor（`⭐⭐⭐`）需要至少两个独立来源
