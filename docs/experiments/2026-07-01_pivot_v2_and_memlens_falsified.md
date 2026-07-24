# 方案 pivot v2 + MemLens 证伪（2026-07-01）

## 两件事，同一天发生

### 1. 方案 v1 → v2 pivot（用户决策）

三条决策：

| 维度 | v1 | v2 |
|---|---|---|
| 阶段 | base / SFT / RLHF | 预训练 / 后训练 |
| 报告 | 可信度 + 阶段归因 两份 | 可信度 一份 |
| Positive control | 自训（未到位）| 开源已知污染 |
| SPV-MIA reference | 同源 pre-SFT | 同架构开源 base |
| 差分归因层 | 核心创新 | 删除；contrastive 保留 |

**触发原因**：
- 三阶段框架 vs 公司实际训练形态（预训练+后训练）不匹配
- 自建 positive control 太贵，且本项目 GT 只能证明"对我们自己的 SFT 剂量 model 敏感"，无法证明"对已知污染开源模型敏感"
- 阶段归因需求可延后（终极目标是"检出污染"，不必知道哪阶段引入）

**产出文档**：
- `模型污染检测方案-v2.md`（supersedes `模型级污染检测：两阶段复用方案.md`）
- `实验计划-v2.md`（E0-E7 实验清单）

**代码目录不动**：`stage_base` / `stage_sft` / `stage_rlhf` 命名保留（改动成本高），语义解读为"方法本身的适用形态"，不再对应报告结构。

### 2. MemLens 方法论证伪 + 全删

**证伪证据**：8 卡 layer-wise smoke `outputs/2026-06-27_sft_contam_gt_memlens/20260701-151624/`（Qwen3-1.7B 28 层，4 ckpt × mmlu-pro × 10 题）。

| ckpt | L0 | L15 | L18 | L21 | L24 | L27 | **L28** |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 0 | 6.7e-4 | 4.9e-4 | 0.278 | 0.229 | 0.298 | 5.8e-5 |
| gsm8k_heavy | 0 | 4.6e-5 | 2.8e-4 | 0.218 | 0.258 | 0.268 | 2.4e-3 |
| **mmlu_heavy** | 0 | 7.3e-5 | 1.1e-5 | 0.298 | 0.398 | 0.483 | **0.502** |
| humaneval_heavy | 0 | 1.3e-3 | 4.7e-5 | 0.287 | 0.274 | 0.327 | 2.4e-3 |

**三条命中**：
1. 前 20 层 target prob 全 <1e-3 —— 论文"早期层 shortcut"假设不成立
2. L21-L27 clean 也有 0.22-0.30 concentration —— 通用答案表征位置，非记忆
3. **唯一区分层是 L28**（末层 lm_head）—— 改末层 signal 与 SPV-MIA / `_score_multiple_choice` mean-logprob 高度重叠，不是 MemLens 而是冗余信号

**处置**：全删（源码 + 单测 + configs + capability API + 引用清理）。
- `reports/trustworthiness.py` 留一段"MemLens 已删 + 理由"注释，防未来再入坑
- 历史 experiments/*/notes.md 未动（当时的结论属历史记录）

## Session 12 附带修复

Paraphrase smoke v2 gsm8k acc_orig=0 → session 12 修 `_score_math_cot` chat template。v3 smoke（`20260701-160629`）：`gsm8k_heavy × gsm8k acc_orig` 从 0 抬到 0.6，DIRTY；但 MMLU 侧 `_score_multiple_choice` 未同步修 → 后续 E0（见下一份摘要）。

## 数据 pointer

- MemLens layer-wise smoke（决策证据）：`outputs/2026-06-27_sft_contam_gt_memlens/20260701-151624/`
- Paraphrase v2（gsm8k acc_orig=0，未修版）：`outputs/2026-06-27_sft_contam_gt_paraphrase/20260701-143447/`
- Paraphrase v3（gsm8k 修复版）：`outputs/2026-06-27_sft_contam_gt_paraphrase/20260701-160629/`
- v2 计划：根目录 `实验计划-v2.md`

## 单测状态（截 2026-07-01 晚）

148 单测 pass（169 - memlens 17 - hf_local 4 网络）。

## 溯源

- `session12-handoff-2026-07-01-evening.md`（handoff 入口）
- `memlens-hypothesis-check-2026-07-01.md`（layer-wise 硬数据备份）
- `pivot-v2-scope-simplification-2026-07-01.md`（pivot 决策记录）
