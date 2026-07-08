# Min-K%++ 论文复现 — WikiMIA × Pythia-6.9B

论文：Zhang et al. 2024, *Min-K%++: Improved Baseline for Detecting Pre-Training Data*（arXiv:2404.02936）
方法实现：`src/model_contamination/stage_base/min_k_plus_plus.py`（`_mink_pp_sample_score`）
运行：`run_repro.py`（配置 `run.yaml`），结果 `outputs/wikimia_repro.json`（不进 git）

## 目的

验证 Min-K%++ 核心打分实现无误。WikiMIA 自带 ground-truth 标签（label=1 = 训练 cutoff 前
Wikipedia，模型见过 = member；label=0 = cutoff 后新增 = non-member），直接算 member vs
non-member 的 AUROC，与论文数值对齐 → 若量级吻合即证明实现正确。

## 配置

- 模型：Pythia-6.9B（`hf_cache/models/pythia-6.9b`，论文 WikiMIA 主表对照模型之一）
- 数据：`swj0419/WikiMIA`，splits length32 / 64 / 128（非 gated）
- 每样本对**原始文本**打分（绕开 `_split_prompt_completion` 的 QA 封装——WikiMIA 是自由文本 MIA）
- prefix=`\n\n`：gpt_neox 无 BOS，空 prompt 会让 logprobs 崩；用分隔符做 conditioning boundary
- k 扫描 {0.1..1.0}，每样本前向只跑一次、复用缓存 per-token 数组
- 同时算 **Min-K%（loss，无 z-normalize）** 基线：论文核心主张是 z-normalization 带来增益，
  Min-K%++ 应显著 > Min-K% loss

## 论文对照基准

论文 Table 里 Pythia-6.9B 在 WikiMIA 上 Min-K%++ AUROC ≈ 0.70（Min-K% loss 基线更低）。
复现判定：**Min-K%++ 落在 0.68–0.72 量级、且高于 Min-K% loss** → 实现正确。

| split | n | Min-K%++ @k=0.2 | Min-K%(loss) @k=0.2 | best-k Min-K%++ | 判定 |
| --- | --- | --- | --- | --- | --- |
| length32 | 776 | **0.699** | 0.657 | 0.702 @k=0.3 | ✅ 量级吻合 + z-norm 增益 |
| length64 | 542 | **0.711** | 0.644 | 0.714 @k=0.1 | ✅ 同上 |
| length128 | 250 | **0.695** | 0.685 | 0.709 @k=0.5 | ✅ 同上 |

结果文件：`outputs/wikimia_repro.json`（含全 k 扫描曲线）。

## 结论

**复现成功，实现无误。**
1. **数值对齐**：三个 split 的 Min-K%++ AUROC 全部落在 0.695–0.711，正中论文 Pythia-6.9B ≈ 0.70。
2. **方法排序正确**：每个 split 上 Min-K%++ 均 > Min-K%(loss)（length64 最明显 0.711 vs 0.644，
   +6.7pt）——论文核心主张「z-normalization 带来 AUROC 增益」在本仓库实现上复现。
3. **k 稳健**：best-k 与 k=0.2 差距 ≤ 1.5pt，说明 statistic 对 k 不敏感，k=0.2 默认合理。

→ `_mink_pp_sample_score`（含 `token_logprob_stats` 的 μ/σ 计算）可信，后续剂量-响应 /
特异性实验建立在正确实现之上。
