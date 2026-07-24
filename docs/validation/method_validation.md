# 方法验证登记

> 状态：截至 2026-07-24 的仓库整理快照。数值来自 `experiments/` 中的 notes 和已有公开模型报告；未重新运行大模型。

## 证据等级

- **A：受控闭环**：代码、单测、负对照、正注入和剂量/特异性证据齐全；
- **B：部分闭环**：有受控信号或论文复现，但存在结构性失效面；
- **C：研究/未接通**：只有原型、单点结果或缺少必要 control。

| 方法 | 级别 | 主要证据 | 当前生产语义 |
|---|---|---|---|
| CoDeC | A-/B+ | Pythia/OPT 跨架构、剂量、特异性和公开模型 sweep | 强信号候选；阈值仍需本仓库 positive control 冻结 |
| Min-K%++ | B+ | WikiMIA 论文复现、GSM8K 剂量/特异性 | 弱信号；无 control 只报 mean/summary stats |
| Option Permutation | B+ | Qwen 多规模复现、epoch ρ≈0.97、FP 分解 | MC 专用弱/相对信号；MMLU-Pro 易饱和 |
| SPV-MIA | B | Qwen3-1.7B SFT 数学/代码 dose 和 per-sample ROC | 有同源 base 时的强候选；必须标明 control 与短 completion 失效 |
| Paraphrase Stress | B- | 数学 CoT 剂量响应；MC 侧结构性失效 | 仅数学 CoT 的辅助强信号，不支持任意黑盒输入 |

## 关键可参考指标

- `codec`：公开 sweep 中 OLMo-2-7B-Instruct 的平均 signal 约 0.850，DeepSeek-7B 约 0.110；这只是跨模型相对量，不是污染概率。
- `perm_option`：MMLU-Pro 多数模型约 0.935–0.985，说明该题型接近统计饱和，不能精细排序。
- `mink_plus_plus`：有 control 的 MMLU/MMLU-Pro 才报告 AUC；math、math-500、evalplus 的 mean-only 数值不能横向当 AUC。
- `spv_mia`：公开 sweep 中多数结果是无 control 的 mean-only，统一标 `inconclusive`。

## 仍需补齐

1. 固定正/负 anchor，按方法和 benchmark 冻结阈值；
2. 每个关键结论至少 2 个 seed，并给 bootstrap CI；
3. 接通统一 runner 和 `run_manifest.json`；
4. 对 loader 注册表逐项标记“可加载/需 gated/待实现”，避免 metadata 看起来比代码更完整。
