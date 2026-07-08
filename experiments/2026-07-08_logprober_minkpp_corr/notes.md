# LogProber vs Min-K%++ 信号相关性

**日期**：2026-07-08
**状态**：脚本就位，待 8 卡机跑

## 问题

LogProber（arXiv:2408.14352）和 Min-K%++（arXiv:2404.02936）都从「单次前向的 per-token
log-prob」派生。`log_prober.py` 目前是 `NotImplementedError`。实装前先量化它相对 Min-K%++
的边际价值：**冗余还是互补**。

## 方法

同一次前向里算三个信号，拆开两个正交维度（归一化 vs 聚合方式）：

| 信号 | 用的量 | 聚合 |
| --- | --- | --- |
| `minkpp` | 归一化 `(chosen-μ)/σ` | bottom-K% 均值 |
| `mink_loss` | 原始 `chosen` | bottom-K% 均值 |
| `logprober_b` | 原始 `chosen` | 累积 surprisal 曲线 `A(1-e^{-Bx})` 拟合的 B |

- `minkpp` vs `mink_loss` → 隔离 **z-normalization** 的贡献
- `mink_loss` vs `logprober_b` → 隔离 **bottom-k 聚合 → 曲线水平度** 的贡献

数据 WikiMIA + Pythia-6.9B（member/non-member 有 ground truth，Min-K%++ 已复现 AUROC≈0.70）。
LogProber 论文聚焦 question，WikiMIA 无 Q/A 划分 → 退化为聚焦整段文本，与 Min-K%++
在 repro 里用法一致，保证两信号跑在相同输入上。

**不含 paraphrase 对照**：那是论文区分 confidence vs contamination 的应用层手段，会让前向×2；
相关性研究只需 per-item 原文水平度。

## 判据

- ρ(logprober_b, minkpp) ≥ ~0.85 且 logprober AUROC ≤ minkpp → **冗余**，保留 NotImplementedError，docstring 标注
- ρ 中等 (0.4~0.7) 且 AUROC 相当 → **可能互补**，值得真实装 + paraphrase 对照
- logprober AUROC ≈ 0.5 → 此场景不 work，冗余

member/non-member 拆开看相关性：防止 pooled ρ 只是 member-nonmember 整体分层带来的假相关。

## 跑法

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HOME=/mnt/public/code/chennuoxi/hf_cache \
  PYTHONPATH=src python3 experiments/2026-07-08_logprober_minkpp_corr/run_corr.py
```

## 结论（2026-07-08 跑完，WikiMIA + Pythia-6.9B）

| split (n) | MinK++ AUROC | LogProber_B AUROC | ρ(B,minkpp) pooled | ρ(minkpp,mink_loss) |
| --- | --- | --- | --- | --- |
| len32 (776) | 0.699 | 0.574 | +0.427 | +0.711 |
| len64 (542) | 0.711 | 0.480 | +0.265 | +0.709 |
| len128 (250) | 0.693 | 0.445 | +0.178 | +0.731 |

1. MinK++ 复现有效（0.69~0.71，对齐 minkpp-validation）→ 接线正确，基线可信。
2. LogProber_B 在 WikiMIA **不判别**：AUROC 0.57→0.48→0.45，len32 勉强过 0.5，
   len64/128 掉到 0.5 以下，随长度单调退化。logprober_A 同样 ≈0.5。
   指数拟合的「早期触底」信号在长序列上被主体稀释。
3. **不冗余但也不是有用正交**：ρ(B,minkpp) 仅 0.18~0.43（随长度降），远低于
   mink_loss↔minkpp 的 0.6~0.75。低相关 + 无判别力 = 噪声，非互补信号。

**Caveat（重要）**：WikiMIA 是 MinK++ 主场（弥散预训练 membership），不是 LogProber
本命（短 benchmark 题目逐字记忆）。本实验证明 LogProber 替代/增强不了 MinK++ 做 MIA，
但**未测其本命场景**。决定性结论需在注入型逐字过拟合 ckpt（minkpp_dose_response 那批）
上重跑同一对比。

**当前裁决**：保留 `log_prober.py` 的 NotImplementedError，低优先级。理由叠加——
(a) 在可测的 MIA 场景输给 MinK++；(b) 本命场景需另做专门验证才能主张价值；
(c) MinK++ 自身已降级为弱信号，LogProber 边际价值更低（见 project-scope-internal-model）。

## 备注

- LogProber 分数是论文原版（指数拟合 B），不是排序累积面积。绝对值不必对齐论文常数，
  相关性/AUROC 对单调变换不变。
- `logprober_a`（asymptote）也一并输出：member 应更可预测 → A 更小 → 其 AUROC 预期 <0.5，
  作方向 sanity check。
- `log_prober.py` 现有 docstring 写的是「P(Q)/P(A|Q)/P(A) 三分类」，与论文实际方法不符；
  待本实验定论后统一修正（实装 or 标注冗余）。
