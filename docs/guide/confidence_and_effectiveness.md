# 置信度、效力与发布边界

## 当前结论

本仓库目前能证明“某方法在某个受控实验中有区分信号”，还不能证明一个跨模型、跨 benchmark 的统一污染概率。原因是 benchmark 难度、模型能力、格式模板和数据来源都会改变 raw signal。

## 必须区分的指标

| 指标 | 适用条件 | 可回答的问题 | 不能回答的问题 |
|---|---|---|---|
| AUROC | target/member 与 control/non-member 标签可靠 | 两类样本的排序区分度 | 绝对污染率；小样本下稳定性 |
| AUPRC/F1 | 类别比例和阈值已声明 | 给定阈值的决策效力 | 类别改变后的可比性 |
| TPR@FPR | 有足够 clean control | 固定误报预算下的召回 | 没有 clean control 的真实 FPR |
| bootstrap CI | 有 per-sample scores，样本可重采样 | 指标不确定性 | 训练/数据分布外泛化 |
| dose-response ρ | 有多个已知注入剂量 | 信号是否随污染强度单调 | 真实世界的训练来源 |

## 建议的自证协议

1. **Null baseline**：未注入模型 + 随机 member/non-member split，AUC 应接近 0.5，不能有系统假阳。
2. **Positive control**：可控注入模型，至少 3 个剂量和 2 个 seed，报告 AUROC、TPR@FPR、bootstrap CI。
3. **Specificity**：跨模型家族、不同 benchmark、不同题型复测；注明模型能力是否变化。
4. **Ablation**：移除 context/reference/paraphrase 等核心组件，证明信号不是模板或长度伪影。
5. **Regression**：固定少量 clean/positive anchor，代码或依赖升级后自动重跑。

只有在 positive/negative control 都完成并冻结阈值后，才能把 `verdict_hint` 升级为发布门禁标签。否则报告应使用 `relative ranking`/`inconclusive` 语义。

## 置信区间和失败项

报告必须同时给：样本数、有效数、跳过数、NaN/异常数、随机种子、control/reference 身份。任一方法的前置条件不满足时，失败项必须显示在报告中；“没有结果”不是“clean”。

## 公开模型基线的正确用途

公开模型 sweep 可用于发现量纲、模型家族差异和候选回归 anchor；不能直接作为内部 checkpoint 的绝对阈值。见 [公开模型报告](../reports/public_model_baseline.md)。
