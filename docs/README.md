# 文档索引

文档按“使用契约 → 验证证据 → 报告 → 历史研究”分层。根目录 `README.md` 只描述当前可运行能力；旧方案和 session 记录不再作为用户契约。

## 给使用者

- [快速开始](guide/getting_started.md)：安装、列出 benchmark、运行测试和调用方法。
- [输入/输出契约](guide/input_output_contract.md)：题目归一化字段、模型能力、结果与 manifest。
- [方法说明](guide/methods.md)：每个方法的适用格式、指标方向、参数和失效条件。
- [置信度与效力](guide/confidence_and_effectiveness.md)：positive control、AUC、置信区间和发布门禁边界。
- [资源与规模](guide/performance.md)：前向次数、显存/时间估算、数据规模和 API 成本记录。

## 给开发者

- [配置说明](../configs/README.md)
- [Schema 说明](../schemas/README.md)
- [实验目录约定](../experiments/README.md)
- [验证证据总表](validation/method_validation.md)
- [排除的方法](validation/excluded_methods.md)

## 报告和历史

- [公开模型 baseline](reports/public_model_baseline.md)：2026-07-08 sweep，保留原始信号与限制。
- [Positive control 池](validation/positive_controls.md)：已知污染/干净锚点候选，不等于自动标签。
- [历史里程碑](experiments/INDEX.md)：只收录结论摘要，原始脚本仍在 `experiments/`。
- [调研归档](archive/)：两份早期调研，仅作为背景，不代表当前实现。

## 文档维护规则

1. 新增方法先写代码接口、单测和验证 notes，再加入 `method_validation.md`。
2. 没有 control 的数值必须标为 mean-only/relative，不得改写成 AUC 或污染概率。
3. 运行结果必须带 checkpoint、commit、数据版本、样本数、seed 和失败信息。
4. 已删除方法只在 `excluded_methods.md` 保留结论，不在默认配置、方法标签或报告分档中出现。
