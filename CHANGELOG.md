# Changelog

## 2026-07-24 — repository cleanup

- 统一 README、docs、configs、schemas 和 experiments 的职责与入口。
- 删除 Self-Critique 实现、测试、训练脚本和约 1.05 TB 本地 checkpoint；保留失败复现结论。
- 从默认方法/报告信号中移除 Self-Critique；将 CoDeC 纳入当前强信号候选。
- 删除未实现的 vLLM、阶段归因渲染和配方反馈占位模块。
- 将公开模型 sweep 移入 `docs/reports/`，两份调研报告移入少量历史归档。
- 将 HF tiny-model smoke 改为 `RUN_HF_SMOKE=1` 显式运行，默认测试保持离线。

## Earlier research

早期实验和实现沿用 git 历史与 `experiments/` 目录中的 notes；不要把历史方案文档当作当前接口契约。
