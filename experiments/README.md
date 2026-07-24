# 实验目录约定

实验目录是研究过程的可复现入口，不是生产输出仓库。命名建议为 `{YYYY-MM-DD}_{主题}/`，例如 `2026-07-08_market_model_sweep/`。

## 必备文件

- `run.yaml` 或等价配置：模型 checkpoint、stage、benchmark、方法参数、seed、control/reference；
- `notes.md`：目的、实际命令、硬件/时间、结果、失败项、结论和下一步。

## 可选文件

运行脚本、数据准备脚本、`RUNBOOK.md` 和小型日志可跟踪；原始结果、checkpoint、parquet、jsonl、logprobs cache、TensorBoard 和日志目录不进 git。`.gitignore` 已覆盖常见大文件模式。

## 新实验模板

```bash
mkdir -p experiments/$(date +%F)_my_test
touch experiments/$(date +%F)_my_test/run.yaml
touch experiments/$(date +%F)_my_test/notes.md
```

notes 至少回答：

1. 这次实验验证哪个假设；
2. target/member 与 clean/control 如何构造；
3. 运行了多少样本、多少 seed、用了什么硬件和 wall time；
4. 指标是什么，是否有 CI/基线；
5. 结论是支持、否证还是 `inconclusive`。

## 归档规则

历史实验目录保留脚本和结论即可，不再把每个 session 复制成 docs。跨实验、可引用的结论回写到 `docs/validation/` 或 `docs/reports/`，并链接回本目录。

Self-Critique 两个实验目录已按项目决策删除；结论见 [excluded_methods.md](../docs/validation/excluded_methods.md)。
