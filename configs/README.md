# 配置说明

`benchmarks.yaml` 是 benchmark 元数据注册表，`pipeline.yaml` 是默认方法参数，`examples/` 放可复制的运行输入。

注意：当前没有统一 runner 解析所有 pipeline 字段；配置首先作为实验契约和未来 runner 的输入。实验脚本可以覆盖配置，但必须在 `notes.md` 中记录实际生效值。

```bash
PYTHONPATH=src python -m model_contamination.cli validate-config --config configs/benchmarks.yaml
```

模型路径、token、API key 和大文件只通过环境变量或本地未跟踪配置提供，不要提交到仓库。
