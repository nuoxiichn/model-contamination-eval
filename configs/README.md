# 配置说明

`benchmarks.yaml` 是 benchmark 元数据注册表，`pipeline.yaml` 是可直接运行的默认模板，`examples/` 放按场景复制的运行输入。

运行配置由 `mcd run` 严格解析。推荐先做不加载模型和数据的任务矩阵检查：

```bash
PYTHONPATH=src python -m model_contamination.cli validate-config --config configs/benchmarks.yaml
MODEL_CHECKPOINT=/models/checkpoint mcd run --config configs/pipeline.yaml --dry-run
```

模型路径可写为 `${MODEL_CHECKPOINT}` 或 `${ENV:MODEL_CHECKPOINT}`。本地训练模型只需是 Transformers/Hugging Face 可读取的目录，无需上传到 Hub。token、API key 和大文件只通过环境变量或本地未跟踪配置提供，不要提交到仓库。

字段和失败语义见 [输入输出契约](../docs/guide/input_output_contract.md)，机器可读约束见 [run_config.schema.json](../schemas/run_config.schema.json)。
