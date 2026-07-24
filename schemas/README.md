# Schema

Schema 是 `mcd run`、报告脚本和 CI 之间的稳定边界。统一 runner 会写出与这些契约对应的配置快照、逐任务结果和 manifest；独立研究脚本如需进入正式报告，也应转换为相同字段。

- `run_config.schema.json`：模型、benchmark、方法和资源输入；
- `detection_result.schema.json`：与 `DetectionResult` 对齐的单条结果；
- `run_manifest.schema.json`：代码、环境、数据、时间、资源和任务状态。

不把模型权重、API key、完整 token 序列或大数组提交到 git；结果文件可用外部 URI 和 sha256 指针引用。
