# Schema

Schema 是运行产物的稳定边界，供未来统一 runner、报告脚本和 CI 校验。当前实验脚本尚未全部写出 `run_manifest.json`，但新增实验应尽量遵循这些字段。

- `run_config.schema.json`：模型、benchmark、方法和资源输入；
- `detection_result.schema.json`：与 `DetectionResult` 对齐的单条结果；
- `run_manifest.schema.json`：代码、环境、数据、时间、资源和任务状态。

不把模型权重、API key、完整 token 序列或大数组提交到 git；结果文件可用外部 URI 和 sha256 指针引用。
