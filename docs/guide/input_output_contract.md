# 输入与输出契约

## 输入层

### Benchmark 注册表

`configs/benchmarks.yaml` 是静态元数据，不是数据集本身。每个 entry 至少包含：

| 字段 | 含义 |
|---|---|
| `name` | 唯一 kebab-case 名称 |
| `format` | `multiple_choice`、`math_cot`、`code_completion`、`open_generation` 或 `cloze` |
| `data_source`/`data_id` | 数据来源和可复现标识；`TBD` 明确表示不可运行 |
| `variants` | 同族 control/对照 benchmark |
| `applicable_methods` | 方法候选，不代表 loader 和方法在当前版本一定可运行 |
| `stage_targets` | `base`/`sft`/`rlhf` 研究目标 |
| `tiers` | `p1-*`、`control`、`experimental`，决定是否进入正式报告 |

loader 将每条题归一化为 `BenchmarkQuestion`：

```json
{
  "id": "gsm8k-17",
  "benchmark": "gsm8k",
  "format": "math_cot",
  "prompt": "题面，不含答案",
  "answer": "标准化答案",
  "choices": null,
  "answer_index": null,
  "full_answer": "可选的完整 CoT",
  "raw": {}
}
```

多选题必须填 `choices` 和 `answer_index`；数学 CoT 若要运行 SPV-MIA/Paraphrase，应填 `full_answer`。原始字段只放在 `raw`，方法层不应猜数据集列名。

### 模型后端

方法只依赖 `ModelInterface`，并在调用前检查 `Capability`：

- `LOGPROBS`：给定 prompt/completion 的逐 token log-prob；
- `TOKEN_DIST_STATS`：逐位置 chosen log-prob、μ、σ，Min-K%++ 需要；
- `GENERATE`：自由生成，当前 evaluator 可用；
- `BATCH`：批量执行的性能能力。

当前仓库只提供并验证 `HFLocalModel`。不存在可用于生产结论的 vLLM 或远程 API adapter。

## 结果层

每个方法必须返回 `DetectionResult`，字段语义如下：

```json
{
  "method": "codec",
  "stage": "base",
  "benchmark": "gsm8k",
  "signal": 0.73,
  "verdict_hint": "suspect",
  "prerequisites_met": true,
  "evidence": {"n_samples": 200, "n_skipped": 0, "seed": 42},
  "error": null
}
```

约束：

- 前置条件缺失时 `signal` 必须为 `null`，`prerequisites_met=false`，并填写可诊断的 `error`；
- `verdict_hint` 是方法内部阈值提示，不等于最终发布裁决；
- `evidence` 至少包含样本数、有效/跳过数、关键参数、seed 和阈值来源；
- `signal` 的量纲必须在方法文档中定义，跨方法比较只能通过明确的 contrastive/校准层完成。

## 运行产物契约

正式 runner 接通后，每个 run 应落在 `outputs/runs/<run-id>/`：

```text
input_config.yaml    # 实际生效的配置快照
results.jsonl        # 每个 method × benchmark 一行 DetectionResult
report.md             # 人读报告，包含限制和失败项
run_manifest.json     # 模型/数据/代码/资源/时间元数据
```

对应 JSON Schema 在 `schemas/`。在 runner 尚未接通前，实验脚本也应尽量遵循相同字段，不要只保存一张手工汇总表。
