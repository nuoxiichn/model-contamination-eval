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

`mcd run` 的 `model.path` 接受本地 checkpoint 目录或 Hugging Face Hub model ID。这里的 HF 是 `AutoModelForCausalLM.from_pretrained` / `AutoTokenizer.from_pretrained` 可读取的格式约定；本地模型不需要上传。SFT 的 SPV-MIA 配置还必须提供 `model.reference`，通常指向同源 base checkpoint。

## 运行配置

runner 输入是严格校验的 YAML。完整示例见 `configs/examples/`，核心结构为：

```yaml
schema_version: "1.0"
run:
  id: local-smoke
  seed: 42
  output_dir: outputs/runs/local-smoke
  fail_fast: false
model:
  backend: hf_local
  path: ${MODEL_CHECKPOINT}
  stage: base
  dtype: bfloat16
  device_map: auto
benchmarks:
  - name: gsm8k
    split: test
    max_samples: 50
    selection: random
methods:
  - name: codec
    params: {n_seeds: 2}
```

字符串中的 `${VAR}` 和 `${ENV:VAR}` 都从环境变量展开；缺失变量会在加载配置时显式报错。benchmark 和 method 名称在单次 run 中必须唯一，未知字段、未知方法参数和非法参数范围均会拒绝执行。

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

每个 `mcd run` 落在 `run.output_dir`（通常为 `outputs/runs/<run-id>/`）：

```text
input_config.yaml    # 实际生效的配置快照
results.jsonl        # 每个 method × benchmark 一行 DetectionResult
report.md             # 人读报告，包含限制和失败项
run_manifest.json     # 模型/数据/代码/资源/时间元数据
```

对应 JSON Schema 在 `schemas/`。`results.jsonl` 按配置顺序对完整的 `benchmark × method` 矩阵逐行写入，所以中途方法失败后，已完成结果仍然可读；未执行、前置条件不足和异常分别在 manifest 中标为 `skipped`、`inconclusive` 或 `error`，不会静默省略。

`run_manifest.json` 记录 Git commit/dirty 状态、模型与方法配置、Python 和关键包版本、开始/结束时间、任务级样本数/耗时/错误以及产物清单。它记录的是配置资源信息，不等价于 GPU 峰值显存监控；实测显存、功耗等仍需实验环境额外采集。

默认不覆盖已有标准产物。`--overwrite` 会从头重写同一输出目录中的四个标准文件；`--output-dir` 只覆盖本次运行的输出位置，并写入生效后的 `input_config.yaml`。
