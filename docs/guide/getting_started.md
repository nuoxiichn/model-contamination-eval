# 快速开始

## 环境

项目代码兼容 Python 3.10+；`.[hf]` 额外依赖用于加载 Hugging Face 本地模型，`.[dev]` 用于测试和 lint。

```bash
python -m pip install -e '.[dev]'
python -m pip install -e '.[hf]'
```

权重、数据集、log-prob 缓存和实验输出都不进 git。建议把它们放在共享存储，并在运行记录中保存绝对路径或可解析的 URI。

## 检查注册表

```bash
PYTHONPATH=src python -m model_contamination.cli validate-config
PYTHONPATH=src python -m model_contamination.cli list-benchmarks --method codec
```

`validate-config` 只校验 YAML 和变体引用，不会下载数据，也不会验证每个 `data_id` 当前可访问。真正的 loader 支持以 [输入/输出契约](input_output_contract.md) 和 loader 单测为准。

## 运行一个本地模型

`hf_local` 表示模型目录遵循 Transformers/Hugging Face 的 `from_pretrained` 格式，不表示模型必须上传到 Hub。本地训练产出的完整 checkpoint、merge 后的 LoRA checkpoint 都可以直接使用；目录通常应包含模型 config、权重和 tokenizer 文件。

先校验配置、benchmark 与方法兼容性。`--dry-run` 不加载模型、不下载数据，也不创建输出目录：

```bash
export MODEL_CHECKPOINT=/models/Qwen3-1.7B-Base
mcd run --config configs/examples/local_hf.yaml --dry-run
mcd run --config configs/examples/local_hf.yaml
```

默认输出目录由 `run.output_dir` 指定，也可以临时覆盖：

```bash
mcd run --config configs/examples/local_hf.yaml \
  --output-dir outputs/runs/qwen3-base-smoke
```

已有标准产物时命令默认拒绝覆盖；确认要替换同一个 run 时显式使用 `--overwrite`。每个配置中的 `benchmark × method` 都会在 `results.jsonl` 和 manifest 中留下记录，模型、数据或方法失败不会被静默写成零分。

SFT 模型运行 SPV-MIA 时还需要同源 base/reference checkpoint，参见 `configs/examples/sft_with_reference.yaml`。如果没有 reference，应从配置中移除 `spv_mia`；无 reference 不能把该任务解释为“检测为干净”。

`load_questions` 仍可能因数据集 gated、下架、本地文件缺失或尚未注册 normalizer 而失败。runner 会保留其他任务的结果，并把具体错误写入 `results.jsonl`、`report.md` 和 `run_manifest.json`。

## 配置要点

- `run.seed` 同时控制随机抽样和未单独覆盖 seed 的方法；
- `selection: random` 在 `max_samples` 下做可复现抽样，`head` 取前 N 条；
- `control` 仅由 Min-K%++ 和 SPV-MIA 使用，用于 AUC；未配置时这两种方法只输出 mean-only/inconclusive；
- `model.reference` 只在 SPV-MIA 任务首次执行时加载；
- 相对路径以命令执行时的工作目录为基准，团队运行建议从仓库根目录执行。

## 测试和静态检查

```bash
PYTHONPATH=src python -m pytest -q
ruff check src tests scripts
```

默认测试不联网、不下载模型。需要显式运行 Hugging Face tiny-model smoke 时使用：

```bash
RUN_HF_SMOKE=1 PYTHONPATH=src python -m pytest tests/test_hf_local.py -q
```

需要 GPU 的研究实验仍可单独放在 `experiments/<date>_<name>/`；正式 runner 的配置快照、环境、运行时间和任务错误以输出目录中的四个标准产物为准。
