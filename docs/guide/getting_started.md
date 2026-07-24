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

## 直接调用一个方法

当前没有完整的 pipeline runner，实验脚本应显式完成“加载 spec → 加载 questions → 初始化 model → 调用方法 → 序列化结果”。示例：

```python
from model_contamination.benchmarks import load_questions, load_registry
from model_contamination.models.hf_local import HFLocalModel
from model_contamination.shared.codec import codec_detect

registry = load_registry("configs/benchmarks.yaml")
spec = registry.get("gsm8k")
questions = load_questions(spec, limit=200)
model = HFLocalModel(
    "/models/Qwen3-1.7B-Base",
    stage_tag="base",
    device_map="auto",
)
result = codec_detect(model, spec, questions, keep_deltas=True)
print(result.signal, result.prerequisites_met, result.evidence)
```

`load_questions` 可能因数据集 gated、下架或尚未注册 normalizer 而失败；失败应记录为 run error，不要用空列表代替。

## 测试和静态检查

```bash
PYTHONPATH=src python -m pytest -q
ruff check src tests scripts
```

默认测试不联网、不下载模型。需要显式运行 Hugging Face tiny-model smoke 时使用：

```bash
RUN_HF_SMOKE=1 PYTHONPATH=src python -m pytest tests/test_hf_local.py -q
```

需要 GPU 的实验单独放在 `experiments/<date>_<name>/`，并把运行命令、模型路径、硬件和结论写入 `notes.md`。
