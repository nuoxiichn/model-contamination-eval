# model-contamination-eval

模型级 benchmark 污染检测实验仓库。它把同一组 benchmark、模型后端和检测方法组织成可复现的实验，并输出样本级信号、方法级结果和带限制说明的报告。

## 当前定位

本仓库目前是“可复现研究工具 + 受限生产原型”，不是已经闭环的发布门禁服务。当前可运行链路是：

- benchmark 注册表和部分 Hugging Face 数据集 loader；
- Hugging Face 本地 checkpoint 的生成、log-prob 和 token 分布统计；
- CoDeC、Min-K%++、Option Permutation、SPV-MIA、数学 CoT Paraphrase Stress 五类方法；
- 统一 `DetectionResult`、可信度聚合和跨 checkpoint contrastive/diff 工具；
- 实验配置、结论记录、公开模型 sweep 报告和方法验证登记。

以下能力目前明确不在生产方法集：Self-Critique（已删除，保留排除结论）、vLLM/远程 API 后端、自动阶段归因、数据配方反馈、绝对红黄绿阈值校准。

## 仓库结构

```text
configs/       benchmark 注册表、pipeline 参数、可复制的运行示例
docs/          使用契约、方法验证、公开模型报告和少量历史归档
experiments/   可复现实验脚本与 notes.md；大输出不入 git
schemas/       运行配置、结果和 manifest 的机器可读契约
src/           Python 包 model_contamination
scripts/       数据下载和实验辅助脚本
tests/         单元测试（不需要下载模型）
outputs/       本地运行产物（git ignored，仅保留目录占位）
```

## 快速开始

```bash
# 推荐 Python 3.10+；本地 HF 模型和开发依赖
python -m pip install -e '.[hf,dev]'

# 查看注册表与方法过滤
PYTHONPATH=src python -m model_contamination.cli list-benchmarks
PYTHONPATH=src python -m model_contamination.cli list-benchmarks --method codec

# 校验 benchmark 配置
PYTHONPATH=src python -m model_contamination.cli validate-config

# 校验任务矩阵，不加载模型和数据
MODEL_CHECKPOINT=/absolute/path/to/checkpoint \
  mcd run --config configs/examples/local_hf.yaml --dry-run

# 执行并生成 input_config.yaml、results.jsonl、report.md、run_manifest.json
MODEL_CHECKPOINT=/absolute/path/to/checkpoint \
  mcd run --config configs/examples/local_hf.yaml

# 运行测试
PYTHONPATH=src python -m pytest -q
```

`model.path` 可以是本地 Transformers/Hugging Face 格式的 checkpoint 目录，也可以是 `from_pretrained` 能解析的 Hub model ID。本地训练模型只需导出为该目录格式，不需要上传到 Hugging Face Hub。完整配置和失败语义见 [快速开始](docs/guide/getting_started.md) 与 [输入输出契约](docs/guide/input_output_contract.md)。旧的 `mcd detect` 只保留迁移提示。

## 方法状态摘要

| 方法 | 需要的模型能力 | 当前输入 | 当前证据 | 主要限制 |
|---|---|---|---|---|
| CoDeC | log-probs | 至少 2 条同一 benchmark 样本 | 剂量、特异性、跨模型实验较完整 | 同质 context/多主题 benchmark 会改变解释；阈值尚未本仓库校准 |
| Min-K%++ | token-level 全词表统计 | 任意可 token 化文本；有 control 才能算 AUC | 论文复现、剂量和特异性 | 无 control 只能给位置统计，不能叫 AUC |
| Option Permutation | log-probs | ≥3 选项的多选题 | 剂量响应和 FP 分解 | MMLU-Pro 等题目可能信号饱和；绝对 FPR 高 |
| SPV-MIA | target/reference log-probs | 有同源 base 的 SFT、足够长的 completion | 数学 CoT/per-sample 侧有证据 | MC 单字母和短答案结构性失效；无 control 仅 mean-only |
| Paraphrase Stress | log-probs | 当前只支持 `math_cot` | 数学 CoT 有限剂量响应 | 规则改写，不支持当前配置里的所有格式；不是黑盒 API 方法 |

详细的适用格式、方向、证据等级和失败条件见 [方法说明](docs/guide/methods.md) 与 [验证登记](docs/validation/method_validation.md)。

## 结果如何解读

每个方法返回统一的 `DetectionResult`：`signal`、`verdict_hint`、`prerequisites_met`、`evidence` 和 `error`。`signal` 不是跨方法通用的污染概率：

- 有 positive control、同源 reference 或同分布 control 时，才可使用 AUC、对比差值或校准阈值；
- 没有这些前置条件时，结果只能作为相对排序或 mean-only 诊断；
- 单个 benchmark 的单个方法不能证明“模型训练过该数据集”；
- 报告必须同时展示样本数、失败数、方法参数、模型和 checkpoint 身份。

当前可信度聚合默认是 ranked-list 语义，不能把 `dirty/suspect` 当成发布阻断结论。见 [置信度与效力](docs/guide/confidence_and_effectiveness.md)。

## 外部 API 模型

GPT、Claude、Kimi、DeepSeek 等 API 不是当前基线的必需项：本仓库的主方法依赖 log-prob 或全词表统计，远程 API 通常无法提供这些能力，且当前没有稳定的 API backend。只有在需要验证“黑盒迁移性”时，才建议新增单独的生成式 adapter 和预注册样本预算；不得把 API 结果与本地白盒结果直接混排。

API 实验应记录真实的 input/output token、模型版本、请求失败和价格快照。成本按 `input_tokens × input_price + output_tokens × output_price` 计算，不能只记录调用次数。具体决策建议见 [性能与规模](docs/guide/performance.md)。

## 参考报告

- [公开模型 baseline sweep](docs/reports/public_model_baseline.md)
- [方法验证登记](docs/validation/method_validation.md)
- [排除的方法：Self-Critique](docs/validation/excluded_methods.md)
- [文档总索引](docs/README.md)
