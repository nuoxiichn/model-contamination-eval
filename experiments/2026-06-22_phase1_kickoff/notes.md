# Phase 1 Kickoff（2026-06-22）

> 本文档是 Phase 1 端到端 plumbing 完成节点的实况记录，下一个 agent 接手前先读完。

## 0. 结论

- Phase 1 端到端 sanity-check 跑通：`mcd sanity-check --model hf-internal-testing/tiny-random-gpt2 ...` 输出 `outputs/sanity/20260622-180304/signal_table.{json,md}`
- 51 个测试全部通过（43 单测 + 8 真模型 smoke）
- **Phase 0 硬前置 5 项尚未真正满足**（真 checkpoint 还没拿到），所以本次只用 tiny-random-gpt2 验证 plumbing，没有真实污染信号

## 1. 本 session 完成的代码改动

### 新增文件
- `src/model_contamination/benchmarks/loader.py` — `load_questions(spec, limit, split, subset)`，5 个 normalizer
- `src/model_contamination/shared/evaluator.py` — `evaluate_accuracy`（MC + math_cot dispatch）+ `extract_math_answer`
- `tests/test_loader.py`、`tests/test_hf_local.py`、`tests/test_evaluator.py`、`tests/test_oren.py`

### 实装从 stub → 完成
- `src/model_contamination/models/hf_local.py` — generate / batch_generate / logprobs / hidden_states 全部实装；含 context-length 守护
- `src/model_contamination/shared/oren_permutation.py` — 分片排列 + 单侧 t 检验
- `src/model_contamination/shared/family_diff.py` — 加 `run_family_diff(model, registry, name)` 端到端 wrapper
- `src/model_contamination/cli.py` — sanity-check 真跑

### 修改
- `src/model_contamination/types.py` — 加 `BenchmarkQuestion` 数据类；`BenchmarkSpec` 加 `data_subset` 字段
- `src/model_contamination/benchmarks/registry.py` — 识别 `data_subset`
- `scripts/download_benchmarks.py` — 用 `spec.data_subset` 传 HF config 名；跳过 `data_id=TBD`
- `configs/benchmarks.yaml` — gpqa-diamond / mgsm / gsm8k 加 `data_subset`；math 换源到 `EleutherAI/hendrycks_math`（algebra）；livecodebench 标 TBD

## 2. 已确认的设计决策

### 方案 A：spec 与 questions 分离
方法接口 `detect(model: ModelInterface, spec: BenchmarkSpec, questions: list[BenchmarkQuestion])`。
- spec 保持 `frozen=True` 只读元信息
- questions 单独由 loader 返回，便于切片 / 采样 / paraphrase 改写 / canary 注入
- **不要改回方案 B**（用户已在本 session 明确确认 A）

### 数据下载路径
`/mnt/public/code/chennuoxi/hf_cache`（跨项目复用、用户个人路径）。环境变量 `HF_DATASETS_CACHE` 指过去。镜像 `HF_ENDPOINT=https://hf-mirror.com`。

### CLI sanity-check 默认参数
`shard_size=3 / limit=40 / min_samples=shard_size*2`。这是 tiny-gpt2 max_position=512 下的安全值。**生产模型（≥7B，n_positions=4096+）应改用 `shard_size=25 / min_samples=200 / limit=200`**，对齐 Oren 论文协议。

### context-length 守护
`hf_local.logprobs` 超长截断保尾部、`batch_generate` 留 max_tokens 给生成、prompt 左截断。两层守护防 IndexError。**不要拆掉守护**（PR 化时容易被当作"无用代码"）。

### 红线遵守
sanity-check 输出**只有 ranked list + 原始 signal**，不出红/黄/绿绝对裁决。CLAUDE.md 明确"positive control 未到位前不输出绝对裁决"。

## 3. 已知坑 / 不要踩

1. **datasets 5.0 不再支持 `trust_remote_code=True`** → 任何带 loader script 的数据集（livecodebench 是例子）现在加载失败。修法是换 parquet 镜像。
2. **GSM8K 单题 token 数 ~115**，shard_size=5 拼起来 ≈ 580 > tiny-gpt2 的 512。**真模型 max_position=4096+ 时这不是问题**，但写测试用 tiny-gpt2 时要小心。
3. **gsm-plus normalizer 未注册** → family_diff variants 现在跑不出来；只是 follow-up，不阻塞主线。
4. **gsm1k 标的是 `data/gsm1k.jsonl`，但文件不在仓库** → 同上。
5. **uv 在开发机上没装**，CLAUDE.md 写的 `uv run` 命令实际上跑不了，目前用 conda `python3` 直接跑。要么装 uv，要么改文档。**没改 CLAUDE.md，留给用户决定**。
6. **Pylance 报 pytest 找不到** → IDE 配置问题（VSCode 用的 Python 解释器不是 conda env），不影响 pytest 真跑。

## 4. 测试命令

```bash
PYTHONPATH=src python3 -m pytest tests/ -q                      # 全量（首次会下 tiny-gpt2）
PYTHONPATH=src python3 -m pytest tests/ -q --ignore=tests/test_hf_local.py  # 不跑真模型 smoke
HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_CACHE=/mnt/public/code/chennuoxi/hf_cache PYTHONPATH=src python3 -m model_contamination.cli sanity-check --model hf-internal-testing/tiny-random-gpt2 --device cpu
```

## 5. follow-up 任务（TaskList 已记录）

| # | 任务 | 阻塞主线? |
|---|---|---|
| #7 | MATH 全 7 子集合并加载 | Phase 2 evaluator 前修 |
| #8 | LiveCodeBench: 补可用 data_id | 不阻塞 Phase 1，影响代码类对照 |
| #9 | gsm-plus / gsm1k / lbpp / mhpp / aime normalizer | 让 family_diff 真有 variants |

## 6. 下一步选项（agent 接手时和用户对齐）

**等真 checkpoint 路径**：用户已在催老师要 base + SFT 同源 checkpoint。一拿到就：
```bash
mcd sanity-check --model <base_path> --stage base --shard-size 25 --limit 200
mcd sanity-check --model <sft_path>  --stage sft  --shard-size 25 --limit 200
```
对比两个 stage 的 Oren p-value 与 family_diff，第一份**真实**信号。

**checkpoint 还没到位时可推**（并行做不浪费）：
- **#9 normalizer 补齐** → family_diff 在 sanity 中也能拉到信号
- **`shared/guided_instruction.py`** 实装：黑盒方法，只需要 generate，门槛低
- **`shared/paraphrase_stress.py`** 实装：同上，配一个简单改写器（规则 / LLM 都行）
- **`stage_base/min_k_plus_plus.py`**：白盒方法，**只需要单 checkpoint**（不需要 base+SFT 配对），可以直接用 tiny-gpt2 跑通 plumbing

**不该提前做**（缺前置）：
- SPV-MIA：必须 base + SFT 双 checkpoint
- MemLens / LogProber：需要 LOGITS_LENS 能力，hf_local 当前 supports 返回 False，要先实装该能力
- Self-Critique：RLHF 阶段方法，无 RLHF checkpoint

## 7. 给新 agent 的提示

- **先读 `CLAUDE.md`**（项目根 + 用户 `/mnt/public/code/chennuoxi/CLAUDE.md`），里头有红线和命名约定
- **再读 `TaskList`**（用 TaskList 工具），所有 follow-up 都在里头
- **测试命令在第 4 节**
- **不要假装实现没验证过的方法**：未跑通的留 `raise NotImplementedError("Phase X: 见 issue #N")` + docstring
- **不要悄悄修改红线**：尤其是"不出绝对裁决 / 不要把单 benchmark MIA AUC 当唯一证据"
- **真 checkpoint 没到位前，Phase 1 的 sanity-check 已经是上限**，再多做集成测试没意义。把时间投到第 6 节的"checkpoint 没到位时可推"项里

## 8. 当前 git 状态

main 分支上有未 commit 的改动（这次 session 全部）。新 agent 接手前先决定：
- 现在 commit 一个"Phase 1 plumbing 完成"的节点，还是
- 继续做 follow-up 攒到一起再 commit

CLAUDE.md "git push/rebase/reset" 在红线里，**`git commit` 不在红线**，但 push 一定要问用户。
