# experiments/

每次实验单独一个目录，命名 `{YYYY-MM-DD}_{benchmark}_{stage}/`，例如：
- `2026-06-22_gsm8k_sft/`
- `2026-06-23_mmlu_base-vs-sft/`

每目录至少包含两个文件，git 跟踪：

- `run.yaml` — 输入配置：模型 checkpoint 路径、benchmark 列表、方法参数、cache 设置
- `notes.md` — 实验结论、看到的 surprise、下一步动作

不进 git 的：
- `outputs/` — 大文件、logprobs 缓存、原始结果数据
- `logs/` — 运行日志
- `*.parquet` / `*.jsonl` — 中间数据

## 模板

新实验前先 cp 一份：

```bash
cp -r experiments/_template experiments/2026-06-22_gsm8k_sft
```

`_template/run.yaml` 示例：

```yaml
model:
  checkpoint: /data/llama3-8b-sft
  stage: sft
  parent_checkpoint: /data/llama3-8b-base
benchmarks:
  - gsm8k
  - gsm1k
  - livebench
methods: [oren, family_diff, paraphrase]
oren:
  n_shards: 50
  alpha: 0.01
cache_dir: ./outputs/logprobs_cache
```

## 与 docs/ 仓库同步

实验结论摘要回写到 `../docs/experiments/`，附本仓库 commit hash + experiments/ 目录名。
代码细节不重复写到 docs/。
