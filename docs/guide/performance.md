# 资源、时间和可扩展规模

下面是方法级估算，不是 SLA。实际耗时取决于模型大小、序列长度、GPU、batch、dtype、数据加载和缓存命中率；正式 run 必须记录 wall time 和硬件。

## 前向次数估算

设样本数为 `N`，平均 completion token 数为 `T`：

| 方法 | 近似工作量 | 主要资源瓶颈 |
|---|---:|---|
| CoDeC | `N × (1 + n_seeds)` 次 log-prob；默认约 `6N` | 长 context 带来的显存和序列长度 |
| Min-K%++ | `N` 次 full-vocab token stats；有 control 再加 `N_control` | 全词表 logits 显存/吞吐 |
| Option Permutation | `N × min(n!, max_permutations)` 个候选；实现会批量复用 | MC 选项数，6+ 选项会快速膨胀 |
| SPV-MIA | target/reference 对每题约 `1 + n_neighbors` 组 log-prob；默认 10 neighbors | 两个 checkpoint 的总显存和重复前向 |
| Paraphrase Stress | 原题 + `n_paraphrases` 改写的评估；默认 5 | 改写缓存和重复评分 |

## 数据集规模建议

- smoke：20–50 条，只验证 loader、模板和前置条件，不下结论；
- 方法实验：100–200 条，适合快速 dose/specificity；
- 稳定报告：约 500–1000 条，CoDeC 和 bootstrap CI 更可靠；
- MMLU-Pro/GPQA 等大 MC 任务应限制排列数并记录 `max_permutations`，不要盲目全枚举。

公开 sweep 的历史参考是单模型×benchmark 通常 164–200 条、8 卡运行；这只是本仓库已有实验的量级，不代表所有模型都能在同一时间完成。

## 时间记录

每个 run manifest 至少记录：开始/结束时间、wall time、GPU/CPU/内存、batch、dtype、样本数、缓存命中数和失败数。不要只在 notes 中写“跑了几小时”。

## API 模型的 token 与费用

GPT/Claude/Kimi/DeepSeek 等 API 只能作为后续黑盒扩展，不应与当前白盒方法混在默认 stage。预注册时记录：

```text
estimated_cost = input_tokens * provider_input_price
                 + output_tokens * provider_output_price
```

每次请求还要保存模型版本、地区/价格快照、重试次数和实际 usage。若只测试生成型 Paraphrase/评估，建议先用 50–100 条 smoke，再决定是否扩大；没有稳定 API adapter、缓存和预算上限前，不建议直接做全量公开模型 sweep。
