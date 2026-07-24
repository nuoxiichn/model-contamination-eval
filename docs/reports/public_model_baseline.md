# 公开模型数据污染检测报告

> 报告快照：2026-07-08 → 2026-07-24。本文用于公开模型间的相对比较和方法回归，不是已校准的污染概率或发布门禁。原始结果位于 `experiments/2026-07-08_market_model_sweep/outputs/`（本地忽略文件）。

## 1. 摘要

本报告汇总 `2026-07-08_market_model_sweep` 对 16 个公开模型、8 类 benchmark 的交叉污染检测结果。实验使用 CoDeC、Min-K%++、paraphrase stress 和 option permutation 四类检测；对 instruct/chat 模型另外使用同源 base 模型作为 reference 做 SPV-MIA。

结论应理解为跨模型相对信号，而不是“模型一定训练过该 benchmark”的绝对证明。当前 sweep 没有统一的 positive control，因此报告保留原始信号和方法 verdict，不把 `dirty` 直接等同于事实性数据泄漏。

主要结论：

- CoDeC 的跨 benchmark 平均信号最高的是 OLMo-2-7B-Instruct（0.850）、Yi-1.5-9B-Chat（0.752）和 Qwen2.5-7B-Instruct（0.722）。
- DeepSeek base/chat 的 CoDeC 平均信号最低（分别为 0.110、0.169）；本次 DeepSeek logits/token 边界问题已修复，相关 28 个失败格均已重跑成功。
- option permutation 在几乎所有模型上都给出较高泄漏率，尤其 MMLU-Pro（约 0.935–0.985）；该 benchmark 的 10 选项随机排列接近信号饱和，不适合做模型间绝对排序。
- Min-K%++ 在有干净 control 的 MMLU/MMLU-Pro 上区分度较明显；无 control 的 math、math-500、evalplus 结果统一标为 `inconclusive`，不应单独下结论。
- SPV-MIA 本轮主要是无 control 的 mean-only 结果，全部为 `inconclusive`；MMLU-Pro 因单字母 completion 已从当前 SPV 任务矩阵移除。

## 2. 实验范围与数据状态

| 项目 | 设置 |
|---|---|
| 模型 | 16 个（8 个 base + 8 个 instruct/chat） |
| Benchmark | gsm8k、math-500、math、mmlu-pro、mmlu、gsm-plus、mgsm、evalplus；MC permutation 另含 mmlu-cf、gpqa、gpqa-diamond、mmmlu |
| 每个 target 样本数 | 通常 200；evalplus 为 164；GPQA-Diamond 实际有效题数为 198 |
| 精度/硬件 | bfloat16，8 张 MetaX GPU |
| control | gsm8k→gsm1k，mmlu-pro/mmlu→mmlu-cf；其余部分方法无 control |
| 结果文件 | [`sweep_results.jsonl`](../../experiments/2026-07-08_market_model_sweep/outputs/sweep_results.jsonl) |

当前配置定义的 440 个 key 全部已有非 error 的终态记录：182 clean、121 dirty、45 suspect、88 inconclusive、4 reused。JSONL 中额外保留了 7 条历史 SPV-MIA×MMLU-Pro 记录，其中 2 条是旧的 gated 加载失败；它们不属于当前配置矩阵。

## 3. 模型清单

| 家族 | Base | Instruct/Chat |
|---|---|---|
| DeepSeek | deepseek-7b | deepseek-7b-chat |
| Gemma 2 | gemma-2-9b | gemma-2-9b-it |
| Llama 3.1 | llama-3.1-8b | llama-3.1-8b-instruct |
| Mistral | mistral-7b-v0.3 | mistral-7b-instruct |
| OLMo 2 | olmo-2-7b | olmo-2-7b-instruct |
| Qwen2 | qwen2-7b | qwen2-7b-instruct |
| Qwen2.5 | qwen2.5-7b | qwen2.5-7b-instruct |
| Yi 1.5 | yi-1.5-9b | yi-1.5-9b-chat |

## 4. 方法说明

- **CoDeC**：比较无 context 与加入同分布 context 后的 completion 平均 log-prob，报告 `Δ<0` 的样本比例。当前阈值为 ≥0.80 dirty、≥0.60 suspect。
- **Min-K%++**：对 token-level chosen log-prob 做归一化，取 bottom 20%。有 control 时报告 AUC；无 control 时只报告 mean-only 位置统计，因此标为 inconclusive。
- **Paraphrase stress**：对数学/CoT completion 做规则型词替换，比较原文与改写后的 log-prob 差异。该方法不用于 MC 和代码 completion。
- **Option permutation**：枚举或采样选项排列，检测最大排列 log-prob 是否为异常值。4 选项 benchmark 通常全枚举 24 个排列；MMLU-Pro 使用最多 120 个排列。
- **SPV-MIA**：instruct/chat target 与同源 base reference 比较 paraphrase 后的局部概率变化。本轮没有足够的 positive control，主要为 mean-only 弱信号。

## 5. 跨模型总体结果

### 5.1 CoDeC 平均信号

平均值是 8 个 CoDeC benchmark 的简单均值，仅用于相对排序，不是校准后的污染概率。

| 模型 | CoDeC 均值 | Base→Instruct/Chat 变化 |
|---|---:|---:|
| olmo-2-7b-instruct | **0.850** | +0.421（相对 olmo-2-7b） |
| yi-1.5-9b-chat | **0.752** | +0.313 |
| qwen2.5-7b-instruct | **0.722** | +0.075 |
| qwen2.5-7b | 0.646 | — |
| qwen2-7b | 0.582 | — |
| mistral-7b-instruct | 0.561 | +0.299 |
| qwen2-7b-instruct | 0.558 | -0.024 |
| yi-1.5-9b | 0.439 | — |
| olmo-2-7b | 0.429 | — |
| llama-3.1-8b-instruct | 0.393 | +0.013 |
| llama-3.1-8b | 0.380 | — |
| gemma-2-9b | 0.264 | — |
| mistral-7b-v0.3 | 0.263 | — |
| gemma-2-9b-it | 0.188 | -0.076 |
| deepseek-7b-chat | 0.169 | +0.059 |
| deepseek-7b | 0.110 | — |

### 5.2 Base→Instruct/Chat 对照

同源差值具有比跨家族比较更清晰的解释价值，但仍会受到 chat template、训练格式和模型能力差异影响。

| 家族 | Base | Instruct/Chat | 差值 |
|---|---:|---:|---:|
| OLMo 2 | 0.429 | 0.850 | **+0.421** |
| Yi 1.5 | 0.439 | 0.752 | **+0.313** |
| Mistral | 0.263 | 0.561 | **+0.299** |
| Qwen2.5 | 0.646 | 0.722 | +0.075 |
| DeepSeek | 0.110 | 0.169 | +0.059 |
| Llama 3.1 | 0.380 | 0.393 | +0.013 |
| Qwen2 | 0.582 | 0.558 | -0.024 |
| Gemma 2 | 0.264 | 0.188 | -0.076 |

## 6. 方法间观察

### 6.1 CoDeC

CoDeC 对数学/CoT benchmark 的模型差异最明显。OLMo-2-7B-Instruct 在 gsm8k、math、math-500、gsm-plus、mgsm 上均接近 0.9–0.985；Yi-1.5-9B-Chat 和 Qwen2.5-7B-Instruct 也呈现较高信号。DeepSeek 两个版本整体最低，且在 evalplus、gsm-plus、mgsm 上接近 0。

MMLU-Pro 与 mmlu 的 CoDeC 信号普遍低于数学 benchmark；这符合多主题 MC 数据集的已知局限：随机 context 未必与目标题目共享足够结构。

### 6.2 Min-K%++

有 control 的结果更适合作为相对证据：

- MMLU-Pro：gemma-2-9b（0.866）、gemma-2-9b-it（0.799）、mistral-7b-instruct（0.756）、deepseek-7b（0.746）较高；llama-3.1-8b base（0.200）和 olmo-2-7b-instruct（0.252）较低。
- MMLU：yi-1.5-9b（0.873）、qwen2-7b（0.777）、gemma-2-9b（0.743）较高；OLMo-2-7B-Instruct（0.278）和 DeepSeek base（0.558）相对较低。
- gsm8k：所有模型大致在 0.32–0.56，区分度有限。

math、math-500、evalplus 没有干净 control，本报告不把其 mean-only 数值解释为 AUC 或绝对污染判断。

### 6.3 Option permutation

该方法在 MC benchmark 上对所有模型都产生较高泄漏率，尤其 MMLU-Pro：

- base 模型范围约 0.955–0.985；
- instruct/chat 模型范围约 0.935–0.965；
- 因而 MMLU-Pro 主要表现为排列统计饱和，不适合作为跨模型精细排序指标。

GPQA/GPQA-Diamond 的结果约为 0.37–0.61，仍有模型间差异，但由于没有统一 positive control，暂不作绝对“已污染”结论。

### 6.4 Paraphrase stress

绝大多数结果接近 0，说明规则型短 paraphrase 在本轮模型/数据设置下只提供弱信号。相对较高的值包括：olmo-2-7b 的 gsm8k/mgsm（0.080）、qwen2-7b 的 math（0.121）、yi-1.5-9b 的 gsm-plus（0.080）和 qwen2.5-7b-instruct 的 gsm-plus（0.055）。这些值仍不足以脱离其他方法单独解释。

### 6.5 SPV-MIA

SPV-MIA 仅对 instruct/chat 模型运行，且当前 benchmark 主要是 math-500、math、evalplus。由于缺少 positive control，结果均为 `inconclusive`。例如：

- olmo-2-7b-instruct：math-500 -43.05、math -24.60；
- yi-1.5-9b-chat：math-500 -75.62、math -51.91；
- deepseek-7b-chat：math-500 -24.43、math -16.57；
- gemma-2-9b-it：math-500 +2.43、math +8.36。

这些数值可以作为后续 control/calibration 的原始输入，但不能直接转成污染标签。

## 7. 完整信号矩阵

以下数值为最新 key 的 `signal`，保留三位小数；完整 evidence、样本级数组和 verdict 见 JSONL。

### 7.1 CoDeC

| 模型 | gsm8k | math-500 | math | mmlu-pro | mmlu | gsm-plus | mgsm | evalplus |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deepseek-7b | .015 | .162 | .106 | .211 | .350 | .000 | .015 | .018 |
| deepseek-7b-chat | .110 | .242 | .187 | .299 | .344 | .050 | .110 | .006 |
| gemma-2-9b | .225 | .242 | .207 | .436 | .343 | .180 | .225 | .250 |
| gemma-2-9b-it | .165 | .242 | .232 | .169 | .376 | .105 | .165 | .049 |
| llama-3.1-8b | .350 | .556 | .455 | .320 | .372 | .280 | .350 | .360 |
| llama-3.1-8b-instruct | .410 | .586 | .540 | .268 | .417 | .320 | .410 | .195 |
| mistral-7b-v0.3 | .140 | .343 | .338 | .337 | .437 | .160 | .140 | .207 |
| mistral-7b-instruct | .505 | .672 | .677 | .674 | .674 | .480 | .505 | .305 |
| olmo-2-7b | .660* | .505* | .313 | .407* | .404 | .380 | .540 | .226* |
| olmo-2-7b-instruct | .985 | .919 | .894 | .704 | .694 | .975 | .985 | .646 |
| qwen2-7b | .615 | .707 | .742 | .349 | .578 | .565 | .615 | .488 |
| qwen2-7b-instruct | .580 | .687 | .702 | .400 | .639 | .475 | .580 | .402 |
| qwen2.5-7b | .710 | .808 | .828 | .385 | .444 | .735 | .700 | .561 |
| qwen2.5-7b-instruct | .770 | .823 | .869 | .508 | .606 | .760 | .780 | .659 |
| yi-1.5-9b | .380 | .662 | .551 | .477 | .387 | .360 | .380 | .317 |
| yi-1.5-9b-chat | .840 | .843 | .894 | .708 | .613 | .760 | .840 | .518 |

`*` 表示复用 2026-07-06 已完成的 OLMo-2-7B base CoDeC 结果，而非本轮重新前向。

### 7.2 Min-K%++

| 模型 | gsm8k | math-500 | math | mmlu-pro | mmlu | evalplus |
|---|---:|---:|---:|---:|---:|---:|
| deepseek-7b | .439 | -1.677 | -1.709 | .746 | .558 | -1.667 |
| deepseek-7b-chat | .498 | -3.824 | -3.986 | .695 | .555 | -5.700 |
| gemma-2-9b | .318 | -1.921 | -1.933 | .866 | .743 | -1.552 |
| gemma-2-9b-it | .479 | -4.177 | -4.392 | .799 | .730 | -4.264 |
| llama-3.1-8b | .421 | -1.826 | -2.010 | .200 | .671 | -1.354 |
| llama-3.1-8b-instruct | .508 | -2.218 | -2.190 | .720 | .591 | -2.884 |
| mistral-7b-v0.3 | .434 | -1.416 | -1.454 | .727 | .610 | -1.200 |
| mistral-7b-instruct | .429 | -2.232 | -2.458 | .756 | .758 | -2.518 |
| olmo-2-7b | .420 | -1.940 | -1.970 | .562 | .446 | -2.894 |
| olmo-2-7b-instruct | .562 | -7.983 | -8.353 | .252 | .278 | -17.815 |
| qwen2-7b | .482 | -2.200 | -2.330 | .730 | .777 | -1.122 |
| qwen2-7b-instruct | .427 | -4.073 | -4.056 | .262 | .688 | -2.462 |
| qwen2.5-7b | .481 | -1.586 | -1.662 | .631 | .738 | -1.174 |
| qwen2.5-7b-instruct | .421 | -3.762 | -3.955 | .399 | .671 | -4.572 |
| yi-1.5-9b | .490 | -1.532 | -1.355 | .951 | .873 | -.893 |
| yi-1.5-9b-chat | .552 | -4.657 | -4.497 | .605 | .606 | -1.955 |

正数的 MMLU/MMLU-Pro 列为 control-backed AUC；其余列为 mean-only 原始统计，不能横向当作同一量纲比较。

### 7.3 Option permutation

| 模型 | mmlu-pro | mmlu | mmlu-cf | gpqa | gpqa-diamond | mmmlu |
|---|---:|---:|---:|---:|---:|---:|
| deepseek-7b | .980 | .450 | .380 | .535 | .495 | .160 |
| deepseek-7b-chat | .935 | .365 | .400 | .445 | .475 | .140 |
| gemma-2-9b | .980 | .600 | .520 | .520 | .515 | .595 |
| gemma-2-9b-it | .955 | .560 | .375 | .470 | .470 | .550 |
| llama-3.1-8b | .970 | .575 | .455 | .570 | .535 | .395 |
| llama-3.1-8b-instruct | .970 | .415 | .345 | .450 | .450 | .340 |
| mistral-7b-v0.3 | .985 | .610 | .465 | .565 | .545 | .460 |
| mistral-7b-instruct | .960 | .500 | .415 | .495 | .500 | .425 |
| olmo-2-7b | .965 | .330 | .445 | .510 | .510 | .385 |
| olmo-2-7b-instruct | .955 | .375 | .405 | .415 | .374 | .365 |
| qwen2-7b | .965 | .505 | .395 | .605 | .581 | .495 |
| qwen2-7b-instruct | .960 | .435 | .380 | .520 | .505 | .370 |
| qwen2.5-7b | .955 | .610 | .385 | .570 | .556 | .645 |
| qwen2.5-7b-instruct | .965 | .590 | .375 | .480 | .510 | .470 |
| yi-1.5-9b | .970 | .625 | .370 | .550 | .495 | .660 |
| yi-1.5-9b-chat | .955 | .420 | .395 | .525 | .540 | .395 |

### 7.4 SPV-MIA（仅 instruct/chat）

| 模型 | math-500 | math | evalplus | 解释 |
|---|---:|---:|---:|---|
| deepseek-7b-chat | -24.427 | -16.570 | -6.446 | mean-only，inconclusive |
| gemma-2-9b-it | +2.427 | +8.358 | +10.680 | mean-only，inconclusive |
| llama-3.1-8b-instruct | -4.757 | -0.950 | +13.293 | mean-only，inconclusive |
| mistral-7b-instruct | -21.744 | -14.224 | -3.383 | mean-only，inconclusive |
| olmo-2-7b-instruct | -43.053 | -24.601 | -11.818 | mean-only，inconclusive |
| qwen2-7b-instruct | -7.316 | -3.856 | -2.112 | mean-only，inconclusive |
| qwen2.5-7b-instruct | -24.876 | -19.731 | -10.228 | mean-only，inconclusive |
| yi-1.5-9b-chat | -75.620 | -51.907 | -6.753 | mean-only，inconclusive |

## 8. 工程状态与可复现性

- DeepSeek logits 问题：DeepSeek tokenizer 将 `"\n\n"` 编成空序列，已通过显式 BOS 边界修复；28 个 DeepSeek 失败格全部成功重跑。
- GPQA-Diamond OOM：新增 tail logits，并将排列路径默认 micro-batch 调为 2；Gemma base/IT 两格已成功完成，结果分别为 0.51515 和 0.46970。
- gated 模型：Llama 3.1 base/instruct、Gemma 2 base/IT 的本地缓存加载均已验证成功。JSONL 中仍有两条历史 `spv_mia × mmlu-pro` load_error，但该 benchmark 已不在当前 SPV 配置内。
- 代码验证：完整测试曾通过 `155 passed, 11 skipped`；tail logits 修改后，HFLocalModel/option-permutation 相关测试 `12 passed`，`ruff` 通过。
- 运行结束后 8 张 GPU 均已释放。

## 9. 解读边界与下一步

1. 补充统一 positive/negative control，尤其为 math、math-500、evalplus 和 GPQA 建立与 target 同分布的干净对照。
2. 对 CoDeC 和 option permutation 做 bootstrap confidence interval，而不是只看单个比例。
3. 将 MMLU-Pro 从 option permutation 的主排序中降级为饱和度诊断项。
4. 对高信号模型（OLMo-2-Instruct、Yi-1.5-Chat、Qwen2.5-Instruct、Mistral-Instruct）抽取样本级 evidence，检查信号是否由少量题目或格式模板驱动。
5. 对 instruct/base 差异做统一 chat-template 审计，避免把模板变化误判成 contamination signal。
