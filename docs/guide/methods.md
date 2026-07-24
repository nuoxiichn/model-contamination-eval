# 方法说明与适用边界

## 方法矩阵

| 方法 | 核心信号 | 方向 | 最小输入 | 当前推荐场景 | 不应使用的场景 |
|---|---|---|---|---|---|
| CoDeC | 加同分布 context 后目标文本平均 log-prob 下降的比例 | 越高越可疑 | ≥2 条；论文默认约 100 条更稳 | 数学/CoT 和同质文本；当前验证最完整 | 多主题混合、重复样本、没有足够样本 |
| Min-K%++ | bottom-k token 的归一化 log-prob | 分数越高越像 member（实现原始量纲依 benchmark 而异） | log-prob + μ/σ；≥30 条 | 有同分布 control 时算 AUC | 无 control 时把 mean 当绝对污染概率 |
| Option Permutation | 选项排列 log-prob 的最大值是否为离群点 | leak fraction 越高越可疑 | ≥3 选项 MC；≥50 条 | MMLU/GPQA/C-Eval 等 MC | 数学、代码、选项数过少；MMLU-Pro 饱和排序 |
| SPV-MIA | target 与同源 reference 在随机邻居上的 Δpv 差 | Δpv 越负越像 member | target/reference log-prob；completion ≥4 词 | SFT 数学 CoT、同源 base 可用 | MC 单字母、短数字答案、无 control 的绝对裁决 |
| Paraphrase Stress | 原题与规则改写后的性能/概率下降 | drop 越高越可疑 | 当前仅 `math_cot`；≥30 条 | 数学 CoT 的压力测试 | MC、代码、open-generation、远程黑盒 API |

## 组合规则

`reports/trustworthiness.py` 当前把 CoDeC、SPV-MIA、数学 Paraphrase 视为强信号，把 Min-K%++、Option Permutation 视为弱信号。但该分档只用于相对排序；positive control 未完成前，不能将聚合结果宣称为绝对红/黄/绿。

建议至少保留一个独立 control：

- MC：MMLU-Pro ↔ MMLU-CF，或同族 control；
- 数学：GSM8K ↔ GSM1k/GSM-Plus，或预先声明的 clean split；
- SFT：同源 base + clean SFT checkpoint，配合 SPV-MIA。

## 方法实现与配置一致性

注册表中的 `applicable_methods` 是候选元数据。调用方仍须检查：格式、loader 是否实现、模型能力、样本数量、control/reference 是否存在。遇到不满足条件的组合应输出 `INCONCLUSIVE`，不能静默跳过或填 0。
