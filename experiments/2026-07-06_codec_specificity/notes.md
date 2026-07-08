# Specificity / False-Positive 实验

## 问题
CoDeC 会不会对**未训练数据**频繁误报。量化三类假阳风险。

## 三个子实验
**2a 阴性对照面板** — 确证未训练的数据集各测 CoDeC → 干净数据 null 分布。
用 per-sample Δ 做 bootstrap 得每数据集 95% CI（零额外前向）。直接喂 calibration。

**2b 溢出/迁移** — gsm8k 高剂量注入（40 batch 纯 gsm8k）后，测 CoDeC on 未训练
同族（gsm1k）+ 相邻/无关（math-500 / mmlu-pro / evalplus）。确认信号**局域在
gsm8k**、没涂抹到邻居（论文 Fig7）。最重要的假阳风险：把「训了 A」误报成「B 也脏」。
gsm1k（GSM8k 重制干净集）是关键探针——同族但真未训练。

**2c 结构性假阳** — 多来源异质拼盘（gsm8k+mmlu-pro+evalplus+math-500 混采）。
codec.py docstring 已警告：多来源混合会被**人为抬到 ~50%**。量化「干净但杂」能
冲多高 → **suspect 阈值必须设在其之上**，否则干净但杂的 benchmark 稳定触发 SUSPECT。
这是本方法对内部模型落地最现实的翻车点。

> 数据集选择：仅用已注册 normalizer 且能加载的集合
> （gsm8k / math-500 / math / mmlu-pro / gsm1k / evalplus）。
> gsm-plus / humaneval / simpleqa 无 normalizer、mmlu-cf split 名不兼容，均剔除。

## 预期
- 2a：所有未训练数据集 signal < suspect 阈值，构成紧的 null。
- 2b：gsm8k 高、三个未训练邻居低（ΔCoDeC 邻居 ≈ 0）。若邻居也升 → 溢出假阳。
- 2c：异质拼盘 signal 明显高于单一干净数据集（可能逼近 0.5）。记录其绝对值。

## 跑法
    HF_ENDPOINT=https://hf-mirror.com HF_HOME=/mnt/public/code/chennuoxi/hf_cache \
    PYTHONPATH=src python3 experiments/2026-07-06_codec_specificity/run_spec.py

成本：2a/2c 纯评测（复用干净 base）；2b 一次 40-batch finetune + 5 数据集评测。
dev C500 ~40min。

## 结果
（待跑后回填：null 分布 + CI、溢出矩阵、异质拼盘绝对值）
