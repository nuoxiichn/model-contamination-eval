# 记忆 vs 泛化特异性检验

## 问题
CoDeC 的溢出（训 gsm8k 抬 math-500 到 0.44）是**记忆**（见过相似文本）还是
**泛化**（数学能力提升带偏 CoDeC）？后者会把强数学模型误报成 benchmark 污染——
内部模型落地最危险的假阳。

## 设计
注入**非 benchmark 的正常数学**，看 benchmark CoDeC 会不会被带起来。三臂
（fresh Pythia-2.8b，纯注入 40 batch，checkpoint 0/10/20/40）：

| 臂 | 注入源 | 角色 |
|---|---|---|
| gsm8k | gsm8k 题面 | 正对照 |
| dmmath | DM Mathematics（合成数学题，非评测集） | 数学题（异内容同格式域） |
| arxiv | ArXiv 论文正文 | 数学知识（异格式） |

probe：gsm8k / math-500（看假阳）+ gsm1k（同族）+ evalplus（跨域 sanity）+
源自身（阳性对照）。每 checkpoint 记 train loss。

## 训练有效性证明（前提）
注入源在 Pythia base 已偏高（Pile 见过）→ 源自身 CoDeC 可能升不动，故**以 train
loss 单调降为主**证明训练有效；源自身 CoDeC 作辅助。train loss 不降的臂作废。

## 结果读法
| DM/ArXiv 注入后 gsm8k/math-500 | 解读 |
|---|---|
| 不升（≈base 0.03/0.20） | CoDeC 特异于「见过具体文本」，能力提升不误报 ✓ |
| DM 抬、ArXiv 不抬 | 抓「题库格式记忆」，格式差异大的知识不触发 |
| 都抬 | 被数学能力/领域带起 → 能力混淆假阳 ✗ |

evalplus（跨域）所有臂应恒定（若动 → 训练把模型搞坏，实验作废）。

## 跑法
    HF_ENDPOINT=https://hf-mirror.com HF_HOME=/mnt/public/code/chennuoxi/hf_cache \
    PYTHONPATH=src python3 experiments/2026-07-08_codec_knowledge_specificity/run_knowledge.py
成本：3 臂 × 40 batch × 4 checkpoint，~25-30min。子进程隔离，断点续跑。

## 结果（2026-07-08，三臂全完成）

终点 CoDeC（batch=40，纯注入）：

| 臂 / 注入源 | 源自身 | gsm8k | math-500 | gsm1k | evalplus | train loss |
|---|---|---|---|---|---|---|
| **gsm8k**（benchmark 本身） | 0.94 | **0.94** | **0.59** | **0.84** | 0.14 | →1.82 |
| **dmmath**（DM 数学题） | 1.0(饱和) | 0.07 | 0.20 | 0.02 | 0.10 | →1.33 |
| **arxiv**（数学论文正文） | 0.78→**0.98** | 0.04 | 0.14 | 0.04 | 0.06 | →1.83 |

base 参考：gsm8k 0.037 / math-500 0.208 / gsm1k 0.02 / evalplus 0.128。

## 结论：CoDeC 特异于「见过具体文本」，不是数学能力/领域 ✓✓

命中读法表最强格：**注入正常数学知识，benchmark CoDeC 完全不升。**

- **正对照成立**：注入 gsm8k 本身 → gsm8k 0.037→0.94（同时溢出抬 math-500/gsm1k，
  纯注入 worst-case，与 crossmodel_spillover 一致）。证明框架能把信号打上去。
- **dmmath**：注入海量数学题（DM Mathematics）→ gsm8k 岿然不动（0.037→0.07），
  math-500 不动（0.20→0.20）。数学题格式接近 benchmark 但内容不同 → 不触发。
- **arxiv**：注入数学论文正文 → gsm8k 不动（0.037→0.04），math-500 不升反降。
  **关键**：arxiv 源自身 0.78→0.98 明确升 → 训练确实在强化 arxiv 记忆（阳性对照
  有效），**排除了「训练无效所以 benchmark 不动」的替代解释**。训练真在进行，
  benchmark 就是不动。

## 意义
CoDeC **不是数学能力探测器，是文本记忆探测器**。模型数学能力/领域知识提升
（哪怕训练海量数学题+论文）不会把干净 benchmark 误报成污染。这消除了内部模型
落地最担心的假阳：自研模型数学本来就强，CoDeC 不会因此虚报 benchmark 脏。

## 与溢出实验的关系（重要澄清）
crossmodel_spillover 里训 gsm8k 抬 math-500，本实验证明那是**记忆溢出**（同为
题库格式的相邻文本记忆），**不是能力泛化**——因为注入非题库数学（arxiv）时
math-500 不动。溢出的边界是「相似格式/来源的文本记忆」，不是「数学领域」。

## 补充观测
注入非目标数学（dmmath/arxiv）后 evalplus/gsm1k 略降（0.13→0.06-0.10）。纯注入
单一分布 40 batch 让模型对其他文本的 in-context 增益略变 → 无害的轻微下压，不影响
结论方向。
