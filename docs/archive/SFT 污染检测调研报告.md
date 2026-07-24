---
title: SFT 污染检测调研报告
type: research
topic: contamination-detection
created: 2026-06-17
updated: 2026-06-17
status: draft
dingtalk_url:
---

# SFT 污染检测调研报告

> 历史调研归档。本文不代表当前代码、默认方法或阈值；请以根目录 README 和 docs/guide、docs/validation 为准。

> 本报告是 [Benchmark 污染检测调研报告](./Benchmark%20%E6%B1%A1%E6%9F%93%E6%A3%80%E6%B5%8B%E8%B0%83%E7%A0%94%E6%8A%A5%E5%91%8A.md) 的姊妹篇。原报告覆盖预训练阶段，本报告专门处理 SFT 阶段，并在最后一节给出两阶段的异同对比。

## 总结

SFT 污染与预训练污染在**来源、检测方法有效性、影响形式**三个层面都存在质的差异：

*   **来源新路径**：预训练污染主要是 web 爬取被动混入；SFT 污染的最大新风险是**教师模型蒸馏的"二次传导"**（GPT-4 自承包含 MATH/GSM8K，所有 Self-Instruct/Evol-Instruct/Magpie 下游继承），其次是被动渠道改头换面（ShareGPT 用户对话直接灌入 benchmark 题目，Vicuna RTE/QNLI 各被污染 28–33 条）。
*   **MIA 在 SFT 上真正可用**：预训练 MIA AUC < 0.6（接近随机），SFT MIA AUC 通常 0.65–0.95，full fine-tune + 多 epoch 可破 0.99。核心原因是 SFT **数据少 4–6 个数量级 + 多 epoch + 同源 base 是天然 reference**。但 **RLHF/GRPO 一轮即抹除 SFT 阶段的 MIA 信号**（[arXiv:2510.02386](https://arxiv.org/abs/2510.02386)），评测目标必须区分"base+SFT" vs "走完 RLHF 流程"。
*   **影响呈"任务孤岛"特征**：SFT 污染只抬被污染任务本身（如 GSM8K↑，GSMPlus 不动），与预训练污染不同；但 80 条数据即可让 MT-Bench 上涨 45%（KIEval 实证），路径劫持效应远比预训练直接。
*   **联合污染归因目前不可信**：公开文献无成熟方法在最终模型上区分"预训练污染" vs "SFT 污染"，alignment 阶段的分布锐化会污染 base/SFT 差分信号。**保留 pre-SFT checkpoint 是工程上最有效的归因手段**。
*   **检测推荐组合**：n-gram（prompt-only + 白名单 + token-ratio）+ embedding+LLM judge（捕获改写/翻译）+ Paraphrase Stress Test（黑盒诊断）+ MemLens（白盒早期层）+ 双阶段 canary 注入（结构性防御）。单一方法不可靠。

---

## 一、SFT 污染的独特路径

### 1.1 教师模型蒸馏的二次污染（data laundering）

预训练阶段不存在的污染路径。Mansurov et al. (ACL 2025) 系统化命名为 "Data Laundering"：teacher 接触过 benchmark → distillation 把 test 知识"洗"给 student → student 在原 benchmark 上被合法评测。

*   **GPT-4 的"原罪"**：技术报告自承训练中混入 MATH 与 GSM8K（[arXiv:2303.08774](https://arxiv.org/abs/2303.08774)）。所有以 GPT-4 为教师的 Self-Instruct / Evol-Instruct / Magpie / OSS-Instruct 数据，分布上均继承这一污染。
*   **极低比例即显著**：ConStat（[arXiv:2405.16281](https://arxiv.org/abs/2405.16281)）在 ranking distillation 任务中证实 **<0.1% 污染** 即带来显著性能提升。
*   **Preference Leakage**（[arXiv:2502.01534](https://arxiv.org/abs/2502.01534)）：GPT-4 同时充当 SFT 数据生成器 + AlpacaEval/Arena-Hard 裁判时，对"亲生学生"系统性偏好，导致评测分数与真实能力解耦。
*   **反证视角**：OpenReview "Knowledge Distillation as Decontamination?"（[openreview](https://openreview.net/forum?id=W8VCH9x1HZ)）在 8 个分类 benchmark 上发现 laundering 的精度增益在 6/8 任务上统计不显著——效应强度因任务而异。

### 1.2 公开指令数据集污染审计

权威数字主要来自 LLM Decontaminator（[arXiv:2311.04850](https://arxiv.org/abs/2311.04850)）与 Task Contamination（[arXiv:2312.16337](https://arxiv.org/abs/2312.16337)）：

| 数据集 | 实测污染 | 来源 |
| --- | --- | --- |
| **MathInstruct** | 含 769 条 MATH **test** rephrased 样本（占 MATH test 的 **15.4%**） | LLM Decontaminator |
| **WizardLM-Evol-Instruct** | 75 条 MMLU rephrased (0.5%) | LLM Decontaminator |
| **FLAN CoT** | 76 条 MMLU rephrased (0.5%) | LLM Decontaminator |
| **CodeAlpaca** | 21 条 HumanEval rephrased（占 HumanEval **12.8%**） | LLM Decontaminator |
| **Magicoder-Evol-Instruct** | 13 条 HumanEval rephrased (7.9%)；多方独立确认"严重污染" | [arXiv:2404.18824](https://arxiv.org/abs/2404.18824) |
| **ShareGPT (Vicuna 训练集)** | 33 条 RTE、28 条 QNLI、33 条 WNLI 任务样本；Vicuna 三任务分别 +10.6 / +10.0 / +7.7 个百分点 | Task Contamination |
| **Alpaca** | 8 条 SST-2 (+14.6%)、4 条 NewsMet (+7.2%) | Task Contamination |
| **Tulu v2** | 含 114,046 条 ShareGPT，**直接继承** ShareGPT 的 RTE/QNLI 污染 | Tulu v2 report |

未见独立审计的高风险数据集：**UltraChat（由 ChatGPT 自交互生成，继承 GPT 系教师风险）、Magpie（由 Llama-3-Instruct 自吐生成，继承 Llama-3 后训练污染）、Alpaca-GPT4（继承 GPT-4 的 MATH/GSM8K）**。

### 1.3 主动改写"刷分"模式

被广泛抓到的案例：

*   **2024 Open LLM Leaderboard "merge 刷榜"**：ConStat 检测当时榜首的 BARRAHOME/MISTROLL-7b、YAM-PELEG/EXPERIMENT26-7b、MTSAIR/MULTI\_VERSE\_MODEL 在 GSM8k/HellaSwag/ARC 上均显示 **>10% benchmark-specific 污染**。HuggingFace 因此推出 Open LLM Leaderboard v2 直接更换 benchmark。
*   **Llama-2-13B rephrased GSM8K**：0-shot 从 **28.7 → 95.3**；rephrased MMLU 直接拿 **85.9**；CodeLlama 7B 在 rephrased HumanEval 上 **32.9 → 67.7**（超 GPT-4）。n-gram 检测完全无效。
*   **Phi 系列 HumanEval 争议**：Phi-1 原论文自承存在"CodeExercises ↔ HumanEval 记忆污染"风险；NaturalCodeBench（[arXiv:2405.04520](https://arxiv.org/abs/2405.04520)）显示 Phi/Deepseek-Chat/WizardCoder 在 HumanEval 上的排名显著高于 NCB。
*   **TS-Guessing 实证**：ChatGPT 在 MMLU "猜缺失选项"精确匹配率 **52%**、GPT-4 **57%**；Qwen-1.8B 能完整生成 GSM8K 中 223 题的全部 5-gram（[arXiv:2404.18824](https://arxiv.org/abs/2404.18824)）。

### 1.4 SFT pipeline 污染入口（按风险排序）

1.  **教师模型生成**（最高，SFT 独有）—— Self-Instruct/Evol-Instruct/Magpie。n-gram 完全失效。
2.  **网络对话**（高）—— ShareGPT 用户把 benchmark 题目粘进 ChatGPT。
3.  **数学/代码"in-domain"合成**（高）—— MathInstruct 15.4%、Magicoder 严重污染。
4.  **DPO/RLAIF 偏好对**（中高）—— MetaMath 进入 DPO pair 引入 GSM8K 风格。
5.  **人工标注**（中）—— LIMA/Dolly 证明可控，前提是明确反污染指令。
6.  **数据混合 concat**（终末入口）—— Tulu v2 含 114k ShareGPT 即为例。

---

## 二、SFT 污染的影响特性

### 2.1 量化对比：clean vs contaminated SFT

| 实验 | 量化结果 | 来源 |
| --- | --- | --- |
| Llama-2-13B + rephrased GSM8K, 16–64 epoch | 28.7 → **95.3**（+66.6） | [arXiv:2311.04850](https://arxiv.org/abs/2311.04850) |
| 80 条 MT-Bench 模板 + GPT-4 输出 SFT | MT-Bench **+45%**，ARC-Challenge 不变，KIEval 反而下降 | KIEval [arXiv:2402.15043](https://arxiv.org/abs/2402.15043) |
| Mistral/Phi 系列 GSM8K → GSM1k 同分布新题 | 掉 10–13 分；前沿模型与 Llama-2 几乎无 drop | [arXiv:2405.00332](https://arxiv.org/abs/2405.00332) |
| MBPP 实测污染率 | **65.4%**；DeepMind LBPP 对照下 SOTA 模型 HumanEval 比 LBPP 高出最多 43% | [arXiv:2405.11430](https://arxiv.org/abs/2405.11430), [arXiv:2407.07565](https://arxiv.org/abs/2407.07565) |
| Pretraining 注入 5 份 benchmark + 25B 干净 token | 性能膨胀**趋近为零**（catastrophic forgetting） | [arXiv:2601.06103](https://arxiv.org/abs/2601.06103) |
| 同样污染经过 SFT 后 | **复活并被外显** | [arXiv:2601.06103](https://arxiv.org/abs/2601.06103) |

**关键反差**：Pretraining 污染信号会随后续训练衰减；SFT 污染在多 epoch 下却会被显著放大。

### 2.2 任务类型敏感度

| 任务 | 敏感度 | 机制 |
| --- | --- | --- |
| **多选题**（MMLU、C-Eval） | 极高 | 输出空间小（A/B/C/D），SFT 几个 epoch 就把"题面→字母"固化 |
| **数学**（GSM8K、MATH） | 高 | CoT 路径被记忆，但有 GSM1k 类对照能曝光 |
| **代码**（HumanEval、MBPP） | 高 | 函数签名 + docstring 易触发复述；间接污染（合成代码喂回）尤其严重 |
| **开放生成**（MT-Bench、AlpacaEval） | 中（形式不同） | 不是题目泄漏，而是**风格/格式劫持**；80 条数据 +45% |

### 2.3 SFT 污染的"任务孤岛"特征

[arXiv:2601.06103](https://arxiv.org/abs/2601.06103) 的关键对照实验给出最重要的结构性发现：

*   **SFT 污染**：只抬被污染任务本身（GSM8K↑，GSMPlus 不动）；
*   **GRPO 污染**：把泄漏迁移到 GSMPlus / HumanEval 等同族任务，**伪装成真泛化**。

这一发现的实操含义：评测时如果一个 benchmark 显著高于其同族变体（GSM8K vs GSMPlus、MMLU vs MMLU-CF、HumanEval vs LBPP），高度怀疑 SFT 污染；如果连同族变体也涨，则需怀疑 RL 阶段污染或真泛化。

### 2.4 SFT 污染的关键指纹（评测信号）

判断 SFT 污染时按以下信号交叉验证：

1.  格式精准复述（0-shot 输出比 few-shot 还规整地匹配测试集分隔符）
2.  同分布新题大幅掉分（>10 分）
3.  LLM judge 与客观题分歧（MT-Bench 远高于同期客观 benchmark）
4.  选项位置扰动崩盘
5.  CoT 与答案脱钩（推理写错但答对，或推理与官方解逐字相似）
6.  训练前后单 benchmark 跳变 > 同族任务跳变 10 倍
7.  改写题面后分数骤降（>15 分）
8.  题面前 50% 续写能复述原题
9.  训练配方本身高危（3+ epoch、小数据、benchmark-sized）

---

## 三、SFT 数据级检测的调整

原报告的 n-gram / 软匹配 / Embedding 三层结构在 SFT 上仍适用，但参数与策略要调整。

### 3.1 检测对象：prompt-only 是主流

**Tülu 3**（[arXiv:2411.15124](https://arxiv.org/abs/2411.15124)）明确说明只对 prompt（多轮取所有 user turn）做匹配，原因是 SFT response 经常被 LLM 重新生成（rejection sampling、teacher distill），response 端 n-gram overlap 噪声远大于信号。

**例外**：数学和代码任务需要同时匹配 response，因为答案推导步骤会直接命中。Qwen2.5-Math（[arXiv:2409.12122](https://arxiv.org/abs/2409.12122)）和 AceMath（[arXiv:2412.15084](https://arxiv.org/abs/2412.15084)）都做 prompt + solution 双匹配 + LCS≥0.6。

### 3.2 模板假阳问题——SFT 独有

高度模板化的 response（"答案是 A"、固定 system prompt、"```python ... ```"）是 SFT 特有的假阳源。工业实践三类解决方案：

*   **Phi-4 高频 n-gram 白名单**（[arXiv:2412.08905](https://arxiv.org/abs/2412.08905) Appendix B.1）—— 在 Wiki + train 上提取高频 13-gram 作为白名单，例如多选题选项支架 "a i only b ii only c iii only" 直接进白名单。
*   **Tülu 3 token-ratio threshold** —— 单条 train 样本只在 "≥50% test token 都命中" 时才判污染，间接过滤短模板。
*   **Qwen2.5-Math 文本归一化 + LCS≥0.6** —— 归一化把"答：A"和"答案是A"映射到同一表示。

### 3.3 主流团队参数

| 项目 | n | 阈值 | 备注 |
| --- | --- | --- | --- |
| Llama-3 post-training | 8 | 50% test token 命中 | prompt 主匹配 |
| Tülu 3 SFT | 8 | 50% token + 数据集级 >2% 即整集删 | prompt only |
| Qwen2.5-Math | 13 | LCS≥0.6 | normalize + 双字段 |
| Phi-4 hybrid | 13 + 7 | 高频 n-gram 白名单 | 7-gram overlap ratio 兜底 |
| AceMath | 13 | 数学 + LCS；非数学只 13-gram | — |
| Seed-Coder | 10 | any 10-gram overlap | 代码场景 |
| LLM Decontaminator | n/a | embedding top-k + GPT-4 judge | 改写专用 |

**SFT 独有的"放大器"**：多 epoch + 单样本曝光高使污染影响放大 3–5 倍，所以 SFT 阈值应比预训练严。Tülu 3 的"数据集级 >2% 即整集删除"就是保守策略的直接体现。

### 3.4 Embedding + LLM judge——SFT 改写检测的标配

LLM Decontaminator（[lmsys blog](https://www.lmsys.org/blog/2023-11-14-llm-decontaminator/)）的两阶段流程在 SFT 场景几乎是必装：

*   Stage 1: embedding (multilingual-e5 等) → top-k 召回
*   Stage 2: GPT-4 judge 是否 "rephrased / translated"

工业经验阈值：cosine ≥ 0.85 直接删 / 0.75–0.85 进 LLM judge / <0.75 放行。Tülu 3 的负面经验需注意——"难区分 distributional similarity 和真 paraphrase"，所以阈值不能调太低。

### 3.5 中文 SFT 数据的特殊处理

*   **分词**：中文 BPE 无 whitespace，n=8 token 在中文上只覆盖约 12 个汉字过短；建议 **n=10–13 token 或字符级 n=15–20**，配合归一化（全半角、繁简、空格删除）。
*   **跨语言污染**：英文 benchmark 翻译为中文 SFT 是高发污染源（GSM8K→中文小学数学题、MMLU→C-Eval 风格题）。字符 n-gram 完全检测不到，**必须用多语 embedding（BGE / multilingual-e5）+ LLM judge**，怀疑英→中翻译时还应把中文 prompt 机翻回英文再做一次匹配。
*   **改写丰富度**：中文同义改写空间比英文大，n-gram 召回更弱，LLM Decontaminator 相对收益比英文更高。

---

## 四、SFT 模型级检测

### 4.1 MIA 在 SFT 上真正可用

预训练 MIA 在 Pythia 上 AUC 几乎全部 0.5–0.6（[Duan et al., COLM 2024](https://arxiv.org/abs/2402.07841)），SFT 场景完全不同：

| 方法 | 场景 | AUC |
| --- | --- | --- |
| **SPV-MIA** ([NeurIPS 2024](https://arxiv.org/abs/2311.06062)) | GPT-2 / Wikitext-103, LoRA 10 epoch | **0.975** |
| SPV-MIA | LLaMA-7B / Wikitext-103 | **0.951** |
| LiRA-Candidate | 同上 | 0.769 |
| Min-K% | 同上 | 0.658 |
| LOSS | 同上 | 0.614 |
| **Full FT + 10 epoch** ([LoRA-Leak, 2025](https://arxiv.org/abs/2507.18302)) | GPT-2 XL / XSum | **0.993** （TPR@1%FPR = **82.6%**）|
| LoRA + 保守 FT | 同上 | 0.553 |
| Min-K%++ + 同源 pre-trained reference | Llama-2 / MedQA | **0.775** |
| LiRA（含分布访问） | Qwen2.5-7B-Instruct + 受控 SFT 污染 | **0.891** ([arXiv:2510.09259](https://arxiv.org/abs/2510.09259)) |
| Min-K% / Loss / Max-K% 均值 | 同上 | 0.734 |

**SFT MIA 有效性的三个结构性原因**：

1.  **数据少 4–6 个数量级**（10K–1M vs 万亿 token）
2.  **多 epoch**（3–5 vs 1）每条样本曝光高
3.  **base 模型本身是天然 reference**——这是 SFT 相对预训练最大的优势，预训练 reference 需要训独立模型（成本不可承受），SFT 直接用未 SFT 的 base 即可。LoRA 场景更极致：reference 就是"不加 LoRA 权重的同一个模型"。

### 4.2 RLHF/GRPO 抹除 SFT 信号——关键警示

[arXiv:2510.09259](https://arxiv.org/abs/2510.09259) 实证：GRPO 之后所有 MIA 方法 AUROC 一致下降，传统检测器接近随机猜测。[PREMIA](https://arxiv.org/abs/2407.06443) 同样指出 DPO 比 PPO 更易受攻击。

**实操含义**：评测对象是 "base + SFT (无 RL)" 还是 "走完 RLHF 流程"，MIA 结论会截然不同：

*   base + SFT：MIA 可作为强信号
*   含 RLHF/GRPO：MIA 退化为辅助，需配合 Self-Critique 或专用 RL-MIA

### 4.3 SFT 专属检测方法

这一节内容**完全不在原报告中**，是 SFT 评测的新工具箱：

*   **MemLens**（[arXiv:2509.20909](https://arxiv.org/abs/2509.20909), 2025）—— 用 logit lens 分析答案 token 在各层的概率轨迹。**污染样本在早期层就锁定答案（"shortcut reasoning"）**，干净样本是逐层证据累积。LoRA 注入污染因果实验复现同样模式。**关键优势：对 rephrasing 和 perturbation 稳健**，正是 SFT 场景的痛点。白盒方法。
*   **Paraphrase Stress Test**（[arXiv:2510.08616](https://arxiv.org/abs/2510.08616)）—— 同义改写 prompt，观察准确率下降幅度。比纯生成扰动更敏感，因为 SFT 直接优化 P(response|prompt)，prompt 改写打击 SFT 最敏感维度。已知污染对照集上典型 gap 10–30%。黑盒方法。
*   **LogProber**（[arXiv:2408.14352](https://arxiv.org/abs/2408.14352)）—— 区分 "Q-A" / "Q-" / "-A" 三种污染类型，分离 prompt 和 answer 的条件概率。短文本 Q-A 上 AUC > 0.9。
*   **MIA-Tuner**（[AAAI 2025](https://arxiv.org/abs/2408.08661)）—— 把 instruction tuning 本身用作探测机制。AUC 从 0.71 提升到 0.97。
*   **Self-Critique**（[arXiv:2510.09259](https://arxiv.org/abs/2510.09259)）—— 利用 entropy 和 token-level dependency 在 RL 后阶段检测 SFT 污染。在 dual-contamination 场景 AUC 提升 55%。
*   **Fine-tuned Score Deviation (FSD)**（[arXiv:2410.10880](https://arxiv.org/abs/2410.10880)）—— 用少量未见数据微调放大 member/non-member perplexity gap。在 aligned model 上显著优于 Min-K%。
*   **Canary 注入**（[The Canary's Echo, arXiv:2502.14921](https://arxiv.org/abs/2502.14921)）—— SFT 数据中注入 unique (q, r) pair，训练后查询。SFT 多 epoch 让记忆更深，比预训练 canary 更有效。

### 4.4 假阳性来源——SFT 独有

*   **Chat template 复用**：所有 SFT 样本共享 `<|im_start|>system\n...\n<|im_end|>`，模型对模板 token 必然给高概率，Min-K% 类直接被误判为"成员"。规避：**MIA 评分前剥离 template，仅对 response token 算 loss**。
*   **Instruction format 收敛**：见过同格式不同内容样本后，模型对所有"该格式"输入呈 train-like 行为。
*   **领域分布偏移**：评测集与 SFT 数据同领域时 AUC 虚高。规避：严格控制 non-member 集与 member 集 distribution match。

### 4.5 立场论文警示

[Zhang/Das/Kamath/Tramèr SaTML 2025] 明确：**MIA 高 AUC 不能"证明"数据被训练**。即使 AUC=0.9，单条样本判定的 calibration 和置信区间在审计场景都难以站住。这对"作为评测主信号"是关键约束。

---

## 五、联合污染归因——目前不可信

**核心结论**：公开文献中无成熟方法在最终模型上区分"预训练污染"与"SFT 污染"。可行的工程方案围绕"保留 checkpoint + canary 注入 + ConStat 参考"展开。

### 5.1 差分检测——理论可行，实践脆弱

直观思路：对比 Llama-3-Base 与 Llama-3-Instruct 在同一 benchmark 上的 Min-K% 差值 ΔMIA。但 [arXiv:2506.17871](https://arxiv.org/abs/2506.17871) 指出 **alignment 会系统性压缩 token 概率分布**（probability concentration），导致 base vs instruct 的 Min-K% 差值方向甚至会反转。

ΔMIA 可作为弱信号，但 Δ 大不等于 SFT 引入污染，也可能是分布锐化。

### 5.2 阶段归因的信息论极限

[arXiv:2601.06103](https://arxiv.org/abs/2601.06103) 实证：pretraining 注入 5 份 benchmark 后继续训 25B 干净 token，性能膨胀**趋近为零**（catastrophic forgetting）。ICML 2025 "How Much Can We Forget about Data Contamination?" 用 weight decay 累积论证 Llama-3 405B 训练早期数据已被遗忘。

形式化定理虽未给出，但等价含义是：**当 pretraining 污染发生在足够早期且后续训练足够长，与 SFT 污染在最终模型上不可区分**。

### 5.3 实操建议——按可信度排序

1.  **首选：保留 pre-SFT checkpoint**。pretraining 结束后跑一次全 benchmark + MIA，SFT 后再跑一次。Δscore 与 ΔMIA 是阶段归因最可信的证据。**没有 checkpoint 时一切归因都是猜测**——比任何算法都重要。建议团队制度化保留。
2.  **Canary 双阶段注入**：分别在 pretraining 和 SFT 数据中混入两组结构不同的 canary（pretraining 用 N-gram 噪声句、SFT 用 instruction-format canary），用 exposure 指标量化各阶段记忆强度。这是**公开文献的空白点**——分别向两阶段注入不同 canary 族的严格实验目前没有完整发表，是潜在的发表方向。
3.  **被动检测组合**（针对已训好模型）：
    *   ConStat 报"是否污染 + 大致类型"（sample/syntax/benchmark 三粒度）
    *   base proxy（同架构公开 base）与 target 的 MIA Δ
    *   若 base proxy 自身 MIA 已偏高（如 Llama-3-Base 在 GSM8K 上 Min-K%++ AUC > 0.7），pretraining 阶段嫌疑增大
4.  **统计特征辅助**：多次 sampling 看 branching factor / token-entropy 分布。SFT 污染样本显著低熵、低 BF；pretraining 污染样本接近正常分布。
5.  **接受不可识别性**：模型经过 merge / 多轮 DPO / RL 后，公开承认"阶段归因不可信，只报总体污染量"，避免虚假阶段结论。

---

## 六、工具与论文索引

### 6.1 SFT 数据级检测（与原报告 §5.1 互补）

| 工具 | 适用 | 来源 |
| --- | --- | --- |
| **LLM Decontaminator** | rephrased / translated SFT 数据，embedding + GPT-4 judge | [lm-sys/llm-decontaminator](https://github.com/lm-sys/llm-decontaminator) |
| **Phi-4 hybrid n-gram** | 高频 n-gram 白名单防模板假阳 | [Phi-4 报告 Appendix B.1](https://arxiv.org/abs/2412.08905) |
| **Tülu 3 去污染脚本** | prompt-only + 50% token threshold + dataset >2% | [Tülu 3 paper](https://arxiv.org/abs/2411.15124) |
| **Qwen2.5-Math 去污染** | 13-gram + LCS≥0.6，数学/代码场景标准 | [Qwen2.5-Math 报告](https://arxiv.org/abs/2409.12122) |

### 6.2 SFT 模型级检测

| 工具 / 方法 | 访问要求 | 来源 |
| --- | --- | --- |
| **SPV-MIA** | 灰盒，LoRA SFT SOTA | [tsinghua-fib-lab/SPV-MIA](https://github.com/tsinghua-fib-lab/ANeurIPS2024_SPV-MIA) |
| **Min-K%++ + base reference** | 灰盒，reference-free / reference-based 通用 | [zjysteven/mink-plus-plus](https://github.com/zjysteven/mink-plus-plus) |
| **MemLens** | 白盒，对 rephrasing 稳健 | [arXiv:2509.20909](https://arxiv.org/abs/2509.20909) |
| **Paraphrase Stress Test** | 黑盒，性价比最高 | [arXiv:2510.08616](https://arxiv.org/abs/2510.08616) |
| **LogProber** | 灰盒，区分 Q-A 污染类型 | [arXiv:2408.14352](https://arxiv.org/abs/2408.14352) |
| **MIA-Tuner** | 灰盒，LLM 自检测 | [AAAI 2025](https://arxiv.org/abs/2408.08661) |
| **Self-Critique** | 黑盒，RL 后阶段唯一可用 | [arXiv:2510.09259](https://arxiv.org/abs/2510.09259) |
| **LRM Detection Arena** | 综合 benchmark | [ASTRAL-Group/LRM\_Conta\_Detection\_Arena](https://github.com/ASTRAL-Group/LRM_Conta_Detection_Arena) |

### 6.3 SFT 影响诊断对照集

| 对照集 | 用途 |
| --- | --- |
| **MMLU-CF** ([arXiv:2412.15194](https://arxiv.org/abs/2412.15194)) | 无污染 MMLU 对照 |
| **GSM1k** ([arXiv:2405.00332](https://arxiv.org/abs/2405.00332)) | GSM8K 同分布等难度，不公开发布 |
| **LBPP** ([arXiv:2407.07565](https://arxiv.org/abs/2407.07565)) | HumanEval / MBPP 对照 |
| **MHPP** ([arXiv:2405.11430](https://arxiv.org/abs/2405.11430)) | MBPP 实测 65.4% 污染的反衬 |
| **GSMPlus** | GSM8K 同族任务，验证孤岛特征 |
| **LiveBench** ([arXiv:2406.19314](https://arxiv.org/abs/2406.19314)) | 月更滚动，天然 contamination-free |

---

## 七、SFT 评测链路建议

### 7.1 数据级检测 pipeline

针对 10K–1M 规模、含中英、prompt-response 结构的 SFT 数据：

1.  **预处理**：抽 prompt（多轮取所有 user turn）；归一化（标点、全半角、繁简、空白）。
2.  **白名单挖掘**：在 train + Wiki 上提取 top-5000 高频 n-gram 作为模板白名单。
3.  **n-gram 第一道（高召回）**：
    *   英文 prompt：8-gram + 50% token threshold
    *   中文 prompt：13-token 或 20-字符 n-gram + 白名单
    *   数学/代码 response 补做 10-gram
4.  **LCS 兜底**：阈值 0.6，抓"换数字、换变量名"型污染。
5.  **Embedding + LLM judge**：multilingual-e5-large 索引 benchmark prompt，cosine ≥ 0.85 直接删 / 0.75–0.85 进 LLM judge。跨语言场景把中文 prompt 机翻回英文再做一次。
6.  **数据集级裁决**：单 source dataset 命中率 >2% 整集删除。
7.  **审计**：记录每条删除样本的命中规则。

### 7.2 模型级检测 pipeline

```plaintext
自研 SFT 模型 + benchmark 参考集
    │
    ├─ 如果有 pre-SFT checkpoint 保留 ────┐
    │                                      │
    │  信号 1: ΔScore (SFT - Base) on benchmark vs 同族变体
    │  信号 2: SPV-MIA 或 Min-K%++ + base reference (实例级)
    │  信号 3: MemLens 早期层概率轨迹 (白盒可用时)
    │  信号 4: Paraphrase Stress Test (worst-case accuracy drop)
    │  信号 5: 同分布新题对照（GSM1k 风格、MMLU-CF）
    │  信号 6: LogProber 区分 Q-A 污染类型
    │  辅助: Min-K%++ (reference-free fallback)
    │
    ├─ 如果模型已走完 RLHF/GRPO ──────────┐
    │  Min-K% / SPV-MIA 信号被抹除
    │  改用 Self-Critique + Paraphrase Test
    │  增加同族变体对照（验证孤岛特征）
    │
    └─ 综合判定:
         - 同族变体落差 >10 分 且 ≥2 个 MIA/probe 信号阳性 → 红灯
         - 仅 1 个信号阳性 → 黄灯（人工复核）
         - 同族落差 <3 分 且 MIA 接近随机 → 绿灯
         - Paraphrase drop >15% → 额外标记"模板劫持风险"
```

### 7.3 结构性防御（事前 + 事中）

预测自研模型未来要做污染审计的话，**事前的制度化措施比事后检测可靠得多**：

1.  **保留 pre-SFT checkpoint**：是阶段归因的唯一可靠手段
2.  **双阶段 canary 注入**：pretraining 与 SFT 注入不同结构的 canary，作为团队 "contamination meter" 基线
3.  **同分布对照集自建**：每个内部 benchmark 配一个 held-out 同分布变体（仿 GSM1k）
4.  **数据 pipeline 透明化**：明确标注每个 SFT 数据源是 GPT-4 蒸馏、ShareGPT、人工标注还是公开集，便于追溯

---

## 八、与预训练污染的异同总结

| 维度 | 预训练阶段 | SFT 阶段 |
| --- | --- | --- |
| **主要污染来源** | Web 爬取被动混入 | 教师模型蒸馏（独有）+ 主动改写 + ShareGPT 用户对话 |
| **典型污染规模** | 占总训练 0.01–1% | 公开指令集 0.5%–15.4%；极端案例 100% rephrased test set |
| **数据规模** | 万亿 token | 10K–1M 样本（小 4–6 个数量级）|
| **训练 epoch** | 1 | 3–5 |
| **n-gram 检测** | 必备（FM-index 已验证 83TB），主流方法 | 适用但需调整：prompt-only、白名单、token-ratio 收紧 |
| **改写/翻译检测** | 软匹配 + embedding | 同左但收益更高（改写比例更大） |
| **MIA 有效性** | AUC < 0.6 接近随机（Duan COLM 2024）| AUC 0.65–0.99（多 epoch + 小数据 + 天然 reference）|
| **MIA 最强方法** | LiRA 理论最优但不实际；Min-K%++ 实践首选 | SPV-MIA + base reference；full FT 多 epoch 接近 0.99 |
| **失效场景** | 时间分布偏移虚高、1 epoch 信号弱 | **RLHF/GRPO 一轮即抹除**；chat template 假阳 |
| **影响形式** | 知识层面（记住事实可能仍泛化） | **任务孤岛**（只抬被污染任务，不抬同族变体） |
| **路径劫持** | 弱（事实记忆，inference 路径不一定调用） | **强**（直接优化 prompt→answer，正是 inference 调用路径） |
| **极端案例** | MMLU web 检测 13.8% / 行为探测 72.5% | Llama-2 rephrased GSM8K：28.7 → 95.3 |
| **典型对照集** | LiveBench、MMLU-CF、LiveCodeBench | + GSM1k、LBPP、MHPP、同族变体 |
| **跨语言污染** | 已验证（[Yao EMNLP 2024](https://arxiv.org/abs/2406.13236)） | 同左且更普遍（英文 benchmark 翻为中文 SFT 极常见） |
| **黑盒检测推荐** | Guided Instruction、DCQ、TS-Guessing | + Paraphrase Stress Test、Self-Critique |
| **白盒独有方法** | — | **MemLens 早期层概率轨迹** |
| **结构性防御** | 滚动更新 benchmark | **保留 pre-SFT checkpoint + 双阶段 canary** |
| **遗忘速率** | 25B 干净 token 后污染信号趋近零 | 多 epoch 反而放大记忆；但 RL 抹除 |
| **阶段归因** | — | 联合污染场景下**目前不可信** |

### 关键判断

1.  **MIA 在 SFT 上才真正可用**，是 SFT 评测相对预训练最大的"能力升级"，但 RLHF 一轮即抹除信号——使用前必须明确目标模型在 RL 之前还是之后。
2.  **教师模型蒸馏是预训练完全没有的新风险**，且 n-gram 完全无法检测，必须用 embedding + LLM judge 配合教师模型链路追溯。
3.  **SFT 污染的"任务孤岛"特征**（[arXiv:2601.06103](https://arxiv.org/abs/2601.06103)）是评测时最有效的诊断指纹——同族变体对照应当成为 SFT 评测的标配。
4.  **阶段归因目前无解**——这是 TODO，工程上唯一可靠的应对是制度化保留 pre-SFT checkpoint 和双阶段 canary 注入。

---

## 九、核心论文清单

### SFT 污染来源 / 影响
*   Yang et al. "Rethinking Benchmark and Contamination with Rephrased Samples" — [arXiv:2311.04850](https://arxiv.org/abs/2311.04850)（Llama-rephraser 极端实验，公开数据集污染审计）
*   Li & Flanigan "Task Contamination" — [arXiv:2312.16337](https://arxiv.org/abs/2312.16337)（Alpaca/Vicuna 逐条审计）
*   Zhou et al. "Don't Make Your LLM an Evaluation Benchmark Cheater" — [arXiv:2311.01964](https://arxiv.org/abs/2311.01964)
*   Mansurov et al. "Data Laundering" — [ACL 2025](https://aclanthology.org/2025.acl-long.407.pdf)（蒸馏污染传导框架）
*   Li et al. "Preference Leakage" — [arXiv:2502.01534](https://arxiv.org/abs/2502.01534)
*   Xu et al. "Benchmarking Benchmark Leakage in LLMs" — [arXiv:2404.18824](https://arxiv.org/abs/2404.18824)
*   Yu et al. "KIEval" — [arXiv:2402.15043](https://arxiv.org/abs/2402.15043)（80 条样本拉升 MT-Bench 45% 实证）
*   Scale AI "GSM1k" — [arXiv:2405.00332](https://arxiv.org/abs/2405.00332)
*   "The Impact of Post-training on Data Contamination" — [arXiv:2601.06103](https://arxiv.org/abs/2601.06103)（**SFT 任务孤岛 vs RL 同族迁移的关键对照**）
*   Riddell et al. "Quantifying Contamination in Evaluating Code Generation" — [arXiv:2403.04811](https://arxiv.org/abs/2403.04811)
*   Magar & Schwartz "From Memorization to Exploitation" — [arXiv:2203.08242](https://arxiv.org/abs/2203.08242)

### SFT MIA
*   Fu et al. "SPV-MIA" — [NeurIPS 2024, arXiv:2311.06062](https://arxiv.org/abs/2311.06062)（SFT MIA SOTA）
*   Ruan et al. "LoRA-Leak" — [arXiv:2507.18302](https://arxiv.org/abs/2507.18302)（LoRA SFT 系统评测）
*   Mireshghallah et al. EMNLP 2022 — [arXiv:2203.03929](https://arxiv.org/abs/2203.03929)（reference-based 奠基）
*   Zhang et al. "Min-K%++" — [arXiv:2404.02936](https://arxiv.org/abs/2404.02936)
*   Kandpal et al. "User Inference Attacks" — [EMNLP 2024, arXiv:2310.09266](https://arxiv.org/abs/2310.09266)
*   Feng et al. "PREMIA" — [arXiv:2407.06443](https://arxiv.org/abs/2407.06443)
*   Fu et al. "MIA-Tuner" — [AAAI 2025, arXiv:2408.08661](https://arxiv.org/abs/2408.08661)
*   "Detecting Data Contamination from RL Post-training" — [arXiv:2510.09259](https://arxiv.org/abs/2510.09259)
*   Duan et al. COLM 2024 对照基线 — [arXiv:2402.07841](https://arxiv.org/abs/2402.07841)

### SFT 专属检测
*   "MemLens" — [arXiv:2509.20909](https://arxiv.org/abs/2509.20909)
*   "Paraphrase Stress Tests" — [arXiv:2510.08616](https://arxiv.org/abs/2510.08616)
*   "LogProber" — [arXiv:2408.14352](https://arxiv.org/abs/2408.14352)
*   "Fine-tuned Score Deviation" — [arXiv:2410.10880](https://arxiv.org/abs/2410.10880)
*   "Fragility of LRM Contamination Detection" — [arXiv:2510.02386](https://arxiv.org/abs/2510.02386)（**RL 抹除 SFT 信号实证**）
*   "The Canary's Echo" — [arXiv:2502.14921](https://arxiv.org/abs/2502.14921)

### 工业实践（数据级）
*   Tülu 3 — [arXiv:2411.15124](https://arxiv.org/abs/2411.15124)
*   Llama 3 — [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)
*   Qwen2.5-Math — [arXiv:2409.12122](https://arxiv.org/abs/2409.12122)
*   Phi-4 — [arXiv:2412.08905](https://arxiv.org/abs/2412.08905)
*   AceMath — [arXiv:2412.15084](https://arxiv.org/abs/2412.15084)
*   LLM Decontaminator (LMSYS) — [arXiv:2311.04850](https://arxiv.org/abs/2311.04850)

### 联合污染归因
*   "Impact of Post-training on Data Contamination" — [arXiv:2601.06103](https://arxiv.org/abs/2601.06103)
*   "How Much Can We Forget about Data Contamination?" — ICML 2025
*   "Mosaic Memory of LLMs" — Nature Communications 2026
*   "Landscape of Memorization in LLMs (SoK)" — [arXiv:2507.05578](https://arxiv.org/abs/2507.05578)

### 对照集
*   MMLU-CF — [arXiv:2412.15194](https://arxiv.org/abs/2412.15194)
*   LiveBench — [arXiv:2406.19314](https://arxiv.org/abs/2406.19314)
*   LBPP — [arXiv:2407.07565](https://arxiv.org/abs/2407.07565)
*   MHPP — [arXiv:2405.11430](https://arxiv.org/abs/2405.11430)

---

## TODO

1.  **双阶段 canary 注入实验**：分别向 pretraining 与 SFT 注入不同 canary 族、在最终模型上做严格阶段归因——目前公开文献空白，自研实验有发表价值
2.  **中文 SFT 数据 + Magpie 类自蒸馏的污染审计**：UltraChat、Magpie 系列、Alpaca-GPT4 缺乏独立审计
3.  **MemLens 在生成式任务的扩展**：目前只在数值/可枚举答案验证过，开放生成验证待补
4.  **多次 SFT + DPO + merge 后的归因**：理论上不可识别，但工程边界在哪？需自建对照实验
