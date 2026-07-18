# Self-Critique GSM8K RL-MIA 复现

**方法**：Self-Critique（Tao et al. 2025, [arXiv:2510.09259](https://arxiv.org/abs/2510.09259)）
**日期**：2026-07-13　**模型**：Qwen2.5-7B-Instruct　**RL 机器**：MetaX C500 ×8

## 为什么有这个目录（承接 aime 实验的方法学修正）

前置实验 `../2026-07-08_self_critique_aime_repro/` 想在 AIME 上复现 self_critique，两条路都不成立：

1. **RL 注入（GRPO on AIME）失败**：7B 做不对 AIME → reward 全 -1 → 组内 advantage 恒 0 →
   梯度 0 → member 题学不动 → 无 policy collapse → 检测 AUC 0.44≈null（训练失败，非方法失败）。
   训练曲线铁证：critic/rewards/mean 全程 -0.73~-0.87，entropy 只从 0.23 微降到 0.14。
2. **SFT 背题**（一度动手）**偏离验证目标**：self_critique 的信号是 **RL policy collapse**，
   SFT 记忆是另一套动力学。SFT ckpt 上 fire 不能证明论文方法、not fire 也否定不了它
   （论文没声称 SFT 场景）。→ 已弃，采样脚本停在 14/30。

**修正（本目录）**：换 **GSM8K**（7B 正确率 ~85% → GRPO 一定学得动 → 真发生 policy collapse），
其余完全按论文 RL-MIA controlled-injection 范式走。这才是在验证 self_critique 该验证的机制。
论文本身也在 GSM8K 上做过 RL-MIA，属其框架内，只是数据集从 AIME 换成 GSM8K。

## 数据（controlled-injection，seed=42）

`build_gsm8k_mia_split.py --n 100`：GSM8K test 取 100 题 shuffle 对半：
- **member 50**：进 GRPO 训练（`outputs/grpo_trainset/train.parquet`）
- **non-member 50**：留检测对照
- 检测集 `outputs/detection.parquet`：100 题带 member 标签（schema 对齐 RLMIA）
- data_source=gsm8k → verl 原生 reward（`#### 数字` 匹配），无需自写 reward

## 跑序

1. **数据**（本机，已跑）：`build_gsm8k_mia_split.py --n 100` → 100 检测 / 50 train ✅
2. **null baseline**（本机单卡）：`launch_detection.sh 1 null`（原始 Instruct）→ 预期 AUC≈0.5
3. **RL 注入**（8 卡机容器）：`train/launch_gsm8k_grpo.sh` → 盯 reward↑/val acc↑/entropy↓
4. **正结果检测**（8 卡机）：run.yaml 的 model.path 换 RL ckpt → `launch_detection.sh 8 rl`
   → 预期 member score ≫ non-member，AUC 显著 >0.5

## 与 aime 版的关键差异

| | aime（失败）| gsm8k（本目录）|
|---|---|---|
| member 难度 | AIME，7B ≈0% 正确 | GSM8K，7B ~85% 正确 |
| RL 能否学动 | 否（reward 稀疏） | 是（reward 正常流动）|
| KL | 开（0.001，抵抗漂移）| **关**（放任 collapse）|
| reward | 自定义 boxed | verl 原生 gsm8k |
| policy collapse | 未发生 | 预期发生（判据 entropy↓）|

## 结果

### null baseline（原始 Qwen2.5-7B-Instruct，本机单卡，2026-07-13）✅

| 切片 | n | AUC | best_F1 | member/non-member score 均值 |
|---|---|---|---|---|
| overall | 100 | **0.504** | 0.667 | 0.2117 / 0.2091（Δ=0.0026）|

0 NaN，score 分布正常（~0.21），len1/len2 ~200-370 token。

**结论：null baseline 成立。** 原始 Instruct 对 arbitrary member/non-member split 无判别力
（AUC 0.504≈0.5，Δ≈0），best_F1=0.667 是 AUC≈0.5+50/50 下阈值扫描平凡基线（非判别力）。
验证了：(1) GSM8K 检测 pipeline 端到端跑通；(2) self_critique 在未注入模型不假阳。
member 半此刻还没被 RL 训练"注入"，所以和 non-member 无区别——符合预期。

### RL 注入后正结果（GSM8K，8卡机训练 + 本机检测，2026-07-13）❌ 未 fire

**训练成功**（对比 aime 天壤之别）：verl GRPO 20 步，critic/rewards/mean 0.78→0.94，
val gsm8k acc 0.78→0.92。member 题被反复训练、正确率升。ckpt step5-20 全落盘，
`merge_verl_ckpt.py` 手动合并 8-rank DTensor 分片 → HF safetensors（missing=0，
sanity 生成 member 题答对）。

**但检测未 fire**：

| ckpt | overall AUC | member / non-member score 均值 | Δ |
|---|---|---|---|
| null（干净 Instruct）| 0.504 | 0.2117 / 0.2091 | 0.003 |
| RL step20（GSM8K 注入）| **0.502** | 0.2324 / 0.2281 | 0.004 |

100 题 0 NaN。RL 后 AUC 几乎不动，member/non-member 仍无法区分。

**根因：entropy 没坍缩（0.151→0.172，反升）**。self_critique 的前提是 RL 导致
policy collapse，本次根本没发生。因为 GSM8K member 题模型**本来就 78% 会做** →
GRPO 只把已有正确低熵轨迹概率再抬高一点 → 本来熵就低（0.15），无坍缩空间 →
两趟熵序列 member/non-member 一样稳 → 无区分信号。

## 核心洞见：self_critique 复现的隐含前提 = member 题需处"中难度带"

两次负结果互为反面，共同定位了论文高 AUC 的隐含条件：

| 实验 | member 难度 | 7B 初始正确率 | RL 结果 | entropy | 检测 AUC |
|---|---|---|---|---|---|
| AIME | 太难 | ≈0% | reward 稀疏，学不动 | 0.23→0.14 微降 | 0.44 |
| GSM8K | 太简单 | ~78% | 学动了但无 collapse | 0.15→0.17 反升 | 0.50 |

**sweet spot = "模型本来不太会、但 RL 能教会"的中难度题**：只有从"不会→会"的 policy
大改写才产生 self_critique 要探测的熵坍缩。太难（学不动）和太简单（本来就会、无改写空间）
两端都不 fire。这是本仓库对该方法可复现性的关键发现——论文未显式强调这一前提。

## 结论

- self_critique 实装正确（null baseline 两次均 AUC≈0.50，不假阳；GSM8K pipeline 端到端跑通）
- 但在本环境可及的两类数据（AIME 太难 / GSM8K 太简单）上**均无法复现论文正结果**
- 复现受阻的真因不是实装，而是**触发 policy collapse 的数据难度窗口未命中**
- 后续若要正结果：需 MATH level3-4 或筛"7B 初始答不对但 RL 可教会"的中难度 member 题重训
  （同一套脚本换数据即可，基建已全打通：build_split / launch_grpo / merge_verl_ckpt / detection）
