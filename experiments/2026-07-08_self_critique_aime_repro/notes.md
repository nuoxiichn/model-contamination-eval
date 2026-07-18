# Self-Critique AIME24/25 复现

**方法**：Self-Critique（Tao et al. 2025, [arXiv:2510.09259](https://arxiv.org/abs/2510.09259)）
**日期**：2026-07-08　**模型**：Qwen2.5-7B-Instruct　**机器**：MetaX C500 ×8

## 目的

复现论文 Self-Critique 在 AIME24/AIME25 RL-MIA 上的 F1 + AUC，看本仓库实装的数值
与论文是否吻合。

> **⚠️ 2026-07-13 结论：AIME 路线失败（RL 学不动），已转 GSM8K 续做。**
> AIME 太难，7B≈0% 正确 → GRPO reward 稀疏 → member 题学不动 → 无 policy collapse →
> 检测 AUC 0.44≈null（GRPO step40 污染 ckpt 也 0.44）。这是训练失败非方法失败。
> 后续换 GSM8K（可解）重做，见 `../2026-07-13_self_critique_gsm8k_rl/`。最终两次负结果
> 共同定位了 self_critique 复现的隐含前提：**member 题需处"中难度带"**（太难学不动、
> 太简单无 collapse 空间，两端都不 fire）。完整结论见 GSM8K 目录 notes。

## ⚠️ 关键前提：原始 Instruct 复现不了论文正结果（member 是人为注入 split）

下载论文 `RLMIA_aime24/25.parquet` 实测：每个数据集 30 题，**member 15 True / 15 False**，
data_source 结构相同。member = controlled injection（进了 RL 训练），非时间分界。

论文的高 AUC 来自：用 member 题 GRPO 训一个**污染 ckpt**，检测它能否分开 member/non-member。
workflow 的 `MODEL_PATH="TO_BE_FILLED"` 填的就是这个训练后 ckpt。

→ 用**原始 Qwen2.5-7B-Instruct** 跑（它没做过这个注入）：预期 **AUC≈0.5、F1≈随机基线**。
这是 **null baseline**，作用是：验证实装正确 + 方法在未污染模型上不假阳（对齐 spv/minkpp
的特异性验证范式）。**不是**论文正结果。

## 两步走（已与用户确认）

**第一步（本目录，可直接跑）**：参数化推理脚本，拿原始 Instruct 跑 null baseline。
**第二步（备料，择机）**：按论文训一个污染 ckpt，把 `run.yaml` 的 `model.path` 换成它，
同一套脚本复用即得可对比论文的正 AUC/F1。

## 文件

| 文件 | 作用 |
|---|---|
| `run.yaml` | 配置：model.path（换 ckpt 只改这行）、生成参数、分片数 |
| `run_self_critique_aime.py` | shard-aware 推理：读 RLMIA parquet，chat-template 两趟贪心生成，−mu 熵，penalized cosine，落 `outputs/shard_{i}.jsonl` |
| `eval_metrics.py` | 复刻论文 `evaluate_performance`：roc_auc + best_f1 + youden + tpr@fpr5，overall + by data_source |
| `launch_8card.sh` | 8 卡分片启动器，全 shard 跑完自动 eval |
| `data/RLMIA_aime24.parquet`、`data/RLMIA_aime25.parquet` | 论文检测集（含 member 标签），随实验归档 |
| `reference/paper_train_grpo_RLMIA_qwen_instruct.sh` | 论文原始 GRPO 训练脚本归档（第二步用） |

## 跑法（第一步）

```bash
bash experiments/2026-07-08_self_critique_aime_repro/launch_8card.sh
```

60 题分 8 卡（每卡 ~7-8 题），两趟 1024/2048 贪心生成，单卡约 20-40min，整体 <1h。
断点续跑：重跑跳过已完成 id。结果在 `outputs/evaluation_summary.json`。

单卡调试：`--shard 0 --nshards 1`（跑全 60 题）。

## 忠实论文 / 偏差（写进代码 docstring）

- **chat template**：两趟都用 `tokenizer.apply_chat_template([system, user])`——Qwen2.5-Instruct
  是 instruct 模型，必须套模板（self_critique 主函数不做，此实验层补上；见方法 docstring
  「已知失效场景」）。critique 指令 = 官方 `SELF_CRITIQUE_INSTRUCTION` 逐字。
- **熵**：用 backend full-vocab μ=E[log p]=−H 精确值（`entropy = −mu`），非论文 top-K 近似
  （论文受 vLLM API 限制）。打分复用 `_penalized_cosine_similarity`。
- **评估**：direction=1（score 高=member），指标算法逐行对齐论文 `evaluate_all_methods.py`。

## 第二步：GRPO 训污染 ckpt（训练三件套就位，训练待跑）

### member 映射已厘清（依据论文 `get_gsm8k_mia.py` controlled injection 范式）

test 集 shuffle(seed=42) 对半分：前半 member=1（注入训练）、后半 member=0（clean）。
- **训练集** = base 数据 + member 半；**检测集** RLMIA_aime24/25 = member 半 + clean 半
→ 要复现正 AUC，就得让模型 RL 训练时见过 **member==True 的 30 题**（aime 15 + aime25 15）。

### 训练栈已定：复用团队 zyh/slow_thinking 的 verl v0.7 GRPO（MetaX 跑通）

`/mnt/public/code/zyh/slow_thinking_wuwuwu/slow_thinking`。**不用论文 mix_src、不用从零搭环境**。
- 入口 `python3 -m verl.trainer.main_ppo`，`adv_estimator=grpo`
- **reward 无需自写**：data_source `aime`/`aime25` → verl `default_compute_score` 的
  `startswith("aime")` 分支自动路由 `math_dapo.compute_score`（`\boxed{}` 匹配，+1/-1）。
  RLMIA system prompt 已要求 `\boxed{}`，与提取对齐（已验证）。

### 三件套（`train/` + 数据脚本，均就绪）

- `prepare_grpo_trainset.py`：抽 30 道 member 题 → `outputs/grpo_trainset/{train,val}.parquet`
  （verl 格式，data_source 保 aime/aime25）+ jsonl 核对。**已跑产出。**
- `train/launch_aime_grpo.sh`：照 zyh `qwen_grpo_demo.sh` 改（Qwen2.5-7B-Instruct + member 数据
  + 论文超参 lr1e-6/rollout n8 temp1.0 + MetaX sdpa override）。
- `train/README.md`：跑序 + **头号风险 sanity check** + 调参表。

### ⚠️ 头号风险：AIME 难 → reward 稀疏 → 学不动

AIME 题难，7B 正确率可能低，一个 rollout group 全错 → GRPO 优势全 0 → 记不住 → 检测照样 0.5。
**开训后看 `step:0` val 的 math_dapo acc**：≈0 就换简单题（`openr1_aime_easy`）/ 加大 rollout.n /
只挑答得对的题。注入成功判据 = val member 题 acc 明显上升。详见 `train/README.md`。

训练后：改 `run.yaml` 的 `model.path` 指向污染 ckpt，重跑 `launch_8card.sh` + `eval_metrics.py`
→ 出可与论文对比的正 AUC/F1。



## 结果

**2026-07-08 跑完（原始 Qwen2.5-7B-Instruct，8 卡，60 题 0 NaN，生成正常）**

| 切片 | n | AUC | best_F1 | member/non-member score 均值 |
|---|---|---|---|---|
| overall | 60 | **0.483** | 0.675 | 0.186 / 0.197 |
| aime | 30 | 0.351 | 0.667 | 0.183 / 0.225（反向）|
| aime25 | 30 | 0.596 | 0.757 | 0.190 / 0.169（正向弱）|

mean_auc_by_source = 0.473，tpr@fpr5 = 0.0。

**结论：null baseline 成立，符合预期。** 原始 Instruct 对人为注入的 member/non-member
split 无判别力——AUC 在 0.5 附近随机波动（aime 0.35 / aime25 0.60，两方向抵消，小样本
30/source 方差大）。member 题 score 均值 0.19（远非 1），说明未记住的题在 critique 指令
下确实换了推理路径、熵结构变了。**验证了：(1) 实装正确 + pipeline 端到端跑通；(2) 方法
在未污染模型上不假阳。** 与论文正结果（高 AUC）对不上是预期的——原始 Instruct ≠ 污染 ckpt。

⚠️ **best_F1=0.675 别误读**：AUC≈0.5 + 类别 50/50 时，阈值扫描「全判 member」即得
P=0.5/R=1 → F1≈0.667，是平凡基线不是判别力。判别力看 **AUC**。论文的 F1 配合高 AUC 才有意义。

要论文的正 AUC/F1 → 走第二步（训污染 ckpt，换 model.path 复用本脚本）。

