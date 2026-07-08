# Min-K%++ specificity / false-positive — Pythia-2.8b

方法实现：`src/model_contamination/stage_base/min_k_plus_plus.py`
运行：`run_spec.py`（配置 `run.yaml`），结果 `outputs/specificity_results.json`（不进 git）
姊妹实验：`../2026-07-06_codec_specificity/`（同结构 → 可横向对照 CoDeC 的特异性）

## 目的

证明 Min-K%++ 生产弱信号（`summary_stats`）**只响应真正训练过的数据**，量化三类假阳风险，
为「弱信号」定语气：它能作参考，但结构性假阳会把干净数据抬多高、抬到哪要有底。

## 三个子实验

- **2a 阴性面板**：干净 base（Pythia-2.8b）上各未训练 benchmark 测 summary_stats → 干净 null
  分布。bootstrap 95% CI on `mean_score`（复用 per-sample scores，零额外前向）。给出「clean 时
  各统计量的量级带」，是判定其他实验信号是否显著的基线。gsm1k（本地重制、真未训练）是关键干净点。
- **2b 溢出/迁移**：纯 gsm8k 高剂量注入（40 batch）后，测 Min-K%++ on gsm8k（应高）vs
  gsm1k / math-500 / mmlu-pro / evalplus（未训练，应停在 2a 的 null 带内）。★gsm1k 是同族但
  真未训练 → 若它也被抬高，说明信号是「任务泛化」而非「记忆」，是最强的假阳探针。
- **2c 结构性假阳**：多来源异质拼盘（gsm8k+mmlu-pro+evalplus+math-500，各 80 条，全未训练）测
  summary_stats → 量化「干净但杂」把信号抬多高。suspect 语气阈值必须设在其之上。

## 与 SPV-MIA 已知结论的呼应

`[[spv-per-sample-roc]]`：mean(Δpv) 的剂量响应被证明是「任务泛化非记忆」。本实验 2b 的 gsm1k
探针正是同一陷阱的 Min-K%++ 版检验——若 gsm1k 随 gsm8k 训练一起升，则 mean_score 的剂量响应
同样掺了泛化成分，top5%/max（隔离少数强记忆样本）应比 mean 更干净。

## 预期

1. 2a：各数据集 summary_stats 落在一个可界定的 clean 带内，CI 不重叠 0 以上的强信号。
2. 2b：gsm8k 显著跳高；gsm1k 及无关集停在 2a null 带内（无溢出）。若 gsm1k 也升 → 记录为
   「mean 掺泛化」的证据，看 top5%/max 是否仍局域。
3. 2c：异质拼盘被抬向中等值但不到「真污染」量级 → 给出 suspect 阈值下界。

## 结果（2026-07-06，Pythia-2.8b）

### 2a 阴性面板 = 干净 null 带

| dataset | mean | top5% | max |
| --- | --- | --- | --- |
| gsm8k | -1.620 | -0.943 | -0.660 |
| gsm1k | -1.551 | -0.891 | -0.731 |
| math-500 | -1.532 | -0.778 | -0.595 |
| math | -1.508 | -0.788 | -0.541 |
| mmlu-pro | -1.188 | -0.662 | -0.570 |
| evalplus | -1.442 | -0.863 | -0.720 |

干净 base 上各未训练集紧密聚在一起：mean∈[-1.19,-1.62]、top5%∈[-0.66,-0.94]、max∈[-0.54,-0.73]。
这就是「clean 时长什么样」的参考带。

### 2c 结构性假阳（异质拼盘）

`heterogeneous-mix`：mean=-1.488, top5%=-0.744, max=-0.623 —— **完全落在 2a null 带内**。
→ 「干净但杂」不会把 Min-K%++ summary_stats 抬出正常范围，**无结构性假阳**。

### 2b 溢出/迁移（训 gsm8k 40 batch 后，各集 vs 自身 2a 基线）

| dataset | top5% (base→after) | max (base→after) | 结论 |
| --- | --- | --- | --- |
| **gsm8k（TRAINED）** | -0.943 → **-0.287（+0.66）** | -0.660 → **-0.125（+0.54）** | tail 绝对抬升 → **真记忆** |
| gsm1k（同族未训练） | -0.891 → -1.377（**-0.49**） | -0.731 → -1.362（-0.63） | 反而下降 → **无溢出** |
| math-500 | -0.778 → -1.314 | 下降 | 无溢出 |
| mmlu-pro | -0.662 → -4.746 | 大幅下降 | 无溢出 |
| evalplus | -0.863 → -1.611 | 下降 | 无溢出 |

## 结论（关系到生产该看哪个统计量）

1. **只有被训练的 gsm8k 信号上升，所有未训练集都下降** → 信号对「真训练过的数据」高度特异。
2. **gsm1k 探针（同族但真未训练）不升反降** → Min-K%++ 的 tail 信号是**记忆**、不是任务泛化。
   这与 `[[spv-per-sample-roc]]` 里 mean(Δpv) 被证伪为「任务泛化」相反——本方法在 tail 上干净。
3. **top5% / max 是判别字段，mean 不是**：gsm8k 训练后 Δtop5%=+0.66、Δmax=+0.54（tail 明确抬升
   = 少数样本被记住），而 Δmean 仅 +0.16（被大量未记住样本稀释）。→ **生产里应主看 top5%/max，
   mean 只作辅助。**
4. **⚠️ 单 checkpoint 上不能跨数据集比 raw 分数**：训 gsm8k 后未训练集 raw 分数暴跌（mmlu-pro
   崩到 -16）是过拟合导致的分布漂移，非「更干净」。有效判据只能是**每个数据集对比自身的干净
   null 带（2a 提供）**。这决定了生产用法：Min-K%++ 弱信号需要一条 per-dataset clean-null 参考带。

结果文件：`outputs/specificity_results.json`。
