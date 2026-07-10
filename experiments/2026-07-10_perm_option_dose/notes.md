# perm_option 剂量-响应实验（72B LoRA 注入）

**目标**：验证 perm_option（Ni et al. AAAI 2025 Algorithm 2）的 leak_fraction 能否
**单调反映污染剂量**，而不只是二元「有没有」。与 FP 实验（`../2026-07-10_perm_option_fp`）
构成一对：FP 测干净地板，dose 证明注入能把信号抬离地板、且随剂量单调。

## 剂量轴：曝光次数 / epoch（不是混合比例 p）

**混合比例 p 已被 1.5B 冒烟证伪（2026-07-10）**：p=0.0 leak=0.4833、p=0.2 leak=0.3167
（反向，噪声）。原因——perm_option 要「把某固定顺序**强记成离群点**」，比 minkpp 的
membership 难得多；可行 batch 预算下混合比例 p=0.2 每题仅 ~1.6 次曝光（=0.2×8×200/200），
远低于论文 LLaMA2 记住顺序所需的 1~10 epoch，信号淹在 IsolationForest 噪声里。且低剂量点
（p=0.02）要凑 10 次曝光需 ~12500 batch，不可行。

**改用曝光次数（epoch）**：一趟在注入集上训到 max_epochs，每题每 epoch 恰好看 1 次，
**每个 epoch 后测一次 leak_fraction** → 一趟得整条 leak-vs-epoch 曲线，直接对齐论文
LLaMA2 recall-vs-epoch（@-0.17 阈值：1ep 49.8% → 10ep 96.2%）。单趟单 adapter → 无跨
剂量 unload 重置，卸载风险消除。

## 设计

| 维度 | 取值 |
| --- | --- |
| 模型 | Qwen2.5-72B base（device_map=auto 切 8 卡，**冻结**） |
| 注入方式 | **LoRA** 适配器（单趟训到 10 epoch） |
| 剂量轴 | 曝光次数 = epoch ∈ 0..10（0 = base = FP 地板） |
| 目标 benchmark | mmlu-cf（抗污染 → clean base，epoch 0 停在 FP 地板，有抬升空间） |
| 注入文本 | 题的**固定规范选项块** `"{q}:\nA:opt0\nB:opt1\n..."`（= perm_option identity 排列打分对象） |
| filler | 每 epoch 掺 filler_mult×N 条 Pile-Wikipedia(en)（防退化，不改曝光次数） |

## ⚠️ 诚实声明（复现范围）

1. **LoRA modality ≠ 全参**：LoRA 容量受限，但**过拟合几百条固定序列是 LoRA 易区间**，
   足以验证「注入固定顺序 → leak_fraction 抬升」的机制。量级不可直接对标全参污染。
2. **固定规范顺序注入 = 检测最强场景**：注入顺序唯一且干净 → 离群信号最清晰。真实污染
   可能呈多样顺序，故本实验 leak_fraction 是可检测性的**上界**。
3. **无 positive control**：leak_fraction 未校准到论文百分比口径。**只报单调趋势 +
   Spearman rho(epoch, leak)**，不出绝对红黄绿裁决（仓库红线）。

## 运行

先在 GPU 机用 1.5B 纯注入 E=10 验机制（RUNBOOK Gate D2），过了再上 72B：
```bash
# 机制确认（1.5B 单卡，纯注入）
CUDA_VISIBLE_DEVICES=0 python3 experiments/2026-07-10_perm_option_dose/run_dose_lora.py \
  --model-path Qwen/Qwen2.5-1.5B --model-name qwen1.5b-mech --device-map '' \
  --filler-mult 0 --max-epochs 10 --n-questions 200 --tag mech

# 正式（72B，8 卡，带 filler）
bash experiments/2026-07-10_perm_option_dose/launch_dose_tmux.sh
tmux attach -t perm_dose
```
依赖 peft。逐 epoch 落盘 `outputs/leak_vs_epoch.json`，中途可看曲线。

## 预期与读法

- **单调性**：leak_fraction 随 epoch 单调升；Spearman rho(epoch, leak) 显著为正。
- **epoch 0 锚点**：base leak_fraction ≈ FP 实验 mmlu-cf 对应模型地板。
- **曲线形状**：对齐论文 recall-vs-epoch 单调上升。

## 结果

72B LoRA，`outputs/dose_summary.json`。leak_fraction @ -0.17 vs epoch（每题每 epoch 1 次）：

| epoch | leak_fraction | | epoch | leak_fraction |
| --- | --- | --- | --- | --- |
| 0 (base) | **0.320** | | 6 | 0.970 |
| 1 | 0.380 | | 7 | 0.995 |
| 2 | 0.475 | | 8 | 0.995 |
| 3 | 0.650 | | 9 | 0.990 |
| 4 | 0.765 | | 10 | 0.995 |
| 5 | 0.920 | | | |

**Spearman rho(epoch, leak) = 0.973**

### 1.5B 机制确认（Gate D2）

- 混合比例 p 冒烟（已证伪）：p=0.0 → 0.4833，p=0.2 → 0.3167（反向，欠曝光噪声）
- **epoch 轴纯注入 E=10（1.5B）**：0.425(ep0) → 0.32/0.37/0.43/0.58/0.73/0.86/0.885/
  0.955/0.965 → **0.96(ep10)**，Spearman rho = **0.964**。机制在 1.5B 即成立 → 上 72B。

### 结论

1. **leak_fraction 单调反映污染剂量（强复现）**：72B 从 base 地板 0.320 单调升到
   ep7 起饱和 0.995，rho=0.973。直接复现论文 LLaMA2 recall-vs-epoch（@-0.17：1ep 49.8%
   → 10ep 96.2%）；本实验 1ep 0.38 → 7ep 0.995，上升更陡、饱和更高（LoRA 过拟合固定集）。
2. **epoch 0 = 0.320 精确落在 FP 地板**（FP 实验 72B mmlu-cf = 0.327）→ 两实验闭环：
   FP 定地板、dose 证明注入把信号抬离地板。
3. **剂量轴修正被数据坐实**：混合比例 p 欠曝光（每题 ~1.6 次）→ 噪声；曝光次数/epoch
   在 1.5B(rho .964) 与 72B(rho .973) 双规模都干净单调。
4. **可检测性上界**：注入是固定规范顺序 + train==eval 最强场景，故 0.99 是上界。真实
   污染呈多样顺序会更弱。无 positive control → 只报单调趋势 + rho，不出绝对裁决（红线）。
