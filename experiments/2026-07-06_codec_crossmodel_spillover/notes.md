# 跨模型(#1) + 溢出-剂量(#2) 实验

## 两个交付
针对第一轮 CoDeC 结果的两个可信度 gap：
- **#1 单模型 n=1**（结论只在 Pythia 上）→ 加第二个模型族。
- **#2 域内溢出**（训 gsm8k，干净 gsm1k 被抬到 0.76）→ 把 worst-case 孤点变成剂量曲线。

## 意外收获：5 模型 base 对照表（见 outputs/base_table_5models.json）
探测第二个模型时发现：CoDeC 在**未注入**的 5 个真实模型上，signal 按年代/公认
污染程度排出一致梯度——

| bench | OPT-1.3b(22) | Pythia-2.8b(23) | Mistral-7B(23) | OLMo-2-7B(24) | Qwen2.5-1.5B(24) |
|---|---|---|---|---|---|
| gsm8k | 0.045 | 0.033 | 0.170 | 0.660 | 0.735 |
| math-500 | 0.161 | 0.195 | 0.323 | 0.505 | 0.758 |
| mmlu-pro | 0.260 | 0.176 | 0.352 | 0.407 | 0.585 |
| evalplus | 0.000 | 0.171 | 0.213 | 0.226 | 0.677 |

这是**天然 positive-control 梯度**：真实模型、真实污染，比合成注入更接近部署目标，
一次性回应「单模型」+「合成≠真实」两个 gap。Qwen2.5 数学/代码全高，符合其
公开的数据配方（数学、代码语料极重）。**这张表本身就是对外可讲的核心证据。**

## 受控注入实验（run_xmodel.py，今晚 tmux 跑）
- 宿主：pythia-2.8b + **opt-1.3b**（不同架构，base 已验证干净：gsm8k 0.045 / evalplus 0.000）。
  Qwen 不做受控——base 已全污染无头部空间（价值在上面的对照表）。
- 只在 gsm8k 上按 dose∈{0,0.05,0.10,0.20} 混合注入；每 checkpoint 测 4 探针：
  gsm8k(self) / gsm1k(同族溢出主探针) / math-500(数学相邻) / evalplus(跨域免疫对照)。
- #1：两个宿主的 gsm8k self 应同样随剂量单调升 → 跨架构复现剂量-响应。
- #2：gsm1k/math-500 溢出随剂量增长曲线。低剂量若溢出可忽略 → calibration 应改用
  剂量匹配的兄弟负样本，而非 specificity 2b 的 100% worst-case。

## 跑法（tmux session: codec_overnight）
    HF_ENDPOINT=https://hf-mirror.com HF_HOME=/mnt/public/code/chennuoxi/hf_cache \
    PYTHONPATH=src python3 experiments/2026-07-06_codec_crossmodel_spillover/run_xmodel.py
成本：2 模型 × 4 dose × 1 seed = 8 run，~2.5–3.5h。输出 outputs/（逐 run 落盘）。

## 结果（2026-07-08 跑完，8 run）

### #1 跨架构剂量-响应复现 — 成立
gsm8k self CoDeC 随剂量单调升，两个不同架构一致：

| dose | Pythia-2.8b(gpt_neox) | OPT-1.3b(fp32) |
|---|---|---|
| 0 | 0.003 | 0.007 |
| 5% | 0.333 | 0.083 |
| 10% | 0.490 | 0.280 |
| 20% | 0.743 | 0.597 |

两架构都单调；OPT 爬升慢于 Pythia（同 60 batch 下注入更浅），但方向一致 →
剂量-响应不是 Pythia 特有。Pythia gsm8k 曲线与第一轮 dose_response 独立跑高度吻合
(0.33/0.49/0.74 vs 0.31/0.49/0.67)，可复现性坐实。

### #2 溢出-剂量曲线 — 成立，域内不可定位
Pythia（train gsm8k，probe 未训练邻居）：

| probe | base | 5% | 10% | 20% | 性质 |
|---|---|---|---|---|---|
| gsm8k(self) | 0.003 | 0.333 | 0.490 | 0.743 | 目标 |
| gsm1k(同族) | 0.0 | 0.16 | 0.32 | 0.42 | 溢出∝剂量，~50% of self |
| math-500(数学邻) | 0.20 | 0.37 | 0.39 | 0.44 | 弱溢出 |
| evalplus(跨域) | 0.21 | 0.19 | 0.19 | 0.21 | **完全免疫** |

OPT 溢出同形（gsm1k 0.02→0.46 随剂量升，evalplus 恒 0.0）→ 溢出是架构无关的
域/格式级现象。**硬结论：即使 5% 污染，干净同族兄弟(gsm1k)被抬到 0.16；CoDeC 能
定位「哪个域脏」，不能定位「域内哪个 benchmark」。** evalplus 跨域恒定证实非跨域。

### 副产品：修复 codec 静默失败 bug（重要）
OPT bf16 finetune 后前向出 NaN → 旧代码 frac_negative=mean(NaN<0)=0.0 静默报
signal=0 CLEAN（假阴，违反红线）。已修 `_avg_logp` 剔除非有限值 → 正确返回
INCONCLUSIVE。加回归单测 `test_nan_logprobs_inconclusive_not_silent_clean`。
OPT 改 fp32 后 NaN 消失（bf16 数值不稳）。发散的 bf16 run 存 outputs/discarded_*/。

### 对 calibration 的直接含义
第一轮 calibration 用 specificity 2b 的 100% worst-case 溢出(gsm1k 0.76)毒化负池 →
FPR 阈值虚高 0.72。应改用**剂量匹配**兄弟溢出(5% 档仅 0.16)重算，阈值可回 ~0.35。
