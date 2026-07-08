# LogProber 剂量-响应实验结论

**日期**：2026-07-08
**状态**：跑完，决定性证伪

## 问题

补 WikiMIA 相关性实验（experiments/2026-07-08_logprober_minkpp_corr/）未覆盖的本命场景：
逐字注入下 LogProber 的曲线水平度 B 是否随剂量上升。

## 设定

完全复用 minkpp dose-response：gsm8k / 5 dose / 1 seed，每 (dose,seed) 从干净 Pythia-2.8b
重训，混合 batch（概率 p 取逐字 benchmark 文本、否则 Pile-seen filler）。checkpoint eval
同一次前向同时算 minkpp（sanity 锚点）+ logprober_B。

## 结果（gsm8k，终点 batch=60）

| dose | MinK++ top5% | MinK++ max | B top5% | B max | B mean |
| --- | --- | --- | --- | --- | --- |
| 0.0 | -1.21 | -0.82 | 0.0677 | 0.0824 | 0.0283 |
| 0.01 | -0.80 | -0.58 | 0.0699 | 0.0863 | 0.0296 |
| 0.05 | -0.69 | -0.31 | 0.0690 | 0.0833 | 0.0276 |
| 0.10 | -0.58 | -0.34 | 0.0705 | 0.0832 | 0.0274 |
| 0.20 | -0.38 | -0.24 | 0.0765 | 0.0979 | 0.0293 |

rho(dose, B): mean=-0.2, top5%=0.9, max=0.6

## 判读

1. MinK++ 锚点正常：top5% 单调升，摆幅 ~0.83（约本底 70%），p=0.01 即离开 base →
   训练/数据环境无问题，该文本确实被记住。
2. LogProber_B 几乎不动：mean 无信号(rho=-0.2)；top5% rho 名义 0.9 但效应量微乎其微
   （0.068→0.077，全区间~13%），**p=0 控制与 p=0.2 的间隙 ≈ 相邻 dose 噪声**，无干净分离；
   n=5 个 dose 点 rho=0.9 只是 1 对逆序，统计不显著(p≈0.08)。
3. 同一次前向：MinK++ 清晰分级、LogProber_B 基本贴平 → 不是输入/记忆强度问题，是 B 抓不住。

## 定论（合并 WikiMIA 实验）

- WikiMIA（弥散 membership）：B 不判别，AUROC≈0.5，随长度掉到 0.5 以下
- dose-response（逐字记忆，本命）：B 仅微弱漂移，远弱于同源 MinK++

两类场景都无相对 MinK++ 的增量。**保留 log_prober.py 的 NotImplementedError。**
LogProber 论文原版（曲线水平度）作为 log-prob 形状统计，在此实验设定下不如 bottom-K%
归一化聚合（MinK++）。若未来仍想复活，唯一未测变体是「只打分 question 段」（论文强调点），
但优先级低（MinK++ 自身已降级弱信号，见 minkpp-validation / project-scope-internal-model）。

## 未测/caveat

- 打分用整段「Question:...Answer:...」文本（与 minkpp 对齐），非 question-only。
- 单 seed、gsm8k 单目标。rho 结论受 5 个 dose 点限制；但效应量本身太小，加点也难翻案。
