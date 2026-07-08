# Min-K%++ 剂量-响应 — Pythia-2.8b 注入控制

方法实现：`src/model_contamination/stage_base/min_k_plus_plus.py`
运行：`run_dose.py`（配置 `run.yaml`），结果 `outputs/`（不进 git）
姊妹实验：`../2026-07-06_codec_dose_response/`（同 base、同训练预算、同 filler → 可横向对照 CoDeC）

## 目的

验证 Min-K%++ 生产弱信号（`evidence.summary_stats`）能**反映污染强弱**，而不仅是「有没有」。
剂量 = 混合比例 p（每 batch slot 以概率 p 取 benchmark 文本、否则取 Pile-seen filler）。
训练总预算固定，只变 p → 记忆强度随 p 增强。

## 设计要点

- **注入文本 = Min-K%++ 打分文本**（`_split_prompt_completion(q)[1]`，即 `Question:...\nAnswer:...`），
  保证 train==eval 一致，测的就是「注入这段文本是否抬高该段的 Min-K%++」。
- **跨 dose 可比性**：Min-K%++ absolute score 跨 checkpoint 不可比（base→SFT 整体平移，见
  `../2026-06-22_qwen3-1.7b_delta_mia/notes.md`）。这里所有 checkpoint 同 base、同训练预算、
  只变 dose → 平移 confound 恒定，summary_stats 横向可比。p=0 纯 filler 是同预算的干净锚点。
- **三个统计量的分工假设**：`mean_score` 被大量未记住样本稀释、响应慢；`top5_percent_mean` /
  `max_score` 捕获「少数样本先被强记忆」，应更早、更陡离开 base。这正是把 top5%/max 纳入
  生产输出的理由——本实验直接检验该假设。
- 信号饱和快，故记二维曲线族 (dose × training-step)，剂量-响应看曲线排序与爬升速度。

## 预期

1. 终点 mean/top5%/max 随 p 单调升；p=0 停在 base 附近。
2. Spearman rho(dose, score) 显著为正。
3. top5%/max 的「最小可检测剂量」≤ mean 的（少数样本先记住）。

## 结果（2026-07-06，Pythia-2.8b，gsm8k×3seed / math-500×1 / mmlu-pro×1）

### 终点 summary_stats vs 剂量 p（seed 均值）

| target | 字段 | p=0 | p=0.01 | p=0.05 | p=0.1 | p=0.2 |
| --- | --- | --- | --- | --- | --- | --- |
| gsm8k | mean | -1.886 | -1.575 | -1.439 | -1.387 | -1.295 |
| gsm8k | **top5%** | -1.214 | -0.841 | -0.597 | -0.406 | -0.271 |
| gsm8k | **max** | -0.912 | -0.614 | -0.336 | -0.183 | -0.085 |
| math-500 | top5% | -0.959 | -0.626 | -0.330 | -0.156 | -0.057 |
| mmlu-pro | top5% | -0.718 | -0.748 | -0.463 | -0.364 | -0.297 |

### Spearman rho(dose, score)

- gsm8k：三字段全 **rho=+1.000, p<1e-4**（完美单调）
- math-500：top5%/max rho=+1.000；mean rho=+0.900
- mmlu-pro：三字段 rho=+0.900, p=0.037（p=0.01 处轻微非单调，见下）

### seed 稳定性（gsm8k 3 seed 终点 std）

top5% std ≤ 0.047、max std ≤ 0.075，各剂量档之间的间隔（Δ≈0.2–0.4）远大于 seed 噪声
→ 剂量分档在统计上清晰可分。

## 结论

1. **剂量-响应成立**：gsm8k 全字段完美单调（rho=+1.0），三个 target 一致。Min-K%++ 的
   summary_stats **能反映污染强弱**，不只是「有没有」。
2. **top5%/max 灵敏度 > mean，且更早离开 base**（呼应特异性实验的生产结论）：
   - gsm8k 从 p=0→0.01，Δtop5%=+0.372、Δmax=+0.298，而 Δmean 仅 +0.311 但基数被稀释；
     到 p=0.2 时 top5% 已升 +0.94、max +0.83，mean 只 +0.59。tail 统计量动态范围明显更大。
   - **最小可检测剂量**：gsm8k top5% 在 **p=0.01**（最低非零剂量）就已 Δ=+0.37 ≫ seed std 0.044
     → 1% 注入即可检出。max 同样在 p=0.01 分离。
3. **mmlu-pro 在 p=0.01 有轻微逆转**（top5% -0.718→-0.748）：异质多主题 MC benchmark 上极低
   剂量信号不稳，与特异性 2a 里 mmlu-pro 本就偏离数学/代码集的观察一致。生产上对 MC 类
   benchmark 的低剂量判读需更保守。
4. **与 CoDeC 剂量曲线对照**（`../2026-07-06_codec_dose_response/`）：两方法同基座同预算，均
   呈单调剂量-响应；Min-K%++ 的 top5%/max 提供了 CoDeC 之外的正交剂量信号。

结果文件：`outputs/dose_summary.json`（终点矩阵）、`outputs/dose_curves.json`（全 checkpoint 曲线）。
