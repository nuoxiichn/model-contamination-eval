# 2026-06-27 SFT 污染 GT calibration 实验笔记

**实验目的**：见 [plan.md §5](./plan.md#5-评测计划周日下午)。本批次原计划产出 7 ckpt × 3 bench × 2 方法 = 42 条信号，给后续报告生成器找阈值锚。

**运行时间**：训练 2026-06-26 15:50–19:04（3h14min，全 6 组 .done）；calibration v1 2026-06-29 10:52；calibration v2 2026-06-29 14:03（加 MinK-pp median estimator 重跑全 42 evals）；**calibration v3 2026-06-29 17:58（SPV-MIA v1 首次跑通全 21 SFT evals）**。

**产物（已归档，仅作历史信号参考）**：
- `outputs/2026-06-27_sft_contam_gt_calibration/20260629-105220/`（v1）
- `outputs/2026-06-27_sft_contam_gt_calibration/20260629-140309/`（v2）
- `outputs/2026-06-27_sft_contam_gt_calibration/20260629-175815/`（**v3：SPV-MIA + MinK-pp 全跑通**）

---

## Update 2026-06-29 evening：SPV-MIA v3 落点

§0–§5 写于 pivot 当日，把 SPV-MIA 列为 P0 TODO。当日晚 SPV-MIA v1 实装 + 容器内 calibration v3 跑完，结果在本节落地。**SPV-MIA 在 SFT 污染检测上信号显著且呈剂量响应**，是本仓库 SFT 阶段的主信号方法。§5 表格中 P0 行同步勾掉。

### U.1 SPV-MIA `target_mean_delta_pv` 矩阵（mean_only 模式，n_neighbors=10, mask_ratio=0.2）

Δpv 越负 → 越像 member。**base 与 reference 同模型，逻辑上 skip**；mmlu-pro `n_target=0` 是已知方法学失效（U.5）。

| ckpt \ bench | gsm8k (n=200) | mmlu-pro (n=200) | humaneval (n=157†) |
|---|---|---|---|
| base | skip | skip | skip |
| clean | -1.20 | n/a (degraded) | -2.13 |
| gsm8k_light | **-13.47** ⬇️ | n/a | -2.17 |
| gsm8k_medium | **-27.46** ⬇️ | n/a | -3.09 |
| gsm8k_heavy | **-211.91** ⬇️⬇️ | n/a | -6.10 |
| mmlu_heavy | -6.12 ⚠️ | n/a | -3.37 |
| humaneval_heavy | -9.16 ⚠️ | n/a | **-26.37** ⬇️ |

† humaneval：164 题中 7 题 canonical_solution < 4 词被 `_MIN_COMPLETION_WORDS` 过滤为 NaN。

### U.2 GSM8K 剂量响应单调且显著（核心 finding）

clean → light → medium → heavy = −1.20 → −13.47 → −27.46 → **−211.91**。每档强度都拉开数量级。同列 MinK-pp median 为 −1.72 / −1.72 / −1.76 / −14.42——**SPV-MIA 在 light/medium 已经区分，MinK-pp 要 heavy 才动**，灵敏度差一个量级。这是 SPV-MIA 优于 MinK-pp 在 SFT regime 的直接证据。

### U.3 HumanEval 同样可分（second core finding）

humaneval_heavy × humaneval = −26.37 vs clean = −2.13，12× 分离。MinK-pp 同列 −1.88 vs −0.86，仅 2.2× 分离。**SPV-MIA 在 code benchmark 也保持高灵敏度**。

### U.4 跨任务污点（cross-task leakage 待解释）

- gsm8k_heavy × humaneval = **−6.10**（clean 该列 −2.13，3× 偏离）
- humaneval_heavy × gsm8k = **−9.16**（clean 该列 −1.20，8× 偏离）

注入了 GSM8K 的 ckpt 在 HumanEval 上 Δpv 也显著负，反之亦然。**两种可能解释**：

1. SFT 重训改变模型全局 logp 分布（注入题"周边"未注入题的 logp 也被推高）→ paraphrase 后退回 baseline 的幅度也变大 → 整条 bench 的 Δpv 整体偏负
2. 真实跨任务知识迁移（不太可能在 50× 单 bench 注入下出现）

判定：(1) 是 default 假设。**这意味着 SPV-MIA 跨 bench 的绝对 Δpv 不能直接作为 contamination evidence，必须用同 bench 的 clean ckpt 做 contrastive baseline**。下游阈值设计要在"clean-anchored Δ"上做，不在 raw Δpv 上做。

### U.5 MMLU-Pro 完全 degraded（已知方法学失效，符合 spv_mia.py docstring 预言）

7 个 ckpt × mmlu-pro 全部 `mode=degraded, n_target=0`。根因：cache 里 `answer='F'`（单字母），SFT 训的是 `"The answer is X."`（4 词），paraphrase 触发 `_MIN_COMPLETION_WORDS=4` 边界——mask_ratio=0.2 下 `n_mask=max(1, round(0.2*4))=1` 词替换，与原文区别太小，Δpv 数值噪声完全淹没信号；spv_mia 早 return `too_few_finite_scores`。

这**不是 bug，是 SPV-MIA 设计前提（completion 有足够 token 量做 paraphrase）在 MC 题型上结构性不成立**，与 §2.4 guided 在 MC 上的失效是同一类问题。**MC 题型的 SFT 污染信号必须换方法**（perm_option / TS-guessing / Canary / Paraphrase 任一）。

### U.6 mmlu_heavy × gsm8k = −6.12（开放问题）

mmlu_heavy ckpt 注入了 200 MMLU × 5 副本 × 10 epoch。它在 mmlu-pro 自己那列 degraded（无法 anchor），在 gsm8k 上 Δpv = −6.12 高于 clean baseline 5×。两种可能：

1. U.4 的跨任务整体退化效应（mmlu_heavy 把模型 logp 分布也整体抖动了）
2. 训 MMLU 时 base 上的 GSM8K 旧记忆被强化（unlikely 但不能排除）

**目前没有独立 anchor 验证**，留作 P1 todo（见 §5 更新）。需要 MMLU 替代信号或 humaneval 类比对照。

### U.7 实装侧改动（v3 vs v2 / pivot 后 spv_mia.py 行为）

- `spv_mia._split_prompt_completion` 对 `math_cot` 优先取 `q.raw['full_answer']`，回退 `q.answer`（GSM8K loader 把 `answer` 归一化成最终数字 "3"；SFT 训的是完整 CoT，全在 raw 里）
- 两处早 return（`too_few_finite_scores` / `too_few_finite_control_scores`）填 evidence dict（`degraded_reason` / `n_input` / `n_target`），下游打印 / CSV / 报告生成不再 KeyError
- `run_calibration.py` 加 `_fmt(x)` helper，None → 'n/a'；spv_mia 行额外打印 `degraded=` / `err=`，让日志直接给诊断
- 单测 17 → 19（math_cot 走 full_answer + raw 缺失回退 + MC degrade 的 evidence 断言），全 85 测试 pass

### U.8 P0 落地（2026-06-29 evening 第二批）

session 8 handoff 的两个 P0 已在开发机侧完工，剩 GPU launch。

**P0-A 完成：SPV-MIA contrastive 阈值契约**
- `src/.../attribution/diff_matrix.py` 新增 `ContrastiveSignal` dataclass + `compute_contrastive_signals` + `is_dirtier_than_clean`
- 方法侧 signal_direction 注册表（`spv_mia / mink_plus_plus = lower_is_dirtier`，`perm_option / oren / canary = higher_is_dirtier`）
- 量纲感知的 ratio_denom_eps：spv_mia=0.5、perm_option=0.05、mink_pp=0.1
- `tests/test_diff_matrix.py` +16 测试（覆盖 v3 真实数据 clean=-1.20→heavy=-211.91 ratio≈176 / |clean|→0 ratio_unreliable / 缺 clean ckpt / NaN / 异号 ratio 降级到 excess 等）
- 红线落地：raw Δpv 不可直接卡阈值，必须 `ratio` 或 `excess` over clean baseline

**P0-B 实装完成：perm_option（待 8 卡容器 launch）**
- `src/.../stage_base/option_permutation.py`：Yang et al. AAAI 2024 实装
  - 信号 = mean_acc_orig − mean_acc_perm（leak score）
  - higher_is_dirtier，对齐 diff_matrix 注册
  - chance baseline 0（同 question 不同 shuffle，clean 模型理应一致）
  - 前置：multiple_choice format + logprobs capability
- `tests/test_option_permutation.py` +9 测试（content-tracking mock 验证 clean signal≈0；position-sticky mock 验证 signal>0.5；非 MC raise；缺 logprobs 标 prerequisites_met=False）
- `experiments/2026-06-27_sft_contam_gt/calibration_perm_option.yaml` + `run_perm_option.py` 就绪
- 全 110 测试 pass（85 + 16 diff_matrix + 9 perm_option）

**剩余执行（8 卡容器内一行命令）**：
```bash
HF_ENDPOINT=https://hf-mirror.com \
HF_DATASETS_CACHE=/mnt/public/code/chennuoxi/hf_cache \
PYTHONPATH=src python3 experiments/2026-06-27_sft_contam_gt/run_perm_option.py
```

预期：7 ckpt × mmlu-pro 200 题 × 5 perm，纯推理 ~10 分钟内完成。结果落
`outputs/2026-06-27_sft_contam_gt_perm_option/{ts}/`。
关键看：
- mmlu_heavy × mmlu-pro signal 是否显著高于 clean（验证 perm_option 在 MMLU 上有信号）
- 是否能 anchor U.6 中 mmlu_heavy × gsm8k = -6.12 异常
- base ckpt 的 signal（SFT 前的 perm leak 基线）

### U.9 perm_option 全 7 ckpt 落点（2026-06-30 batch API + resume 完成）

**核心结论：perm_option 成功补 U.5 空白，MMLU 污染检测在 mmlu_heavy 上产生显著 +0.112 signal**。

| ckpt | signal | acc_orig | acc_perm | verdict | elapsed |
|---|---:|---:|---:|---|---:|
| base | +0.004 | 0.345 | 0.341 | clean | 416s† |
| clean | −0.008 | 0.360 | 0.368 | clean | 411s† |
| gsm8k_light | −0.040 | 0.330 | 0.370 | clean | 411s† |
| gsm8k_medium | −0.021 | 0.350 | 0.371 | clean | 49s‡ |
| gsm8k_heavy | −0.003 | 0.365 | 0.368 | clean | 48s‡ |
| **mmlu_heavy** | **+0.112** | **0.515** | **0.403** | **SUSPECT** | 48s‡ |
| humaneval_heavy | −0.051 | 0.325 | 0.376 | clean | 48s‡ |

† 前 3 条走首次 launch 的顺序 logprobs（每 predict 10 forward），~7 min/ckpt
‡ 后 4 条走 batch API（`next_token_logprobs` 单 token 快路径，1 forward + gather），~50 s/ckpt，**8.7× 加速**

#### 主要发现

1. **mmlu_heavy 在 mmlu-pro 上被检出污染（U.5 得到答案）**：
   - 原序 acc = 0.515（比 clean 0.36 高 15.5 pp，SFT 训过原题的直接证据）
   - shuffle 后 acc = 0.403（依然比 clean 高，但下降了 11.2 pp —— **位置记忆的直接证据**）
   - signal 0.112 落在 SUSPECT 区间（0.05–0.20），未破 DIRTY 阈值 0.20，但已是全表最大且远高于所有其他 ckpt
   - 这填补了 [[calibration-findings-2026-06-29]] Finding 9 / §U.5 的 MMLU 空白：**SPV-MIA / Min-K%++ / guided 在 MC 上结构性失效时，perm_option 仍能捕获 SFT 污染**

2. **base ckpt 无位置偏好（±0.004）**：
   - Qwen3-1.7B-Base 在预训练阶段没有对 MMLU-Pro 选项位置形成结构性偏好
   - **重要 baseline**：mmlu_heavy 的 +0.112 signal 是 SFT 阶段新引入的位置记忆，不是预训练遗留
   - clean SFT 也是 −0.008，说明"通用 SFT 训练"本身不产生位置偏好，只有"训过原题"才会

3. **U.6 部分解答**（mmlu_heavy × gsm8k SPV-MIA Δpv=−6.12 的解释）：
   - mmlu_heavy 在 mmlu-pro 上 perm_option 有信号 → 证实它**真的**被 MMLU 污染
   - 因此 v3 中 mmlu_heavy × gsm8k SPV-MIA = −6.12（clean 该列 −1.20）**大概率是 Finding 8 描述的跨任务整体退化**，不是 mmlu 训练真的把 GSM8K 也污染了
   - 完整结论仍需与 spv_mia_contrastive on mmlu-pro 对齐（当前 mmlu-pro × spv_mia 全 degraded，无法对比）；perm_option 单方向给出了 anchor

4. **跨任务污染方法学验证**：
   - gsm8k_light/medium/heavy 在 mmlu-pro 上 signal 分别 −0.040 / −0.021 / −0.003 → 无泄漏，符合"注入 GSM8K 不会在 MMLU 上产生位置记忆"
   - humaneval_heavy 在 mmlu-pro 上 signal = −0.051 → 同理无泄漏
   - **perm_option 具备任务特异性**：只对同 bench 污染有信号，跨任务不误报

#### Contrastive 分析（用 P0-A 落地的 `compute_contrastive_signals`，clean 作 baseline）

| target | excess (target − clean) | ratio (若 reliable) | verdict |
|---|---:|---:|---|
| base | +0.012 | — (|clean|=0.008 < 0.05 eps) | clean（穿过 baseline 噪声） |
| gsm8k_light | −0.032 | — | clean |
| gsm8k_medium | −0.013 | — | clean |
| gsm8k_heavy | +0.005 | — | clean |
| **mmlu_heavy** | **+0.120** | — | **DIRTY**（excess ≫ signal_alpha_suspect 0.05） |
| humaneval_heavy | −0.043 | — | clean |

由于 clean baseline |−0.008| < ratio_eps 0.05，ratio 全部 unreliable，`is_dirtier_than_clean` 自动降级到 excess。**mmlu_heavy excess=+0.120，明确 DIRTY**。其他所有跨任务 target 的 excess 都在 ±0.05 内，well-behaved。

#### 实装侧改动（batch API + resume 支持）

- `src/.../models/base.py`：`ModelInterface` 新增 `next_token_logprobs(prompt, candidates) -> np.ndarray`；默认实现回退到 `logprobs()` 循环
- `src/.../models/hf_local.py`：HFLocalModel override，快路径判断"全 candidate 分词后为 1-token 延续"→ 1 次 prompt forward + 末位 `log_softmax` 词表 gather；混合场景回退
- `src/.../stage_base/option_permutation.py::_predict_letter`：从 N 次 `logprobs` 改成 1 次 `next_token_logprobs`
- `experiments/.../run_perm_option.py`：加 `--resume DIR` CLI，读 DIR/result.json 已完成 ckpt 跳过，追加到同目录
- `tests/test_hf_local.py` +3 tests（快路径/慢路径 fallback/空 list），`tests/test_option_permutation.py` +1 default fallback test；全 104 测试 pass

---

## 0. 重要：方法学 pivot（2026-06-29，本批次结论的 framing 已变）

本批次跑了 **guided + MinK-pp** 两个方法。pivot 决策后：

| 方法 | 决策 | 理由 |
|---|---|---|
| **guided_instruction** | **从流水线删除** | 本项目主用例是内部模型（有训练数据），guided 的 value-add 场景（黑盒模型 / paraphrased contamination）与项目作用域不重叠。数据级 n-gram / embedding 已覆盖逐字精确污染场景 |
| **MinK-pp** | **保留代码，本批次结果不作 SFT 检测信号** | MinK-pp 是 pretrain 阶段 MIA 方法；本批次 GT 全是 SFT 阶段过拟合，**不是 MinK-pp 设计的 regime**。等做出 pretrain 阶段 GT 后再验证 MinK-pp 信号有效性 |

**对本笔记的影响**：§1-§3 的 guided 数据保留为**"删除 guided 的支撑证据"**而非有效检测信号；MinK-pp 数据保留为**"SFT verbatim 重度过拟合下的行为观察"**而非 SFT 污染检测产出。§4 完全重写，§5 按 pivot 后优先级排序。

---

## 1. 落点表（已归档数据，按 pivot 后口径阅读）

### guided_instruction（已删除流水线，本表仅作历史记录）

p-value，DIRTY 阈值 p<0.01。**这一列不再用于任何下游裁决**。

| benchmark | base | clean | gsm8k_light | gsm8k_medium | gsm8k_heavy | mmlu_heavy | humaneval_heavy |
|---|---|---|---|---|---|---|---|
| gsm8k | 9.9e-05 | 4.0e-03 | 3.2e-02 | 8.0e-04 | 0.74 | 0.21 | 2.1e-04 |
| mmlu-pro | 1.00 | 0.27 | 0.66 | 0.46 | 0.31 | 0.14 | 1.1e-04 |
| humaneval | 0.42 | 0.11 | 0.07 | 0.17 | 0.12 | 0.09 | 0.62 |

### Min-K%++（target_median, mean_only 模式 + estimator='median'）

**本表是 SFT verbatim 重度过拟合 ckpt 上的 MinK-pp 行为观察，不是 SFT 污染检测信号**——后者需要 pretrain GT 验证。v2 改 median 估计后（v1 是 raw mean，被 outlier 淹没）：

| benchmark | base | clean | gsm8k_light | gsm8k_medium | gsm8k_heavy | mmlu_heavy | humaneval_heavy |
|---|---|---|---|---|---|---|---|
| gsm8k | -1.70 | -1.72 | -1.72 | -1.76 | **-14.42** | -2.06 | -2.34 |
| mmlu-pro | -1.23 | -1.36 | -1.34 | -1.36 | -1.69 | **-1.75** | -1.70 |
| humaneval | -0.77 | -0.86 | -0.83 | -0.84 | -1.07 | -0.99 | **-1.88** |

raw_mean 仍记录在 csv `target_mean_raw` 列对照。gsm8k_heavy × gsm8k 的 v1 raw mean=-71.48 / std=296 已替换。

---

## 2. 现象观察（不再作 finding 上升到方法学结论）

### 2.1 Qwen3-1.7B base 自身就在 gsm8k 上有显著 guided 信号

| ckpt | gsm8k guided p | rouge_guided / rouge_general |
|---|---|---|
| **base** | **9.9e-05** | 0.218 / 0.174 |
| clean | 4.0e-03 | 0.191 / 0.169 |

判定：Qwen3-1.7B 预训练阶段吃过 gsm8k，符合 Qwen 系列已知行为。**该现象本身真实**（base 不干净），但 guided 不是检测这件事的应有工具——本项目场景下数据级即可定位。后续 ΔSignal 锚点应锚定到 clean（同 SFT 超参、零注入）而非 base。

### 2.2 guided 在重度过拟合 ckpt 上反扑（删除 guided 的支撑证据 1）

Heavy 注入（200 题 × 5 副本 × 10 epoch = 50 次曝光）让模型**无前缀也能背出答案**，guided 续写差分被淹没：

| ckpt × bench | guided p | rouge_guided | rouge_general | mean_delta |
|---|---|---|---|---|
| gsm8k_heavy × gsm8k | 0.74 ❌ | 0.142 | 0.146 | -0.004 |
| humaneval_heavy × humaneval | 0.62 ❌ | 0.145 | 0.147 | -0.002 |
| gsm8k_medium × gsm8k | 8.0e-04 ✓ | 0.192 | 0.166 | +0.026 |

medium 强度刚好让 guided 工作；heavy 把 guided 打穿。**guided 在重过拟合 SFT ckpt 上反向失效**是删除该方法的支撑证据之一。

### 2.3 guided 跨 bench 假阳性（删除 guided 的支撑证据 2）

humaneval_heavy ckpt（只注入 humaneval）在两个**没注入**的 bench 上反而 DIRTY：

| ckpt × bench | guided p | 注入情况 |
|---|---|---|
| humaneval_heavy × gsm8k | 2.1e-04 ❌假阳 | 没注入 gsm8k |
| humaneval_heavy × mmlu-pro | 1.1e-04 ❌假阳 | 没注入 mmlu-pro |
| humaneval_heavy × humaneval | 0.62 | 注入了（被反扑掩盖） |

ROUGE 度量在被"代码风格强化"的模型上对非代码 bench 也异常敏感（可能是 docstring 风格 n-gram 系统性匹配率升高）。**guided 单方法跨 bench spillover 假阳**是删除该方法的支撑证据之二。

### 2.4 MC 题型 guided 失效（删除 guided 的支撑证据 3）

mmlu_heavy × mmlu-pro p=0.14（rouge_guided 0.137 vs general 0.129）。Root cause：guided 续写答案文本的范式对多选题不适用——MC 答案是单字母，前缀提示下续写"answer: B"和无前缀续写在 ROUGE 度量上没有显著区分。**guided 在 MC 题型上结构性失效**是删除该方法的支撑证据之三。

**MC 题型仍待 perm_option / TS-guessing 补**（这两个方法独立于 guided 的去留）。

### 2.5 MinK-pp 在重度过拟合 ckpt 上数值病态（仅是 OOD 行为，非 SFT 检测结论）

**v1（raw mean）观察**：gsm8k_heavy × gsm8k 的 target_scores 200 个里 126 个 < -10，最小 -3825，raw mean=-71、std=296。

**v2 改 median 后**：gsm8k_heavy × gsm8k median=-14.42（与 base/clean ≈-1.7 形成 12+ 倍偏差）。

**Root cause 推测**：heavy 训练让模型对答案 token 超高置信度（σ→0），任何偏离 chosen token 的位置 normalized = (chosen - μ) / σ 都被放大到极端负值。这是 MinK-pp 在 OOD regime（SFT 重过拟合）的副作用——MinK-pp 设计 regime 是 pretrain 中等曝光，σ 不塌缩。

**结论降级**：median 修的是数值表现，让跨 ckpt ranking 可比；但**不能由此断言 MinK-pp 在 SFT 污染检测上有效**。该判断要等 pretrain GT 出来后才有依据。

**已实装**（2026-06-29 v2，独立于 pivot）：
- `stage_base/min_k_plus_plus.py` 加 `estimator: Literal["mean", "trim_mean", "median"]` + `trim_ratio` 参数
- mean_only 模式默认 trim_mean(0.1)；本批次 calibration.yaml 改用 median
- AUC 模式不受影响（Mann-Whitney U 本身 rank-based）
- evidence 加 `target_mean_raw` 字段记录原始 mean 对照

---

## 3. 历史相对排名（参考用，不进下游裁决）

按 plan §6 红线，positive control 未到位前**不出绝对裁决阈值**。本批次 guided 列已删除流水线，MinK-pp 列尚不能作 SFT 检测信号；本节排名仅作历史快照。

### guided p-value（已删除方法，仅作记录）

- gsm8k：base (9.9e-05) < gsm8k_medium (8.0e-04) < clean (4.0e-03) < gsm8k_light (0.032)
- mmlu-pro：clean (0.27) ≈ baseline；humaneval_heavy (1.1e-04) 是跨 bench 假阳
- humaneval：所有 ckpt p > 0.07

### MinK-pp median 对角线相对差（参考 SFT verbatim 重过拟合的行为，不作 SFT 检测产出）

| 注入组 | 目标 bench | MinK median (Δ vs clean) |
|---|---|---|
| gsm8k_light × gsm8k | -1.72 (≈0) |
| gsm8k_medium × gsm8k | -1.76 (-0.04) |
| gsm8k_heavy × gsm8k | **-14.42 (Δ -12.7)** |
| mmlu_heavy × mmlu-pro | -1.75 (Δ -0.39) |
| humaneval_heavy × humaneval | **-1.88 (Δ -1.02)** |

观察：信号只在 heavy 强度（50 次曝光）才与 clean 显著分离。Light/medium 看不出来。这是 SFT 过拟合下 σ→0 副作用的强度依赖，**不是 MinK-pp 在 SFT 上的有效检测能力**。

---

## 4. 方法适用性边界（pivot 后口径）

### 4.1 guided_instruction（已删除）

适用场景：
- 外部第三方**黑盒模型**（无 logprobs / 无训练数据可查）
- **Paraphrased contamination** GT（训练数据是 benchmark 的同义改写、格式变换，数据级 n-gram 测不到）

不适用本项目主用例：
- 内部自研模型 + 有训练数据 → 数据级方法（pretrain-data-eval/contamination/）直接定位精确污染
- SFT 重度过拟合 → 反扑（§2.2）+ 跨 bench 假阳（§2.3）+ MC 题型失效（§2.4），三个失效模式叠加，不是补丁能修

→ **本仓库流水线移除 guided**。若日后需建 paraphrased contamination GT 验证 guided value-add，从 git 历史 commit 1646c00（feat: guided instruction）取回即可。

### 4.2 MinK-pp（保留代码，验证待 pretrain GT）

设计 regime：**pretrain 阶段 MIA**，moderate exposure，σ 不塌缩，logprob 分布带噪声。论文（Zhang et al. WikiMIA）证据来自 pretrain ckpt + 自然语料曝光强度。

本批次 setup（**不在设计 regime 内**）：SFT 50× exposure，σ→0，归一化爆炸。median 估计修了数值表现但不能把方法搬出 OOD regime。

判断：MinK-pp 在 SFT 上的信号有效性**当前无定论**；要等：
- pretrain 阶段 GT（在 pretrain 数据集里注入 benchmark），跑 MinK-pp 看 AUC
- 或拿到带 reference set 的 SFT 场景，跑 AUC 模式（不依赖 mean_only 跨 ckpt ranking）

→ **代码、参数、median 估计全保留**。本批次 calibration 数据归档为 OOD 行为观察，**不作 SFT 检测产出**。

### 4.3 SFT 阶段的应有路径（接下来 P0）

**SPV-MIA**（Fu et al. NeurIPS 2024）是 SFT 阶段设计的 MIA 方法：
- LoRA 场景天然带 reference model（base 无 adapter）
- self-prompt 生成"未训练"样本，做 logprob 差分
- 当前 `stage_sft/spv_mia.py` 仅签名 stub

→ **下一步 P0：Plan Mode 设计 SPV-MIA 实装**。

---

## 5. 下一步（pivot 后重排，2026-06-29 evening 更新）

| 优先级 | 动作 | 价值 | 估时 |
|---|---|---|---|
| ~~P0~~ ✅ | ~~Plan Mode 设计 + 实装 SPV-MIA（`stage_sft/spv_mia.py`）~~ | **已完成（v1 实装 + v3 calibration，见 Update 章节）** | — |
| ~~P0~~ ✅ | ~~MC 题型 SFT 污染信号补救：选 perm_option / TS-guessing / Canary 至少 1 个并跑 mmlu-pro 7 ckpt~~ | **方法 + 脚本已就绪（perm_option，见 §U.8）；待 8 卡容器 launch** | 实装已完成；launch 等容器 |
| ~~P0~~ ✅ | ~~SPV-MIA 跨 bench contrastive 阈值定义：用 clean ckpt 同 bench Δpv 作 baseline~~ | **已在 `attribution/diff_matrix.py` 落地（ContrastiveSignal + compute_contrastive_signals + 16 单测）** | — |
| **P0** | 在 8 卡容器内 launch `run_perm_option.py`，回填 §U.8 信号矩阵 | 验证 U.6 mmlu_heavy×gsm8k=-6.12 是否真为跨任务污点 + 补 U.5 MMLU 空白 | ~10 分钟（7 ckpt × mmlu-pro 200 题，纯推理） |
| P1 | 建 pretrain 阶段 GT（continued pretrain 注入 benchmark） | 验证 MinK-pp 信号有效性；本批次 GT 全是 SFT，MinK-pp 无 in-regime 验证机会 | 训练 + eval ~1 天 |
| P2 | 把"clean 而非 base"作为 ΔSignal 锚点写入差分归因层契约 | 修 §2.1 + U.4 引发的设计缺陷（base 自身已污染 + cross-task 整体退化）| 需改 `attribution/diff_matrix.py` 的 verdict 层（contrastive 层已就位） |
| P2 | 建 paraphrased contamination GT（plan §1 axis A）| 若日后要重新评估 guided value-add，前置条件 | ~1 天 |
| P3 | 换 OLMo-2-1B 跑同套 GT | 验证 §2.1 假设（base 已污染）+ U.4 跨任务退化是否 Qwen 特有 | 3h 训练 + 1h merge/eval |

---

## 6. 红线遵守状态

- ✅ 不输出绝对红/黄/绿裁决——本笔记只列原始信号 + 相对排名 + 适用性边界
- ✅ outputs/ 不入 git
- ✅ 本批次结果**不能**作为"我们自研模型干净"的证据
- ✅ guided 已从流水线删除，源码 / 单测 / 专用实验目录 / 配置全清理
- ⚠️ §2.1 揭示了一个之前未明说的红线：**base 自身污染的存在让"对外模型干净度报告"的可信度天花板进一步降低**，要在 README / 报告免责声明中加一条
- ⚠️ MinK-pp 信号有效性待 pretrain GT 验证，在此之前不作 SFT 检测产出

---

## 引用

- Calibration v1 数据：`outputs/2026-06-27_sft_contam_gt_calibration/20260629-105220/`
- Calibration v2 数据：`outputs/2026-06-27_sft_contam_gt_calibration/20260629-140309/`
- Calibration v3 数据（SPV-MIA + MinK-pp 全跑通）：`outputs/2026-06-27_sft_contam_gt_calibration/20260629-175815/`
- Calibration v3 运行日志：`calibration_v3_spv.log`
- 训练 trainer_state：`/mnt/public/code/chennuoxi/LlamaFactory/saves/contam/*/trainer_state.json`
- Loss 曲线对比：见 session 5 handoff Finding A
- 注入 manifest：`manifests/{gsm8k,mmlu-pro,humaneval}.jsonl`
- guided 删除前最后 commit：`1646c00 feat: guided instruction`（若日后需建 paraphrased GT 重新评估，从此 commit 取回）
- SPV-MIA v1 实装：`src/model_contamination/stage_sft/spv_mia.py` + `tests/test_spv_mia.py`（19 单测）

---

## Update 2026-07-01 evening：Paraphrase smoke v3（chat template fix 验证）+ 方案 pivot 暂停点

session 12 修好 `_score_math_cot` chat template（v2 病根：SFT ckpt 拿不到训练分布 prompt）。v3 smoke 3 ckpt × 2 bench × n=10 结果：

| ckpt | bench | acc_orig | worst | signal | verdict |
|---|---:|---:|---:|---:|---|
| clean | gsm8k | 0.3 | 0.1 | 0.2 | SUSPECT |
| clean | mmlu-pro | 0.2 | 0.1 | 0.1 | SUSPECT |
| **gsm8k_heavy** | **gsm8k** | **0.6** | 0.3 | **0.3** | ✅ **DIRTY** |
| gsm8k_heavy | mmlu-pro | 0.0 | 0.1 | −0.1 | CLEAN |
| mmlu_heavy | gsm8k | 0.1 | 0.2 | −0.1 | CLEAN |
| mmlu_heavy | mmlu-pro | 0.1 | 0.0 | 0.1 | SUSPECT |

**结论**：
- ✅ **chat template fix 在 gsm8k CoT 侧生效**：`gsm8k_heavy × gsm8k acc_orig` 从 v2 的 0 抬到 0.6，`clean × gsm8k` 从 0 抬到 0.3。方法在数学 CoT 上有分辨率（gsm8k_heavy 唯一 DIRTY）。
- ❌ **MMLU 侧仍未通过**：mmlu_heavy 在 mmlu-pro 上 acc_orig 只 0.1、signal 0.1 SUSPECT（不是 DIRTY）。`_score_multiple_choice` 走的分支未确认，可能 fix 只覆盖到黑盒 fallback 分支，logprobs 分支仍旧硬编码 prompt。
- ❌ **cross-task degrade 复现**：`gsm8k_heavy × mmlu-pro acc_orig = 0.0`，与 SPV-MIA Finding 8 同类现象——阈值不能设在 raw signal 上，需 clean ckpt 做 contrastive anchor。
- ⚠️ Clean 也 SUSPECT（signal 0.1–0.2）——门限过松 + Qwen3-Base 自身 gsm8k 污染（§2.1 Finding 1）叠加。

**产物**：`outputs/2026-06-27_sft_contam_gt_paraphrase/20260701-160629/result.json` + `paraphrase_smoke_v3.log`。

**暂停点**：2026-07-01 晚方案 pivot——用户决定放弃"base / SFT / RLHF 阶段归因"框架，改为"两阶段（预训练 / 后训练）× 单一目标（检测污染）"框架，前期方法验证改用开源已知污染模型代替自建对照组。见根目录 `模型污染检测方案-v2.md` + `实验计划-v2.md`。**本笔记（2026-06-27 SFT contam GT）作为方案 v1 时代的历史材料保留，其数据仍可用作方法灵敏度参考**，但差分归因层 / SFT 单阶段专属信号的语义将随 v2 重排。

**下 session 首要动作（v2 路线）**：
1. 排查 `shared/evaluator.py::_score_multiple_choice` 的 chat template 覆盖，修完再扩到 200 题 —— 或者随 v2 决策直接放弃 MC 路径，改用 perm_option 覆盖 MC 场景
2. 按 v2 计划 E1 选定 open-source positive control 模型（Qwen3-1.7B-Base 已就位，可作 GSM8K anchor）

---

## Update 2026-07-03：E0.1/E0.3 收尾（`_score_multiple_choice` 路由修复）

### E0.1 分支覆盖表

`_score_multiple_choice` 两分支：

| 分支 | 触发条件 | chat template 状态 |
|---|---|---|
| LOGPROBS `mean-logprob over choice text` | `model.supports(LOGPROBS)` | ❌ 未走 chat template，直接对 `_mc_prompt(q) + " {choice}"` 打分 |
| 黑盒 fallback | 不支持 LOGPROBS | ✅ session 12 已修 |

SFT ckpt 支持 LOGPROBS → 走上面这条 → v3 smoke `mmlu_heavy × mmlu-pro acc_orig=0.1`。

### E0.3 决策：路由修复，不改 LOGPROBS 分支本身

**根因**：`mean-logprob over raw choice text` 的设计假设是"模型会把 choice 文本作为自然 continuation 输出"——对 chat SFT ckpt 完全不成立（它训练成"输出字母"，raw choice text 不是训练分布）。即使给 LOGPROBS 分支 chat-wrap prompt，让 assistant 续 raw choice text 仍不是训练分布，信号仍被噪声淹没。

**改动**（`src/model_contamination/shared/evaluator.py::_score_multiple_choice`）：
- 有 chat template 的模型 → **强制**走黑盒（要模型输出字母），无论是否支持 LOGPROBS
- 无 chat template + 支持 LOGPROBS（base 类）→ 保留 mean-logprob 路径
- 无 chat template + 仅 generate → 硬模板黑盒 fallback

**为什么不弃 MC paraphrase 走 perm_option**：paraphrase 抓的是"verbatim memorization"（question 文本级别）， perm_option 抓的是"option position memorization"（选项顺序级别），信号维度不同；[[project-scope-internal-model]] 里 perm_option 边际价值低的判断也是相对内部模型场景，不代表要独占 MC 路径。

**单测**：`test_mc_chat_model_forces_blackbox_over_logprobs`——logprob 侧故意让错误 choice 最高，仅当路由到黑盒时才能答对；138 单测全 pass。

### 待做（8 卡机 launch）

E0.4 200 题正式 paraphrase smoke，验证 MC 侧修复：

```bash
PYTHONPATH=src HF_ENDPOINT=https://hf-mirror.com \
HF_DATASETS_CACHE=/mnt/public/code/chennuoxi/hf_cache \
HF_HOME=/mnt/public/code/chennuoxi/hf_cache HF_HUB_OFFLINE=1 \
python3 experiments/2026-06-27_sft_contam_gt/run_paraphrase.py \
    --only-ckpt clean,gsm8k_heavy,mmlu_heavy --only-bench gsm8k,mmlu-pro \
    --n-samples 200 --n-paraphrases 5 \
    2>&1 | tee experiments/2026-06-27_sft_contam_gt/paraphrase_full.log
```

**关键 assertion**：`mmlu_heavy × mmlu-pro acc_orig` 应从 v3 的 0.1 抬到 ≥0.3（SFT ckpt 泛化 + memorization 双重）；signal 应 ≥0.15 判 DIRTY。若 mmlu 侧仍 acc_orig<0.2 → 说明 SFT 训练量不足以让 MC 记住，需换更大剂量 GT 或直接弃 MC paraphrase。
