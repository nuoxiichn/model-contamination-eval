# Scenario b（Ni et al. AAAI 2025 Algorithm 2）Qwen 复现

**目标**：复现《Training on the Benchmark Is Not All You Need》(arXiv:2409.01790, AAAI 2025)
的 **Scenario b**（= Algorithm 2，不需要原始选项顺序，靠 IsolationForest 检测序列
logprob 离群），验证论文 headline：**Qwen 系列在中文 MC benchmark 上泄漏率显著高于
抗污染对照**。

## 方法映射（本仓库实现 ↔ 论文）

| 论文 | 本仓库 |
| --- | --- |
| Scenario b / Algorithm 2 | `option_permutation_test`（`src/.../stage_base/option_permutation.py`） |
| 选项块序列 logprob（从首个 `A:` 起，条件在题面上，不归一） | `_permutation_logprobs`：`sum logp(选项块 \| 题面)` |
| n! 排列枚举 | `max_permutations`；n! ≤ 上限则全枚举，超出随机采样恒含原序 |
| IsolationForest 检测 max-logprob 是否离群 | `IsolationForest(n_estimators=100, contamination='auto')` + `decision_function` |
| 阈值 -0.2 / -0.17 / -0.15 | `_THRESHOLDS`，primary = -0.17 |
| 泄漏率 = 离群题数 / 总题数 | `signal = leak_fraction` ∈ [0,1]，higher_is_dirtier |

实现与官方 `inference_logprobs.py` / `get_outlier.py` 逐行核对一致（见 codec 分支的
调研记录）。单测 11 条覆盖 clean（均匀→无离群）/ memorized（原序离群）/ 采样分支。

## 实验矩阵

模型（均为 base，Scenario b 检测预训练泄漏）：
- **Qwen2.5-1.5B**（本地缓存）
- **Qwen2.5-7B**（本次下载到 `/mnt/public/code/chennuoxi/hf_cache`）

Benchmark（`prepare_data.py` 导出 JSONL）：
- **c-eval** / **cmmlu** / **cmb** —— 中文，Qwen 训练重灾区，期望高泄漏
- **mmlu-cf** —— 抗污染对照（contamination-free by design），期望低泄漏

## ⚠️ 与论文的差异（复现范围诚实声明）

1. **模型规模**：论文 headline（Qwen2-72B 在 CMB 42% 泄漏、Qwen 家族 ~10× 其他模型）
   是在 **Qwen2-72B / Qwen1.5-110B** 上。本地条件只到 **1.5B / 7B**，且是 **Qwen2.5**
   世代（不同预训练数据）。故本复现验证的是 **「Qwen 家族存在泄漏 + 是否随规模上升」
   的趋势**，不是 42% 的绝对量级。论文规律：模型越大泄漏越高 → 期望 7B > 1.5B。
2. **CMB 无 gold answer**：test split 不含 answer（留 leaderboard）。Algorithm 2 本就
   不需要金标，`answer_index` 置 0 占位。
3. **绝对裁决未校准**：无 positive control，`_ALPHA_DIRTY/_SUSPECT` 是 provisional。
   IsolationForest 在干净数据上有基线假阳率，leak_fraction 的 clean 基线 ≠ 0 而是该
   FPR。**结论只看相对排名**：中文三 bench vs mmlu-cf 对照的分离，以及 7B vs 1.5B 的
   单调性。符合仓库红线（positive control 未到位前不出绝对红/黄/绿）。

## 运行步骤

```bash
# 1) 开发机（联网）导出数据 —— 已跑过 --per-benchmark 40 冒烟，正式跑加大
HF_ENDPOINT=https://hf-mirror.com HF_HOME=/mnt/public/code/chennuoxi/hf_cache \
PYTHONPATH=src python3 experiments/2026-07-08_perm_option_scenario_b/prepare_data.py \
  --per-benchmark 300

# 2) 8 卡机（有 GPU）跑推理
HF_HOME=/mnt/public/code/chennuoxi/hf_cache CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=src python3 experiments/2026-07-08_perm_option_scenario_b/run_scenario_b.py
```

## 待跑清单（handoff）

- [x] Qwen2.5-7B 下载完成确认（`/tmp/qwen7b_dl.log` 出现 `7B DONE`）
- [x] `prepare_data.py --per-benchmark 300` 正式导出
- [x] 8 卡机跑满 3×4 矩阵（1.5B / 7B / 72B）
- [x] 分析：中文三 bench leak_fraction vs mmlu-cf 分离度；跨规模单调性
- [ ] 结论回写 `../docs/experiments/`（附 commit hash）

## 结果

leak_fraction @ primary threshold -0.17（300 题/bench，24 排列/题）：

| model | c-eval | cmmlu | cmb | mmlu-cf(对照) | 中文均值 | **Δ(中文−对照)** |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-1.5B | 0.467 | 0.433 | 0.453 | 0.433 | 0.451 | **+0.018** |
| Qwen2.5-7B  | 0.503 | 0.500 | 0.510 | 0.423 | 0.504 | **+0.081** |
| Qwen2.5-72B | 0.667 | 0.683 | **0.713** | **0.327** | 0.688 | **+0.361** |

更严阈值 -0.2 下 72B 分离更干净：cmb 0.553 vs mmlu-cf 0.140（Δ=+0.413）。

### 结论（趋势强复现，量级非论文绝对值）

1. **分离度随规模单调爆发式扩大**：+0.018 → +0.081 → +0.361。72B 上中文三 bench
   全部大幅抬升，复现论文「Qwen 越大、污染 benchmark 泄漏越高」。
2. **对照集随规模不升反降**（0.433 → 0.423 → 0.327）：证明 1.5B/7B 上 mmlu-cf 的
   ~0.42 假阳地板是**小模型 artifact**（弱模型对某排列的流畅度偏好 → IsolationForest
   误判离群），随规模洗掉，而非模型能力驱动。这是补跑 72B 才暴露的、单看小模型看不出
   的关键点，显著提升方法在大模型上的可信度。
3. **cmb 是全矩阵最高格**（72B=0.713）：对上论文 headline（Qwen2-72B 在 CMB 泄漏最重，
   论文 42%）。四个 benchmark 里 cmb 在 72B 上确为最高。

**仍不出绝对红黄绿**：无 positive control，leak_fraction 未校准到论文百分比口径；
结论只报跨规模/跨 bench 的相对趋势（仓库红线）。方向、规模单调性、benchmark 排序均复现。

### 复现的工程注记

- `seq_logprob_sums` batch 前向：按 token 长度分桶、桶内零 padding（padding 无论左右
  都让 RoPE 位置错位）。batch vs 逐条存在浮点非结合噪声（相对 ~1e-4，非 bug），gate
  改卡「相对误差 <1e-3 + 每题 argmax 排列一致」而非绝对误差。见 `verify_batch_logprobs.py`。
- 72B 用 device_map=auto 4 卡层切分 × 2 份数据并行（`run_72b_2shard.sh`）。
