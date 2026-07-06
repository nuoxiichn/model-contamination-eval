# Qwen3-1.7B 同源对 × Min-K%++ ΔMIA（2026-06-22）

## 结论

| | base (Qwen3-1.7B-Base) | sft (Qwen3-1.7B) | Δ |
|---|---|---|---|
| AUC (target vs control) | 0.243 | 0.388 | **+0.145** |
| target_mean (Min-K%++) | -2.118 | -11.661 | -9.543 |
| control_mean (Min-K%++) | -1.274 | -6.350 | -5.077 |
| target - control mean | -0.844 | -5.311 | — |

target = GSM8K test×200，control = MATH-500 test×200，k=0.2。

**ΔAUC = +0.145，方向符合预期**：SFT 阶段让 GSM8K 相对 MATH-500 的"竞争力"上升了 14.5 个百分点。如果 SFT 训练数据里 GSM8K 风格题更多 / 直接见过 GSM8K，这正是 reference-based MIA 想要的归因信号。

但**绝对幅度不强**：AUC_sft 仍 < 0.5，单点上 control 看起来比 target 更可能被见过。两种解释并存：
1. MATH-500 在 SFT 阶段同样被相当程度地见过（control 不干净）
2. SFT 阶段引入的 GSM8K 增量小于"两个集合本身的先验难度差"

## 为什么 base 阶段就有 AUC=0.243 偏置

参见 `experiments/2026-06-22_qwen3-1.7b-base_minkpp/notes.md`：MATH-500 出自 MATH 训练集，Qwen3 base 几乎必然见过 MATH 训练集；LaTeX 数学表达式 token 分布尖锐 → control_mean 自然更高。control 集本身被污染是 base 阶段 AUC 反向的主因。

**重要观察**：base 阶段就有 AUC<0.5 的偏置，ΔAUC 抵消了这部分先验偏置。这就是 ΔMIA 比单点 MIA 可信的核心理由——同源对吃同样的 confound，相减消大头。

## target_mean / control_mean 在 SFT 后大幅下降的原因

base 的 -2.1 → sft 的 -11.7，几乎一个数量级。机制：SFT 把 chat 模型的输出分布训成"更尖锐"（给定上下文下，预期答案 token 概率大、其他 token 概率小），σ_i 增大、μ_i 更负，归一化后的 chosen log p（即 normalized log p = (logp - μ)/σ）整体偏向更负数。

**含义**：Min-K%++ 的 absolute score 在不同 stage 间**不可比**，只能在同一 stage 内做 target vs control。这是把 AUC（rank-based）当 signal 而非 mean-based 的一个有力理由——AUC 不受 score 整体平移/缩放影响。

## 信号强度与红线

- 无 positive control（已知污染对照模型）→ ΔAUC=0.145 算"中等偏弱方向性信号"，**不出绝对裁决**
- 仍然是项目**第一次**拿到非 0.5 中性的、可解读的 SFT 阶段归因信号
- 若后续用 LiveCodeBench / AIME-2025（严格新于训练 cutoff）做 control，再看 ΔAUC，应能更确信

## 配置

详见 `run.yaml`。Qwen3-1.7B README 直接声明 `base_model: Qwen/Qwen3-1.7B-Base`，官方同源；两个 model 都 bf16 / cuda / MetaX C500，单 stage ≈ 16s。

输出文件：`outputs/qwen3-1.7b_delta_mia/20260622-210014/result.json`（不进 git）。

## Follow-up

- 同样的同源对 + 同样 target/control，跑一遍 `guided_instruction_test`（已实装），对比 Δp_value
- 同样的同源对，换 control 为 LiveCodeBench 或 AIME-2025（更可信干净），看 ΔAUC 是否变大
- 把这个脚本泛化成 CLI 命令 `mcd delta-mia --base PATH --sft PATH --target X --control Y`，进 cli.py
- 加 family_diff 跑同源对，看是否在 SFT 阶段出现"任务孤岛"
