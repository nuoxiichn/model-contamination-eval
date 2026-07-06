# Qwen3-1.7B-Base × Min-K%++ smoke（2026-06-22）

## 结论

- **Plumbing 跑通**：HFLocalModel.token_logprob_stats + Min-K%++ 端到端 200 题 ×2 集 / 17 秒 / GPU bf16
- **Base 单点信号**：AUC=0.243，delta_mean=-0.84（target_mean=-2.12，control_mean=-1.27）
- **方向反了，但 verdict 仍诚实**：CLAUDE.md 红线"base 阶段 Min-K%++ AUC ≈ 0.5 不单独定性"，verdict_hint=CLEAN 符合阈值

## 配置

| 项 | 值 |
|---|---|
| model | `/mnt/public/model/Qwen/Qwen3-1.7B-Base` |
| stage | base |
| dtype | bfloat16 |
| device | cuda (MetaX C500) |
| target | gsm8k test, N=200 |
| control | MATH-500 test, N=200 |
| k_ratio | 0.2 |

详见 `run.yaml` 与 `outputs/qwen3-1.7b-base_minkpp/20260622-200802/result.json`。

## 方向反了的备选解读

简单假设是 "GSM8K 见过 → target 分数应该更高 → AUC > 0.5"。实际相反。可能原因：

1. **MATH-style LaTeX 在 Qwen3 base 中更熟悉**：MATH-500 是 MATH 训练集的 test split；如果 base 在 MATH 训练集上过得多，对 LaTeX 数学表达式的 conditional 分布更"尖"，每个 token 的 normalized log p 更高 → control 分数高。这其实是 control 集本身不干净的反例（红线场景）。
2. **GSM8K 自然语言长格式 CoT 难度高于 LaTeX 简洁推导**：Qwen3 base 对 chain-of-thought 自然语言推理整体不擅长（base 没 SFT），token-by-token 不确定性大 → normalized log p 偏低。
3. **样本长度差异**：GSM8K Q+A ≈ 100 token，MATH-500 Q+A 经常 200+ token。bottom-K% 在不同长度上的稳定性不同。
4. **k_ratio=0.2 不是最优**：论文里这是常用值，但不同 corpora 上 sweet spot 可能差异较大；后续可以 sweep k∈{0.1, 0.2, 0.3}。

**重要**：方向反这件事本身**就是 CLAUDE.md 红线想防的**——单 benchmark 的 MIA AUC 不能当唯一证据。需要：
- 同源 SFT checkpoint，看 ΔMIA = AUC_sft - AUC_base 的符号才有归因价值
- 或者 canary 注入做 known-positive 校准
- 当前在没有 positive control 的窗口期，本结果只作"plumbing 已就绪 + base 单点信号已记录"，不解读 / 不裁决

## 时间预算

- 200 题 × 2 集 = 400 次 forward，bf16 GPU 共 17s
- 推算：8 个 P1-pretrain benchmark × 平均 2 个 control 集 ≈ 5 分钟级，完全可行
- 单题 forward 平均 ~40ms（含 ~100-200 token 的 prompt+completion）

## 已知坑

- `dtype="bfloat16"` 触发 transformers 4.x 的 `torch_dtype` deprecation warning（仓库当前还在用旧参数名传 HFLocalModel）
- MetaX C500 上 SDPA 不支持 memory_efficient_attention，会回退到 math 实现；warning 可忽略
- 没有同源 SFT checkpoint，**ΔMIA 跑不出来**；本次只能产 base 单点信号

## Follow-up

- 拿到 Qwen3-1.7B-Base 同源 SFT 后立刻跑差分（同脚本，换 model_path 与 stage）
- 用 LiveCodeBench / AIME-2025 这种"严格时间晚于训练 cutoff"的 benchmark 当 control，比 MATH-500 更干净
- 跑 k_ratio sweep 看 AUC 稳定性
- 加 evidence 里的 `target_lens`/`control_lens` 字段，量化长度差异对 bottom-K% 的影响
