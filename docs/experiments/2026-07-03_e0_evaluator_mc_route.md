# E0 收尾：evaluator chat template MC 路由修复（2026-07-03）

## 背景

Paraphrase smoke v3（`20260701-160629`）暴露两条：
- ✅ gsm8k CoT 侧（session 12 修 `_score_math_cot`）已通：`gsm8k_heavy × gsm8k acc_orig=0.6, signal=0.3 DIRTY`
- ❌ MMLU MC 侧仍未通：`mmlu_heavy × mmlu-pro acc_orig=0.1, signal=0.1 SUSPECT`

E0 任务是定位 MC 侧问题并决策修否。

## E0.1 分支覆盖表

`_score_multiple_choice` 两分支：

| 分支 | 触发条件 | chat template 状态 |
|---|---|---|
| LOGPROBS `mean-logprob over choice text` | `model.supports(LOGPROBS)` | ❌ 未 chat-wrap，直接对 `_mc_prompt(q) + " {choice}"` 打分 |
| 黑盒 fallback | 不支持 LOGPROBS | ✅ session 12 已修 |

SFT ckpt 支持 LOGPROBS → 走上面 → acc_orig 假阴。

## E0.3 决策

**根因**：`mean-logprob over raw choice text` 假设"模型把 choice 文本作为自然 continuation 输出"——对 chat SFT ckpt 完全不成立（它训练成"输出字母"）。给 prompt chat-wrap 也没用，assistant 续 raw choice text 仍不是训练分布。

**改动**（`src/model_contamination/shared/evaluator.py::_score_multiple_choice`）：路由修复，不改 LOGPROBS 分支本身：
- 有 chat template（SFT / chat 类）→ **强制**走黑盒（要模型输出字母），无论是否支持 LOGPROBS
- 无 chat template + LOGPROBS（base 类）→ 保留 mean-logprob 路径
- 无 chat template + 仅 generate → 硬模板黑盒 fallback

**为什么不弃 MC paraphrase 走 perm_option**：paraphrase 抓"verbatim memorization"（question 文本级），perm_option 抓"option position memorization"（选项顺序级），信号维度不同。

**单测**：`test_mc_chat_model_forces_blackbox_over_logprobs`——logprob 侧故意让错误 choice 最高，仅当路由到黑盒时才能答对。138 单测全 pass。

## 待做（8 卡机 launch）

E0.4 200 题正式 paraphrase smoke：

```bash
PYTHONPATH=src HF_ENDPOINT=https://hf-mirror.com \
HF_DATASETS_CACHE=/mnt/public/code/chennuoxi/hf_cache \
HF_HOME=/mnt/public/code/chennuoxi/hf_cache HF_HUB_OFFLINE=1 \
python3 experiments/2026-06-27_sft_contam_gt/run_paraphrase.py \
    --only-ckpt clean,gsm8k_heavy,mmlu_heavy --only-bench gsm8k,mmlu-pro \
    --n-samples 200 --n-paraphrases 5 \
    2>&1 | tee experiments/2026-06-27_sft_contam_gt/paraphrase_full.log
```

**关键 assertion**：`mmlu_heavy × mmlu-pro acc_orig` 应从 v3 的 0.1 抬到 ≥0.3；signal ≥0.15 判 DIRTY。若仍 acc_orig<0.2 → 说明 SFT 训练量不足以让 MC 记住，需换更大剂量 GT 或弃 MC paraphrase。

## 数据 pointer

- 修复前 v3：`outputs/2026-06-27_sft_contam_gt_paraphrase/20260701-160629/`
- 修复后 v3-fix：等 E0.4 launch
- Session 13 (2026-07-03) notes 追加于 `experiments/2026-06-27_sft_contam_gt/notes.md` 尾部
