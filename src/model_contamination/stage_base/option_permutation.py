"""选项排列检测（Yang et al. AAAI 2024, arXiv:2305.10403）。

打乱多选题选项顺序，看模型是否偏好原始选项位置。灰盒（需 logprobs），专用于多选题格式。
在 C-Eval 上发现 Qwen 系列泄漏值是其他模型近 10 倍；Qwen2-72B 在 CMB 上 42% 泄漏。

适用场景（U.5 / Finding 9 闭环之后明确）：
    SPV-MIA / Min-K%++ / guided 三家在 MC 题型上结构性失效（completion 太短，paraphrase
    扰动幅度淹没在噪声里）。perm_option 用"题面 + 各选项 + 字母"作为整体单位，
    不依赖 completion 长度，是 MC SFT 污染的主信号方法。

信号约定（与 diff_matrix.signal_direction_for("perm_option") = "higher_is_dirtier" 对齐）：
    signal = mean(acc_orig) - mean(acc_perm_over_k)

    干净模型：选项位置无关，signal ≈ 0
    污染模型：原始顺序题→答案被记住，permutation 后 acc 掉，signal > 0

    chance baseline = 0（同一道题，shuffle 选项不改变内容，clean 模型理应一致）

输入要求：白盒或灰盒模型（需 logprobs）+ multiple_choice benchmark
"""

from __future__ import annotations

import numpy as np

from model_contamination.models.base import Capability, ModelInterface
from model_contamination.types import (
    BenchmarkQuestion,
    BenchmarkSpec,
    DetectionResult,
    Stage,
    Verdict,
)

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# 经验阈值：v3 SFT 7-ckpt calibration 跑出来后再校准
# 干净模型上 signal 应接近 0；显著大于 0 即可疑
_ALPHA_DIRTY = 0.20      # 准确率掉 20+ 点 = DIRTY
_ALPHA_SUSPECT = 0.05    # 5+ 点掉幅 = SUSPECT


def option_permutation_test(
    model: ModelInterface,
    benchmark: BenchmarkSpec,
    questions: list[BenchmarkQuestion],
    *,
    n_permutations: int = 5,
    min_samples: int = 50,
    seed: int = 42,
) -> DetectionResult:
    """对 MC benchmark 跑选项排列检测。

    n_permutations: 每题随机生成多少个排列。论文 k=5 已能稳定区分。
    min_samples: 题目数下限；MMLU-Pro 200 题足够。
    """
    if benchmark.format != "multiple_choice":
        raise ValueError(
            f"option_permutation 仅适用 multiple_choice 格式，"
            f"benchmark {benchmark.name} 是 {benchmark.format}"
        )

    stage = Stage(model.stage_tag)

    if not model.supports(Capability.LOGPROBS):
        return DetectionResult(
            method="perm_option", stage=stage, benchmark=benchmark.name,
            signal=None, verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            error=f"{model.name} does not support logprobs; perm_option requires logprobs.",
        )

    # 过滤合法 MC 题（必须有 choices 且 answer_index 在范围内）
    valid: list[BenchmarkQuestion] = []
    for q in questions:
        if q.choices and q.answer_index is not None and 0 <= q.answer_index < len(q.choices):
            valid.append(q)

    if len(valid) < min_samples:
        return DetectionResult(
            method="perm_option", stage=stage, benchmark=benchmark.name,
            signal=None, verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            error=(
                f"Need >= {min_samples} valid MC questions (with choices + answer_index), "
                f"got {len(valid)} out of {len(questions)}."
            ),
            evidence={"n_valid": len(valid), "n_total": len(questions)},
        )

    rng = np.random.default_rng(seed)
    acc_orig_list: list[int] = []
    acc_perm_list: list[float] = []  # 每题在 k 个 permutation 上的平均 acc

    for q in valid:
        n_opt = len(q.choices)
        # 原序 accuracy
        pred_orig = _predict_letter(model, q.prompt, q.choices)
        acc_orig = 1 if pred_orig == _LETTERS[q.answer_index] else 0
        acc_orig_list.append(acc_orig)

        # k 个随机排列下的 accuracy
        perm_accs: list[int] = []
        for _ in range(n_permutations):
            perm = rng.permutation(n_opt)
            shuffled_choices = [q.choices[i] for i in perm]
            # 正确内容现在落在哪个新位置：perm[new_idx] = orig_idx
            new_correct_idx = int(np.where(perm == q.answer_index)[0][0])
            pred = _predict_letter(model, q.prompt, shuffled_choices)
            perm_accs.append(1 if pred == _LETTERS[new_correct_idx] else 0)
        acc_perm_list.append(float(np.mean(perm_accs)))

    mean_orig = float(np.mean(acc_orig_list))
    mean_perm = float(np.mean(acc_perm_list))
    signal = mean_orig - mean_perm

    if signal >= _ALPHA_DIRTY:
        verdict = Verdict.DIRTY
    elif signal >= _ALPHA_SUSPECT:
        verdict = Verdict.SUSPECT
    else:
        verdict = Verdict.CLEAN

    return DetectionResult(
        method="perm_option", stage=stage, benchmark=benchmark.name,
        signal=signal, verdict_hint=verdict, prerequisites_met=True,
        evidence={
            "n_questions": len(valid),
            "n_permutations": n_permutations,
            "mean_acc_original": mean_orig,
            "mean_acc_permuted": mean_perm,
            "leak_score": signal,  # 别名，对应论文术语
            "alpha_dirty": _ALPHA_DIRTY,
            "alpha_suspect": _ALPHA_SUSPECT,
        },
    )


def _predict_letter(model: ModelInterface, question: str, choices: list[str]) -> str:
    """对单题一次 forward 拿全部字母的 next-token logp，argmax 出预测字母。

    使用最小提示词避免污染 (Question/Answer 锚点 + 选项行)，与 oren 的 _format_question 风格一致。
    通过 `model.next_token_logprobs` 批量：HFLocalModel 在字母全为单 token 时
    1 forward 完成，比 N=len(choices) 次 logprobs() 快 5-10×。
    """
    n = len(choices)
    body = "\n".join(f"{_LETTERS[i]}. {c}" for i, c in enumerate(choices))
    prompt = f"Question: {question}\n{body}\nAnswer:"

    candidates = [f" {_LETTERS[i]}" for i in range(n)]
    logps = model.next_token_logprobs(prompt, candidates)
    best_idx = int(np.argmax(logps))
    return _LETTERS[best_idx]
