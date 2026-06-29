"""Guided Instruction（Golchin & Surdeanu ICLR 2024, arXiv:2308.14352）。

原理：给模型两种 prompt 让它续写题目后半：
- **guided**：明确告知 benchmark 名 + split + 题目前半
- **general**：只给前半，不告知 benchmark
若模型见过该 benchmark，guided 续写与真实后缀的相似度显著高于 general（"看到名字就回忆起来了"）。
对每题算 ΔROUGE-L = ROUGE_guided - ROUGE_general，配对单侧 t 检验。

接口契约（与 oren_permutation_test 对齐）：
    输入 model (黑盒，只需 GENERATE) + spec + 已加载的 questions
    输出 DetectionResult(method="guided", signal=p_value)

特点：
- 黑盒、门槛低，对 IFEval / SimpleQA 等"传统污染方法不适用"benchmark 也能用
- 可能被 RLHF 安全过滤阻断（自研模型无此问题）
- ROUGE-L 用 LCS-based F1，自实装避免新增依赖
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from model_contamination.models.base import Capability, ModelInterface
from model_contamination.types import (
    BenchmarkQuestion,
    BenchmarkSpec,
    DetectionResult,
    Stage,
    Verdict,
)

_GEN_MAX_TOKENS_DEFAULT = 128
_ALPHA_DIRTY = 0.01
_ALPHA_SUSPECT = 0.05


def guided_instruction_test(
    model: ModelInterface,
    benchmark: BenchmarkSpec,
    questions: list[BenchmarkQuestion],
    *,
    prefix_ratio: float = 0.5,
    max_tokens: int = _GEN_MAX_TOKENS_DEFAULT,
    split_label: str = "test",
    min_samples: int = 30,
) -> DetectionResult:
    """Guided Instruction 主入口。

    prefix_ratio: 题面切分比例（按 word，0.5 = 前半给模型续写后半）
    max_tokens: 单次生成上限；过短会拖低 ROUGE 上限
    split_label: 数据集划分标签（"test" / "validation"），写入 guided prompt 增强提示
    min_samples: 题数低于此返回 INCONCLUSIVE（生成方法成本高，不宜小样本判定）

    信号语义：p_value < alpha → 拒绝"guided 与 general 续写质量相同"原假设 → 污染嫌疑。
    """
    stage = Stage(model.stage_tag)

    if not model.supports(Capability.GENERATE):
        return DetectionResult(
            method="guided", stage=stage, benchmark=benchmark.name,
            signal=None, verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            error=f"{model.name} does not support GENERATE; guided instruction requires generation.",
        )

    if not 0.1 <= prefix_ratio <= 0.9:
        return DetectionResult(
            method="guided", stage=stage, benchmark=benchmark.name,
            signal=None, verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            error=f"prefix_ratio must be in [0.1, 0.9], got {prefix_ratio}.",
        )

    if len(questions) < min_samples:
        return DetectionResult(
            method="guided", stage=stage, benchmark=benchmark.name,
            signal=None, verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            error=f"Need >= {min_samples} samples, got {len(questions)}.",
        )

    pairs = []
    skipped = 0
    for q in questions:
        text = _question_text(q)
        prefix, reference = _split_words(text, prefix_ratio)
        if not prefix or not reference:
            skipped += 1
            continue
        pairs.append((prefix, reference))
    if len(pairs) < min_samples:
        return DetectionResult(
            method="guided", stage=stage, benchmark=benchmark.name,
            signal=None, verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            error=f"Only {len(pairs)} usable samples after splitting (skipped {skipped}).",
        )

    guided_prompts = [_guided_prompt(benchmark, split_label, prefix) for prefix, _ in pairs]
    general_prompts = [_general_prompt(prefix) for prefix, _ in pairs]
    guided_outs = model.batch_generate(guided_prompts, max_tokens=max_tokens, temperature=0.0)
    general_outs = model.batch_generate(general_prompts, max_tokens=max_tokens, temperature=0.0)

    rouge_guided = np.array(
        [rouge_l_f1(g, ref) for g, (_, ref) in zip(guided_outs, pairs)], dtype=np.float64
    )
    rouge_general = np.array(
        [rouge_l_f1(g, ref) for g, (_, ref) in zip(general_outs, pairs)], dtype=np.float64
    )
    deltas = rouge_guided - rouge_general

    p_value = _single_sided_paired_t_test(deltas)
    if p_value < _ALPHA_DIRTY:
        verdict = Verdict.DIRTY
    elif p_value < _ALPHA_SUSPECT:
        verdict = Verdict.SUSPECT
    else:
        verdict = Verdict.CLEAN

    return DetectionResult(
        method="guided", stage=stage, benchmark=benchmark.name,
        signal=float(p_value), verdict_hint=verdict, prerequisites_met=True,
        evidence={
            "n_samples": int(len(pairs)),
            "skipped": skipped,
            "mean_rouge_guided": float(rouge_guided.mean()),
            "mean_rouge_general": float(rouge_general.mean()),
            "mean_delta": float(deltas.mean()),
            "std_delta": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
            "prefix_ratio": prefix_ratio,
        },
        error=None,
    )


# ----------------------------- prompts ----------------------------- #


def _question_text(q: BenchmarkQuestion) -> str:
    """重建题目原文。MC 题目把选项也算进去，让 guided 提示有更多 "记忆锚点"。"""
    if q.format == "multiple_choice" and q.choices:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(q.choices))
        return f"{q.prompt}\n{body}"
    return q.prompt


def _split_words(text: str, ratio: float) -> tuple[str, str]:
    words = text.split()
    if len(words) < 4:
        return "", ""
    cut = max(1, int(len(words) * ratio))
    cut = min(cut, len(words) - 1)
    return " ".join(words[:cut]), " ".join(words[cut:])


def _guided_prompt(benchmark: BenchmarkSpec, split_label: str, prefix: str) -> str:
    """带 benchmark 标识的续写提示。沿用论文用语：明确数据集名 + split + "exactly"。"""
    return (
        f"You are provided with the first piece of an instance from the {split_label}"
        f" split of the {benchmark.name} dataset. Finish the second piece of the instance"
        f" as exactly appeared in the dataset. Only rely on the original form of the"
        f" instance in the dataset to finish the second piece.\n\n"
        f"First piece: {prefix}\n\n"
        f"Second piece:"
    )


def _general_prompt(prefix: str) -> str:
    """对照组：不告知 benchmark 名，仅请求自然续写。"""
    return (
        f"Finish the second piece of the following text based on the first piece, such"
        f" that these two pieces become consistent.\n\n"
        f"First piece: {prefix}\n\n"
        f"Second piece:"
    )


# ----------------------------- ROUGE-L ----------------------------- #


def rouge_l_f1(hypothesis: str, reference: str) -> float:
    """ROUGE-L F1（基于 word LCS）。空输入返回 0。"""
    hyp = hypothesis.split()
    ref = reference.split()
    if not hyp or not ref:
        return 0.0
    lcs = _lcs_length(hyp, ref)
    if lcs == 0:
        return 0.0
    precision = lcs / len(hyp)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longest common subsequence 长度，O(|a|·|b|) DP。"""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # 滚动数组省内存
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        ai = a[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
        for k in range(n + 1):
            curr[k] = 0
    return prev[n]


# ----------------------------- stats ----------------------------- #


def _single_sided_paired_t_test(deltas: np.ndarray) -> float:
    """deltas = ROUGE_guided - ROUGE_general 各样本的差，做单侧 t 检验 H1: mean > 0。"""
    if len(deltas) < 2:
        return 1.0
    # std=0 但 mean>0 时 ttest_1samp 返 NaN；这种情况视为极强信号
    if float(deltas.std(ddof=1)) == 0.0:
        return 0.0 if deltas.mean() > 0 else 1.0
    t_stat, p_two_sided = stats.ttest_1samp(deltas, popmean=0.0)
    if t_stat > 0:
        return float(p_two_sided / 2)
    return float(1.0 - p_two_sided / 2)
