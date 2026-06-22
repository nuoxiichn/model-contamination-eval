"""Oren 分片排列检验（Oren et al. ICLR 2024, arXiv:2310.17623）。

原理：若模型未见过某数据集，对样本的任何排列应赋予等似然度；若见过，会偏好
原始顺序。将测试集分片，比较"原始顺序"vs"随机打乱"的 log-likelihood 之和，
做单侧 t 检验。

接口契约：
    输入 model (灰盒，需 logprobs) + benchmark + 已加载的 questions
    输出 DetectionResult(method="oren", signal=p_value)
    p_value < alpha → 拒绝"未见过"原假设，判为污染

优点：唯一有 FPR 数学保证的方法；不受时间偏移影响
缺点：只能数据集级判定；论文建议 ≥200 样本；不能定位单样本
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

_SEPARATOR = "\n\n"


def oren_permutation_test(
    model: ModelInterface,
    benchmark: BenchmarkSpec,
    questions: list[BenchmarkQuestion],
    *,
    shard_size: int = 25,
    alpha_dirty: float = 0.01,
    alpha_suspect: float = 0.05,
    min_samples: int = 200,
    seed: int = 42,
) -> DetectionResult:
    """对给定 benchmark 跑 Oren 排列检验。

    shard_size: 每个 shard 内的题目数。len(questions) // shard_size = shard 数。
    论文用 25 题 / shard，shard 数 ≥ 8（即至少 200 题）。

    Phase 1 简化：单次置换 / shard（跟论文）。
    """
    stage = Stage(model.stage_tag)

    if not model.supports(Capability.LOGPROBS):
        return DetectionResult(
            method="oren", stage=stage, benchmark=benchmark.name,
            signal=None, verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            error=f"{model.name} does not support logprobs; Oren test requires logprobs.",
        )

    if len(questions) < min_samples:
        return DetectionResult(
            method="oren", stage=stage, benchmark=benchmark.name,
            signal=None, verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            error=f"Need >= {min_samples} samples for Oren test, got {len(questions)}.",
        )

    texts = [_format_question(q) for q in questions]
    rng = np.random.default_rng(seed)

    diffs: list[float] = []
    n_shards = len(texts) // shard_size
    for s in range(n_shards):
        shard = texts[s * shard_size : (s + 1) * shard_size]
        ll_orig, ll_perm = _shard_loglikelihood(model, shard, rng)
        diffs.append(ll_orig - ll_perm)

    diffs_arr = np.asarray(diffs, dtype=np.float64)
    p_value = _single_sided_t_test(diffs_arr)

    if p_value < alpha_dirty:
        verdict = Verdict.DIRTY
    elif p_value < alpha_suspect:
        verdict = Verdict.SUSPECT
    else:
        verdict = Verdict.CLEAN

    return DetectionResult(
        method="oren", stage=stage, benchmark=benchmark.name,
        signal=float(p_value), verdict_hint=verdict, prerequisites_met=True,
        evidence={
            "n_shards": n_shards,
            "shard_size": shard_size,
            "mean_diff": float(diffs_arr.mean()),
            "std_diff": float(diffs_arr.std(ddof=1)) if len(diffs_arr) > 1 else 0.0,
            "diffs": diffs_arr.tolist(),
        },
    )


def _format_question(q: BenchmarkQuestion) -> str:
    """统一拼接形式：题面 + 答案（让 logp 也覆盖正确答案信号）。"""
    if q.format == "multiple_choice" and q.choices:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(q.choices))
        return f"Question: {q.prompt}\n{body}\nAnswer: {q.answer}"
    return f"Question: {q.prompt}\nAnswer: {q.answer}"


def _shard_loglikelihood(
    model: ModelInterface,
    shard_texts: list[str],
    rng: np.random.Generator,
) -> tuple[float, float]:
    """单 shard 的 (L_original, L_permuted)。

    把 shard_size 个题用 _SEPARATOR 拼成一长串，喂 model.logprobs(prefix, full)；
    prefix 取 _SEPARATOR 避免空 prompt 边界（tokenizer 可能不加 BOS）。
    原序 vs 一次随机置换的 LL 差就是该 shard 的 contamination 信号。
    """
    ll_orig = _ll_of_order(model, shard_texts)
    perm_idx = rng.permutation(len(shard_texts))
    shuffled = [shard_texts[i] for i in perm_idx]
    ll_perm = _ll_of_order(model, shuffled)
    return ll_orig, ll_perm


def _ll_of_order(model: ModelInterface, items: list[str]) -> float:
    text = _SEPARATOR.join(items)
    lp = model.logprobs(prompt=_SEPARATOR, completion=text)
    return float(np.sum(lp))


def _single_sided_t_test(diffs: np.ndarray) -> float:
    """diffs = L_orig - L_perm 各 shard 的差值，做单侧 t 检验。返回 p-value。"""
    if len(diffs) < 2:
        return 1.0
    t_stat, p_two_sided = stats.ttest_1samp(diffs, popmean=0.0)
    # 单侧 H1: mean(diff) > 0
    if t_stat > 0:
        return float(p_two_sided / 2)
    return float(1.0 - p_two_sided / 2)
