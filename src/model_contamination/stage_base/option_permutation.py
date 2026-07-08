"""选项排列检测 —— 对齐 Ni et al. AAAI 2025《Training on the Benchmark Is Not
All You Need》(arXiv:2409.01790) 的 Algorithm 2（shuffled / Scenario b）。

官方实现：github.com/nishiwen1214/Benchmark-leakage-detection

核心思想：打乱多选题选项内容不改变题意。若模型在预训练/SFT 见过该题，会对某一
特定选项顺序赋予异常高的序列概率。枚举全部 n! 排列、算每个排列「选项块」的序列
log-prob，若最大值在这 n! 个值里是统计离群点 → 判该题泄漏。Algorithm 2 **不需要
知道原始顺序**（出题方/预训练可能已打乱），只检测「是否存在离群的最大值」。灰盒
（需 logprobs），专用于多选题格式。论文在 C-Eval 上发现 Qwen 系列泄漏值约为其他
模型 10×；Algorithm 1 检出 Qwen2-72B 在 CMB 上 42% 泄漏。

适用场景（U.5 / Finding 9 闭环后明确）：
    SPV-MIA / Min-K%++ 在 MC 题型上结构性失效（completion 太短，paraphrase 扰动
    幅度淹没在噪声里）。perm_option 用「题面 + 各选项」整块序列概率作单位，不依赖
    completion 长度，是 MC 污染的主信号方法。

算法（每题）：
    1. 枚举 n! 个选项排列（n! 超过 max_permutations 时随机采样，恒含原序）；
    2. 每个排列构造 "{question}:\nA:opt\nB:opt\n..." 的选项块文本，
       score = sum_token logp(选项块 | 题面)（不按长度归一，与论文一致）；
    3. 对这批 score fit IsolationForest，取 argmax(score) 排列的 decision_function
       分数；分数 < outlier_threshold → 该题判泄漏；
    4. 泄漏率 = 泄漏题数 / 有效题数。

信号约定（与 diff_matrix.signal_direction_for("perm_option") = "higher_is_dirtier" 对齐）：
    signal = leak_fraction = 在 primary_threshold(-0.17) 下判泄漏的题目占比 ∈ [0, 1]
    干净模型：各排列 log-prob 近似均匀，最大值不离群 → leak_fraction 接近 IsolationForest
             的基线假阳率（低）
    污染模型：记住的顺序 log-prob 异常高 → 大量题被判离群 → leak_fraction 显著抬升

输入要求：白盒或灰盒模型（需 logprobs）+ multiple_choice benchmark

已知失效场景：
    - 论文实测 shuffled 场景的检测精度显著低于 unshuffled；训练 epoch 少时 recall 低
      （LLaMA2-7B 在阈值 -0.17 下 recall 从 1 epoch 的 49.8% 升到 10 epoch 的 96.2%）
    - 无 positive control → 绝对阈值/裁决未校准，verdict_hint 仅供排序参考
"""

from __future__ import annotations

import math
from itertools import permutations as iter_permutations

import numpy as np
from sklearn.ensemble import IsolationForest

from model_contamination.models.base import Capability, ModelInterface
from model_contamination.types import (
    BenchmarkQuestion,
    BenchmarkSpec,
    DetectionResult,
    Stage,
    Verdict,
)

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# 论文用的三档 IsolationForest.decision_function 阈值（越负越离群 → 越可疑）。
# 报告 primary 一档为主信号，另两档进 evidence 便于敏感度对照。
_THRESHOLDS: tuple[float, ...] = (-0.2, -0.17, -0.15)
_PRIMARY_THRESHOLD = -0.17

# 经验裁决阈值（provisional）：leak_fraction 量纲，无 positive control 前不做绝对裁决。
# 论文中干净模型的泄漏率典型 ~0.10，Qwen 系列可高一个量级；此处按 fraction 粗分档，
# 待 SFT calibration + positive control 落地后重校。
_ALPHA_DIRTY = 0.30      # 泄漏率 ≥ 30% = DIRTY
_ALPHA_SUSPECT = 0.15    # ≥ 15% = SUSPECT

# IsolationForest 至少要几个样本才有意义（n! ≥ 此值才纳入统计）
_MIN_PERMS_FOR_IF = 6


def option_permutation_test(
    model: ModelInterface,
    benchmark: BenchmarkSpec,
    questions: list[BenchmarkQuestion],
    *,
    max_permutations: int = 120,
    min_samples: int = 50,
    outlier_threshold: float = _PRIMARY_THRESHOLD,
    seed: int = 42,
) -> DetectionResult:
    """对 MC benchmark 跑 Algorithm 2 选项排列离群检测。

    max_permutations: 每题枚举排列上限。n! ≤ 此值时枚举全部（4 选项=24、5 选项=120
        天然全枚举）；超过（如 MMLU-Pro ≥6 选项）时随机采样这么多个不重复排列，恒
        含原序。默认 120 覆盖 ≤5 选项全枚举。
    min_samples: 有效 MC 题目数下限。
    outlier_threshold: primary 阈值，主信号 leak_fraction 在此阈值下统计。
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

    # 过滤合法 MC 题：至少 2 个选项才能排列（answer_index 非必需，Algorithm 2 不看金标）
    valid: list[BenchmarkQuestion] = [
        q for q in questions if q.choices and len(q.choices) >= 2
    ]

    if len(valid) < min_samples:
        return DetectionResult(
            method="perm_option", stage=stage, benchmark=benchmark.name,
            signal=None, verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            error=(
                f"Need >= {min_samples} valid MC questions (with >=2 choices), "
                f"got {len(valid)} out of {len(questions)}."
            ),
            evidence={"n_valid": len(valid), "n_total": len(questions)},
        )

    thresholds = tuple(sorted(set(_THRESHOLDS) | {outlier_threshold}))
    rng = np.random.default_rng(seed)

    flagged_by_thr: dict[float, int] = {t: 0 for t in thresholds}
    n_scored = 0
    perms_used: list[int] = []

    for q in valid:
        logps = _permutation_logprobs(model, q, max_permutations, rng)
        if len(logps) < _MIN_PERMS_FOR_IF:
            # 选项太少（如 2 选项只有 2 排列），IsolationForest 无意义，跳过该题
            continue
        n_scored += 1
        perms_used.append(len(logps))

        X = np.asarray(logps, dtype=np.float64).reshape(-1, 1)
        clf = IsolationForest(
            n_estimators=100, contamination="auto", random_state=seed
        )
        clf.fit(X)
        scores = clf.decision_function(X)
        max_idx = int(np.argmax(logps))
        max_score = float(scores[max_idx])
        for t in thresholds:
            if max_score < t:
                flagged_by_thr[t] += 1

    if n_scored == 0:
        return DetectionResult(
            method="perm_option", stage=stage, benchmark=benchmark.name,
            signal=None, verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            error=(
                f"No question had >= {_MIN_PERMS_FOR_IF} permutations "
                "(need >= 3 options for IsolationForest outlier test)."
            ),
            evidence={"n_valid": len(valid), "n_scored": 0},
        )

    frac_by_thr = {t: flagged_by_thr[t] / n_scored for t in thresholds}
    signal = frac_by_thr[outlier_threshold]

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
            "n_questions": n_scored,
            "n_valid": len(valid),
            "max_permutations": max_permutations,
            "mean_perms_per_q": float(np.mean(perms_used)),
            "primary_threshold": outlier_threshold,
            "leak_fraction": signal,
            "leak_fraction_by_threshold": {str(t): frac_by_thr[t] for t in thresholds},
            "leak_score": signal,  # 别名，对应论文「泄漏率」术语
            "alpha_dirty": _ALPHA_DIRTY,
            "alpha_suspect": _ALPHA_SUSPECT,
        },
    )


def _permutation_logprobs(
    model: ModelInterface,
    q: BenchmarkQuestion,
    max_permutations: int,
    rng: np.random.Generator,
) -> list[float]:
    """算某题在各选项排列下、选项块的序列 log-prob 之和。

    与论文 inference_logprobs.py 一致：score = Σ_token logp(选项块 | 题面)，
    选项块 = "A:opt\\nB:opt\\n..."，不按长度归一。
    """
    choices = q.choices
    assert choices is not None
    n = len(choices)
    total = math.factorial(n)

    if total <= max_permutations:
        perms: list[tuple[int, ...]] = list(iter_permutations(range(n)))
    else:
        # 随机采样不重复排列，恒含原序（index 0）
        identity = tuple(range(n))
        seen = {identity}
        perms = [identity]
        while len(perms) < max_permutations:
            p = tuple(int(x) for x in rng.permutation(n))
            if p not in seen:
                seen.add(p)
                perms.append(p)

    stem = f"{q.prompt}:\n"
    blocks = [
        "\n".join(f"{_LETTERS[i]}:{choices[perm[i]]}" for i in range(n))
        for perm in perms
    ]
    # batch 前向：一题所有排列一次算完（大模型关键吞吐优化）。后端未覆写时
    # seq_logprob_sums 默认逐条回退，数值等价。
    return [float(x) for x in model.seq_logprob_sums(stem, blocks)]
