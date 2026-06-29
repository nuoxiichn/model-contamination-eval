"""Min-K%++（Zhang et al. 2024, arXiv:2404.02936）。

对每个 token t_i 在前缀 x_<i 下的 log-likelihood 做 z-score 归一化，再取最低 K%
归一化 log p 的均值作为样本级 MIA 分数：

    log p_norm(x_i) = (log p(x_i|x_<i) - μ_i) / σ_i
    MIN-K%++(x)   = mean( bottom-K% of {log p_norm(x_i)} )

μ_i / σ_i 是分布 p_i(·|x_<i) 在自身上的 log-likelihood 的均值和标准差。
直觉：训练过的样本的 token log-likelihood 在条件分布中是"局部最大"，
归一化后的 chosen log p 应整体偏高 → 样本级 score 比未见过样本更大。

⚠️ 单独使用的限制（CLAUDE.md 红线）：
- base 阶段单 checkpoint 时 AUC ≈ 0.5（时间偏移控制后，MIMIR 2024）；
- 必须与一个"假定未见过"的 control 集对比，才有 MIA AUC 可言；
- 无 control 集时，本函数只返回平均 score 作为相对信号，verdict 直接 INCONCLUSIVE。

接口契约（与 oren_permutation_test 对齐）：
    输入 model (灰盒，需 TOKEN_DIST_STATS) + spec + 已加载的 questions [+ control_questions]
    输出 DetectionResult(method="mink_plus_plus", signal=AUC 或 mean_score)
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from model_contamination.models.base import Capability, ModelInterface
from model_contamination.types import (
    BenchmarkQuestion,
    BenchmarkSpec,
    DetectionResult,
    Stage,
    Verdict,
)

_AUC_DIRTY = 0.70
_AUC_SUSPECT = 0.60

Estimator = Literal["mean", "trim_mean", "median"]


def mink_plus_plus(
    model: ModelInterface,
    benchmark: BenchmarkSpec,
    questions: list[BenchmarkQuestion],
    *,
    k_ratio: float = 0.2,
    control_questions: list[BenchmarkQuestion] | None = None,
    min_samples: int = 30,
    estimator: Estimator = "trim_mean",
    trim_ratio: float = 0.1,
) -> DetectionResult:
    """Min-K%++ 主入口。

    k_ratio: bottom-K% 的 K（论文用 0.2）。
    control_questions: 同分布、假定未见过的对照（数学条目通常用 MATH-500 / AIME-2025）。
      给了 → 计算二分类 AUC 作 signal；不给 → 只返 estimator(target_scores)（verdict INCONCLUSIVE）。
    min_samples: 主集最小题数；低于则 INCONCLUSIVE。
    estimator: mean_only 模式下的位置估计。重度过拟合 ckpt 上 target_scores 会出
      现极端 outlier（见 2026-06-29 calibration Finding 5：gsm8k_heavy×gsm8k 200 个
      分数里 126 个 < -10，最小 -3825），mean 会被淹没。默认 trim_mean。AUC 模式
      用 rank-based U 统计量，本来就对 outlier robust，不受此参数影响。
    trim_ratio: estimator='trim_mean' 时两端各 trim 比例。默认 0.1（双端各 10%）。
    """
    stage = Stage(model.stage_tag)

    if not model.supports(Capability.TOKEN_DIST_STATS):
        return DetectionResult(
            method="mink_plus_plus", stage=stage, benchmark=benchmark.name,
            signal=None, verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            error=f"{model.name} does not support TOKEN_DIST_STATS; Min-K%++ requires per-token (μ, σ).",
        )

    if not 0.0 < k_ratio <= 1.0:
        return DetectionResult(
            method="mink_plus_plus", stage=stage, benchmark=benchmark.name,
            signal=None, verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            error=f"k_ratio must be in (0, 1], got {k_ratio}.",
        )

    if len(questions) < min_samples:
        return DetectionResult(
            method="mink_plus_plus", stage=stage, benchmark=benchmark.name,
            signal=None, verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            error=f"Need >= {min_samples} samples, got {len(questions)}.",
        )

    target_scores = _score_questions(model, questions, k_ratio)
    target_scores = _filter_finite(target_scores)
    if len(target_scores) < min_samples // 2:
        return DetectionResult(
            method="mink_plus_plus", stage=stage, benchmark=benchmark.name,
            signal=None, verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            error=f"Only {len(target_scores)} finite scores out of {len(questions)} questions.",
        )

    if control_questions is None or len(control_questions) == 0:
        # 无对照 → 不能算 AUC；返回 estimator(target_scores) 作相对信号
        location, raw_mean = _location_with_raw(target_scores, estimator, trim_ratio)
        return DetectionResult(
            method="mink_plus_plus", stage=stage, benchmark=benchmark.name,
            signal=location, verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=True,
            evidence={
                "mode": "mean_only",
                "k_ratio": k_ratio,
                "n_target": int(len(target_scores)),
                "estimator": estimator,
                "trim_ratio": trim_ratio if estimator == "trim_mean" else None,
                "target_mean": location,           # 保持字段名向后兼容；值是 estimator 输出
                "target_mean_raw": raw_mean,       # 原始 np.mean，便于对照 outlier 影响
                "target_std": float(np.std(target_scores, ddof=1)) if len(target_scores) > 1 else 0.0,
                "target_scores": target_scores.tolist(),
                "note": (
                    "No control set; AUC unavailable. base-stage Min-K%++ AUC ≈ 0.5"
                    " without reference (CLAUDE.md red-line)."
                ),
            },
            error=None,
        )

    control_scores = _score_questions(model, control_questions, k_ratio)
    control_scores = _filter_finite(control_scores)
    if len(control_scores) < 2:
        return DetectionResult(
            method="mink_plus_plus", stage=stage, benchmark=benchmark.name,
            signal=None, verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            error=f"Only {len(control_scores)} finite control scores; need >= 2.",
        )

    auc = _auc_target_vs_control(target_scores, control_scores)
    if auc >= _AUC_DIRTY:
        verdict = Verdict.DIRTY
    elif auc >= _AUC_SUSPECT:
        verdict = Verdict.SUSPECT
    else:
        verdict = Verdict.CLEAN

    return DetectionResult(
        method="mink_plus_plus", stage=stage, benchmark=benchmark.name,
        signal=float(auc), verdict_hint=verdict, prerequisites_met=True,
        evidence={
            "mode": "auc",
            "k_ratio": k_ratio,
            "n_target": int(len(target_scores)),
            "n_control": int(len(control_scores)),
            "target_mean": float(np.mean(target_scores)),
            "control_mean": float(np.mean(control_scores)),
            "delta_mean": float(np.mean(target_scores) - np.mean(control_scores)),
            "target_scores": target_scores.tolist(),
            "control_scores": control_scores.tolist(),
        },
        error=None,
    )


# ----------------------------- core scoring ----------------------------- #


def _score_questions(
    model: ModelInterface,
    questions: list[BenchmarkQuestion],
    k_ratio: float,
) -> np.ndarray:
    scores = np.empty(len(questions), dtype=np.float64)
    for i, q in enumerate(questions):
        prompt, completion = _split_prompt_completion(q)
        scores[i] = _mink_pp_sample_score(model, prompt, completion, k_ratio)
    return scores


def _split_prompt_completion(q: BenchmarkQuestion) -> tuple[str, str]:
    """Min-K%++ 测量"整个样本"的 token 序列，不能只看答案部分。

    论文（Zhang et al. 2024）的 statistic 对整段 x = [x_1, ..., x_n] 计算 token-wise
    normalized log p，再取 bottom-K%。只取"Answer:"后几 token 会导致：
    - bottom-K% 几乎没东西（短答案样本 K·n < 1）
    - 长度参差的 benchmark 间不可比（GSM8K "#### 18" vs MATH 长 LaTeX）

    所以：prompt 只放分隔符做 conditioning boundary，completion = 整个 Q+A。
    """
    if q.format == "multiple_choice" and q.choices:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(q.choices))
        text = f"Question: {q.prompt}\n{body}\nAnswer: {q.answer}"
    else:
        text = f"Question: {q.prompt}\nAnswer: {q.answer}"
    return "\n\n", text


def _mink_pp_sample_score(
    model: ModelInterface, prompt: str, completion: str, k_ratio: float
) -> float:
    """单样本 Min-K%++ score。空 completion / σ 全 0 → NaN（上层过滤）。"""
    stats = model.token_logprob_stats(prompt, completion)
    chosen = stats["chosen_logp"]
    mu = stats["mu"]
    sigma = stats["sigma"]
    if chosen.size == 0:
        return float("nan")

    # 避免 σ=0 处除零（理论上单一确定的下一 token 才会出现）
    safe_sigma = np.where(sigma > 1e-8, sigma, np.nan)
    normalized = (chosen - mu) / safe_sigma
    normalized = normalized[np.isfinite(normalized)]
    if normalized.size == 0:
        return float("nan")

    k = max(1, int(np.ceil(k_ratio * normalized.size)))
    bottom = np.partition(normalized, k - 1)[:k]
    return float(np.mean(bottom))


def _filter_finite(arr: np.ndarray) -> np.ndarray:
    return arr[np.isfinite(arr)]


def _location_with_raw(
    arr: np.ndarray, estimator: Estimator, trim_ratio: float
) -> tuple[float, float]:
    """位置估计：返回 (estimator(arr), raw_mean) 两个值。

    raw_mean 是 np.mean(arr)，留在 evidence 里方便定量判断 outlier 影响。
    重度过拟合 ckpt 上 estimator(trim_mean) 与 raw_mean 可能差几十倍。
    """
    raw_mean = float(np.mean(arr))
    if estimator == "mean":
        return raw_mean, raw_mean
    if estimator == "median":
        return float(np.median(arr)), raw_mean
    if estimator == "trim_mean":
        if not 0.0 <= trim_ratio < 0.5:
            raise ValueError(f"trim_ratio must be in [0, 0.5), got {trim_ratio}")
        n = len(arr)
        k = int(np.floor(trim_ratio * n))
        if n - 2 * k <= 0:
            return float(np.median(arr)), raw_mean  # 样本太少，退回 median
        kept = np.sort(arr)[k:n - k]
        return float(np.mean(kept)), raw_mean
    raise ValueError(f"unknown estimator: {estimator!r}")


def _auc_target_vs_control(target: np.ndarray, control: np.ndarray) -> float:
    """二分类 AUC：target 标 1（疑似见过），control 标 0。

    Mann–Whitney U / n_target / n_control，含 0.5 计 tie。
    """
    n_t = len(target)
    n_c = len(control)
    if n_t == 0 or n_c == 0:
        return 0.5
    # 用配对比较算 U，O(n_t · n_c)；样本量 ≤ 几百够用
    wins = 0.0
    for t in target:
        wins += np.sum(t > control) + 0.5 * np.sum(t == control)
    return float(wins / (n_t * n_c))
