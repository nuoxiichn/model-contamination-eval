"""Min-K%++ 单测。

不依赖真模型；mock 模型直接提供 token_logprob_stats 字典，验证：
- 前置不满足时返回 prerequisites_met=False
- 样本不足返回 INCONCLUSIVE
- 无 control_questions → mean-only 模式 + INCONCLUSIVE（CLAUDE.md 红线）
- "记忆 target、不记忆 control" → AUC > 0.7 → DIRTY
- 完全随机 mock → AUC ≈ 0.5 → CLEAN
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from model_contamination.models.base import Capability, ModelInterface
from model_contamination.stage_base.min_k_plus_plus import (
    _auc_target_vs_control,
    _mink_pp_sample_score,
    mink_plus_plus,
)
from model_contamination.types import BenchmarkQuestion, BenchmarkSpec, Verdict


def _spec(name: str = "gsm8k") -> BenchmarkSpec:
    return BenchmarkSpec(
        name=name, family="t", format="math_cot", language="en",
        variants=[], applicable_methods=["mink_plus_plus"],
        trustworthiness_default=Verdict.SUSPECT,
        data_source="hf", data_id="x/y",
    )


def _q(i: int, benchmark: str = "gsm8k") -> BenchmarkQuestion:
    return BenchmarkQuestion(
        id=f"{benchmark}-{i}", benchmark=benchmark, format="math_cot",
        prompt=f"What is {i}+{i}?", answer=str(2 * i),
    )


# ----------------------------- mock backends ----------------------------- #


class _NoStatsModel(ModelInterface):
    name = "blackbox"
    stage_tag = "base"

    def supports(self, cap: Capability) -> bool:
        return cap == Capability.GENERATE

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        return ""


class _SeededStatsModel(ModelInterface):
    """对 (prompt, completion) 的稳定哈希 → 一组 token 级 (logp, μ, σ)。

    benchmark 名带 'seen' 时把 chosen_logp 整体抬高（模拟"记住"）。
    """

    name = "seeded"
    stage_tag = "base"

    def supports(self, cap: Capability) -> bool:
        return cap in {Capability.LOGPROBS, Capability.TOKEN_DIST_STATS, Capability.GENERATE}

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        return ""

    def token_logprob_stats(self, prompt: str, completion: str):
        digest = hashlib.sha256((prompt + "|" + completion).encode()).digest()[:4]
        rng = np.random.default_rng(int.from_bytes(digest, "big"))
        n = max(3, len(completion))
        # μ 是 log p 在分布上的期望 → 应 ≤ 0；σ 取正
        mu = rng.uniform(-5.0, -2.0, size=n)
        sigma = rng.uniform(0.5, 2.0, size=n)
        # chosen_logp ~ μ + ε（ε 决定 normalized 的分布）
        eps = rng.normal(0.0, 1.0, size=n)
        chosen = mu + sigma * eps
        # 模拟"见过"：让 chosen 比典型 token 更接近 μ（即 ε 整体偏正 → normalized 偏大）
        if "seen" in completion or "seen" in prompt:
            chosen = mu + sigma * (eps + 1.5)
        return {"chosen_logp": chosen, "mu": mu, "sigma": sigma}


# ----------------------------- prerequisites ----------------------------- #


def test_returns_inconclusive_without_stats():
    qs = [_q(i) for i in range(40)]
    r = mink_plus_plus(_NoStatsModel(), _spec(), qs)
    assert not r.prerequisites_met
    assert r.verdict_hint == Verdict.INCONCLUSIVE
    assert r.signal is None
    assert "TOKEN_DIST_STATS" in (r.error or "")


def test_returns_inconclusive_when_too_few_samples():
    qs = [_q(i) for i in range(5)]
    r = mink_plus_plus(_SeededStatsModel(), _spec(), qs, min_samples=30)
    assert not r.prerequisites_met
    assert r.verdict_hint == Verdict.INCONCLUSIVE


def test_returns_inconclusive_on_bad_k_ratio():
    qs = [_q(i) for i in range(40)]
    r = mink_plus_plus(_SeededStatsModel(), _spec(), qs, k_ratio=1.5)
    assert not r.prerequisites_met
    assert "k_ratio" in (r.error or "")


# ----------------------------- behavior ----------------------------- #


def test_no_control_returns_mean_only_and_inconclusive():
    qs = [_q(i) for i in range(40)]
    r = mink_plus_plus(_SeededStatsModel(), _spec(), qs, min_samples=30)
    assert r.prerequisites_met
    assert r.signal is not None
    # 红线：无 control → 不能定性，无论 signal 多大
    assert r.verdict_hint == Verdict.INCONCLUSIVE
    assert r.evidence["mode"] == "mean_only"
    assert "n_target" in r.evidence


def test_memorized_target_beats_control_auc():
    """target 带 'seen' 标记被"记忆" → chosen_logp 偏高 → normalized 偏大 → AUC 大。"""
    target = [
        BenchmarkQuestion(
            id=f"seen-{i}", benchmark="seen-set", format="math_cot",
            prompt="Q seen-Q", answer=f"seen {2 * i}",
        )
        for i in range(60)
    ]
    control = [
        BenchmarkQuestion(
            id=f"c-{i}", benchmark="clean", format="math_cot",
            prompt=f"What is {i}*{i}?", answer=str(i * i),
        )
        for i in range(60)
    ]
    r = mink_plus_plus(
        _SeededStatsModel(), _spec("seen-set"), target,
        control_questions=control, min_samples=30,
    )
    assert r.prerequisites_met
    assert r.signal is not None
    assert r.evidence["mode"] == "auc"
    assert r.signal > 0.70
    assert r.verdict_hint == Verdict.DIRTY
    assert r.evidence["delta_mean"] > 0


def test_random_target_vs_control_yields_mid_auc():
    """target / control 都不带 'seen'，分布一致 → AUC 应在 ~0.5 附近 → CLEAN。"""
    target = [_q(i, benchmark="bench-a") for i in range(80)]
    control = [_q(i + 1000, benchmark="bench-b") for i in range(80)]
    r = mink_plus_plus(
        _SeededStatsModel(), _spec("bench-a"), target,
        control_questions=control, min_samples=30,
    )
    assert r.signal is not None
    # 允许 0.5 附近合理抖动（mock 用 prompt hash，两组结构差异会留点漂移）
    assert 0.30 < r.signal < 0.70
    # 在 SUSPECT 阈值（0.60）以下应判 CLEAN；漂移到 SUSPECT 也接受，关键是没到 DIRTY
    assert r.verdict_hint in {Verdict.CLEAN, Verdict.SUSPECT}


# ----------------------------- helpers ----------------------------- #


def test_auc_separating_distributions_is_one():
    auc = _auc_target_vs_control(np.array([1.0, 2.0, 3.0]), np.array([-1.0, -2.0]))
    assert auc == 1.0


def test_auc_identical_distributions_half():
    arr = np.array([1.0, 2.0, 3.0])
    assert _auc_target_vs_control(arr, arr) == pytest.approx(0.5)


def test_sample_score_picks_bottom_k():
    """直接构造 stats：chosen=μ+σ·z，z=[3, -2, 1, -1]，k=0.5 → 取最低 2 个 normalized → 均值=-1.5。"""
    class _M(ModelInterface):
        name = "m"
        stage_tag = "base"
        def supports(self, cap):
            return True
        def generate(self, *a, **k):
            return ""
        def token_logprob_stats(self, prompt, completion):
            return {
                "chosen_logp": np.array([3.0, -2.0, 1.0, -1.0]),
                "mu": np.zeros(4),
                "sigma": np.ones(4),
            }
    score = _mink_pp_sample_score(_M(), "p", "c", k_ratio=0.5)
    assert score == pytest.approx(-1.5)


def test_sample_score_handles_zero_sigma():
    """σ=0 的位置应被剔除，不污染均值。"""
    class _M(ModelInterface):
        name = "m"
        stage_tag = "base"
        def supports(self, cap):
            return True
        def generate(self, *a, **k):
            return ""
        def token_logprob_stats(self, prompt, completion):
            return {
                "chosen_logp": np.array([0.0, -1.0, -2.0]),
                "mu": np.array([0.0, 0.0, 0.0]),
                "sigma": np.array([0.0, 1.0, 1.0]),  # 首位被剔除
            }
    score = _mink_pp_sample_score(_M(), "p", "c", k_ratio=0.5)
    assert score == pytest.approx(-2.0)
