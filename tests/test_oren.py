"""Oren 排列检验单测。

不依赖真模型；用 mock model 控制 logprobs 输出，验证：
- 前置不满足时返回 prerequisites_met=False
- 样本不足时返回 INCONCLUSIVE
- "记忆模型"（对原序给高 logp）输出小 p_value
- "无知模型"（与顺序无关）输出大 p_value
- 单侧 t 检验在 mean(diff) < 0 时返回 ≥ 0.5 的 p_value
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from model_contamination.models.base import Capability, ModelInterface
from model_contamination.shared.oren_permutation import (
    _single_sided_t_test,
    oren_permutation_test,
)
from model_contamination.types import BenchmarkQuestion, BenchmarkSpec, Verdict


def _spec(name: str = "gsm8k") -> BenchmarkSpec:
    return BenchmarkSpec(
        name=name, family="t", format="math_cot", language="en",
        variants=[], applicable_methods=["oren"],
        trustworthiness_default=Verdict.SUSPECT,
        data_source="hf", data_id="x/y",
    )


def _q(i: int) -> BenchmarkQuestion:
    return BenchmarkQuestion(
        id=f"q-{i}", benchmark="gsm8k", format="math_cot",
        prompt=f"What is {i}+{i}?", answer=str(2 * i),
    )


# ----------------------------- mock models ----------------------------- #


class _NoLogprobModel(ModelInterface):
    name = "blackbox"
    stage_tag = "base"

    def supports(self, cap: Capability) -> bool:
        return cap == Capability.GENERATE

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        return ""


class _RandomLogpModel(ModelInterface):
    """对每个 completion 返回稳定 hash → Normal 采样的 logp —— 与顺序无关、无记忆。"""

    name = "random"
    stage_tag = "base"

    def supports(self, cap: Capability) -> bool:
        return cap in {Capability.LOGPROBS, Capability.GENERATE}

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        return ""

    def logprobs(self, prompt: str, completion: str) -> np.ndarray:
        digest = hashlib.sha256(completion.encode()).digest()[:4]
        seed = int.from_bytes(digest, "big")
        rng = np.random.default_rng(seed)
        return rng.normal(0.0, 5.0, size=1).astype(np.float64)


class _MemorizedOrderModel(ModelInterface):
    """对相邻二元组 "What is i+i? ... What is (i+1)+(i+1)?" 给 bonus。

    原序里每个 shard 都有 shard_size-1 个这样的二元组；随机置换会破坏大多数。
    模拟"模型记住了 dataset 的全局顺序"。
    """

    name = "memorized"
    stage_tag = "base"

    def supports(self, cap: Capability) -> bool:
        return cap in {Capability.LOGPROBS, Capability.GENERATE}

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        return ""

    def logprobs(self, prompt: str, completion: str) -> np.ndarray:
        bonus = 0.0
        for i in range(300):
            pattern = f"What is {i}+{i}?"
            next_pattern = f"What is {i + 1}+{i + 1}?"
            idx = completion.find(pattern)
            if idx < 0:
                continue
            tail = completion[idx + len(pattern) : idx + len(pattern) + 200]
            if next_pattern in tail:
                bonus += 10.0
        return np.array([-1.0 + bonus], dtype=np.float64)


# ----------------------------- prerequisites ----------------------------- #


def test_returns_inconclusive_without_logprobs():
    qs = [_q(i) for i in range(200)]
    r = oren_permutation_test(_NoLogprobModel(), _spec(), qs)
    assert not r.prerequisites_met
    assert r.verdict_hint == Verdict.INCONCLUSIVE
    assert r.signal is None
    assert "logprobs" in (r.error or "")


def test_returns_inconclusive_when_too_few_samples():
    qs = [_q(i) for i in range(10)]
    r = oren_permutation_test(_RandomLogpModel(), _spec(), qs, min_samples=200)
    assert not r.prerequisites_met
    assert r.verdict_hint == Verdict.INCONCLUSIVE


def test_can_lower_min_samples():
    qs = [_q(i) for i in range(50)]
    r = oren_permutation_test(
        _RandomLogpModel(), _spec(), qs,
        shard_size=10, min_samples=50,
    )
    assert r.prerequisites_met
    assert r.signal is not None


# ----------------------------- behavior ----------------------------- #


def test_random_model_yields_large_pvalue():
    """logp 仅依赖 completion 内容的稳定哈希 → mean(diff) 在 0 附近随机 → p > 0.05。"""
    qs = [_q(i) for i in range(200)]
    r = oren_permutation_test(
        _RandomLogpModel(), _spec(), qs, shard_size=25, min_samples=200, seed=7,
    )
    assert r.prerequisites_met
    assert r.signal is not None
    assert r.signal > 0.05
    assert r.verdict_hint == Verdict.CLEAN


def test_memorized_model_yields_small_pvalue():
    """对原序给特别高 logp → mean(diff) > 0 → p_value 应 << 0.05。"""
    qs = [_q(i) for i in range(200)]
    r = oren_permutation_test(
        _MemorizedOrderModel(), _spec(), qs, shard_size=25, min_samples=200, seed=7,
    )
    assert r.signal is not None
    assert r.signal < 0.05
    assert r.verdict_hint in {Verdict.DIRTY, Verdict.SUSPECT}
    assert r.evidence["mean_diff"] > 0


# ----------------------------- t-test helper ----------------------------- #


def test_single_sided_t_test_zero_mean():
    diffs = np.array([0.1, -0.2, 0.05, -0.05])
    p = _single_sided_t_test(diffs)
    # 接近 0.5
    assert 0.3 < p < 0.7


def test_single_sided_t_test_positive_mean():
    diffs = np.full(20, 1.0)
    p = _single_sided_t_test(diffs)
    assert p < 0.01


def test_single_sided_t_test_negative_mean():
    diffs = np.full(20, -1.0)
    p = _single_sided_t_test(diffs)
    assert p > 0.99


def test_single_sided_t_test_single_value():
    p = _single_sided_t_test(np.array([1.0]))
    assert p == 1.0
