"""option_permutation 单测（Algorithm 2：序列 logprob + IsolationForest 离群）。

Mock model 直接返回「选项块」的序列 logprob（单元素数组，方法只用其和）：
- 干净模型：各排列 logprob 恒定 → 无离群 → leak_fraction ≈ 0
- 污染模型：只有「原序（选项内容升序 opt0/opt1/…）」logprob 异常高 → 该排列成离群点
            → 大量题判泄漏 → leak_fraction 高
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from model_contamination.models.base import Capability, ModelInterface
from model_contamination.stage_base.option_permutation import option_permutation_test
from model_contamination.types import BenchmarkQuestion, BenchmarkSpec, Verdict

_OPT_RE = re.compile(r"opt(\d+)")


def _spec(name: str = "mmlu-pro", fmt: str = "multiple_choice") -> BenchmarkSpec:
    return BenchmarkSpec(
        name=name, family="t", format=fmt, language="en",
        variants=[], applicable_methods=["perm_option"],
        trustworthiness_default=Verdict.SUSPECT,
        data_source="hf", data_id="x/y",
    )


def _mcq(i: int, n_opt: int = 4) -> BenchmarkQuestion:
    """选项内容形如 q{i}_opt{j}，原序为 j 升序（opt0,opt1,...）。"""
    return BenchmarkQuestion(
        id=f"q-{i}", benchmark="mmlu-pro", format="multiple_choice",
        prompt=f"[QID={i}] pick the right one",
        answer="A",
        choices=[f"q{i}_opt{j}" for j in range(n_opt)],
        answer_index=0,
    )


def _parsed_order(completion: str) -> list[int]:
    """从选项块 'A:qI_opt0\\nB:qI_opt1\\n...' 解析各显示位置的 opt 序号。"""
    return [int(m) for m in _OPT_RE.findall(completion)]


# ----------------------------- mock models ----------------------------- #


class _NoLogprobModel(ModelInterface):
    name = "blackbox"
    stage_tag = "sft"

    def supports(self, cap: Capability) -> bool:
        return cap == Capability.GENERATE

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        return ""


class _UniformModel(ModelInterface):
    """干净模型：选项块 logprob 与排列无关（恒定）→ 无离群点。"""

    name = "uniform"
    stage_tag = "sft"

    def supports(self, cap: Capability) -> bool:
        return cap in {Capability.LOGPROBS, Capability.GENERATE}

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        return ""

    def logprobs(self, prompt: str, completion: str) -> np.ndarray:
        return np.array([-10.0], dtype=np.float64)


class _MemorizedOrderModel(ModelInterface):
    """污染模型：只有原序（opt 升序）的选项块 logprob 异常高，其余低。

    模拟「训练时见过 (Q, 原始顺序选项) 文本」→ 该顺序序列概率被记住抬高。
    """

    name = "memorized"
    stage_tag = "sft"

    def supports(self, cap: Capability) -> bool:
        return cap in {Capability.LOGPROBS, Capability.GENERATE}

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        return ""

    def logprobs(self, prompt: str, completion: str) -> np.ndarray:
        order = _parsed_order(completion)
        if order == sorted(order):        # 升序 = 记住的原序
            return np.array([-1.0], dtype=np.float64)
        return np.array([-15.0], dtype=np.float64)


# ----------------------------- prerequisites ----------------------------- #


def test_rejects_non_mc_benchmark():
    qs = [_mcq(i) for i in range(60)]
    with pytest.raises(ValueError, match="multiple_choice"):
        option_permutation_test(_MemorizedOrderModel(), _spec(name="gsm8k", fmt="math_cot"), qs)


def test_returns_inconclusive_without_logprobs():
    qs = [_mcq(i) for i in range(60)]
    r = option_permutation_test(_NoLogprobModel(), _spec(), qs, min_samples=50)
    assert not r.prerequisites_met
    assert r.verdict_hint == Verdict.INCONCLUSIVE
    assert r.signal is None
    assert "logprobs" in (r.error or "")


def test_returns_inconclusive_when_too_few_samples():
    qs = [_mcq(i) for i in range(10)]
    r = option_permutation_test(_MemorizedOrderModel(), _spec(), qs, min_samples=50)
    assert not r.prerequisites_met
    assert r.signal is None
    assert "valid MC questions" in (r.error or "")


def test_filters_questions_without_choices():
    """缺 choices 或选项数 < 2 的题不计入有效样本。"""
    good = [_mcq(i) for i in range(40)]
    bad = [
        BenchmarkQuestion(
            id=f"bad-{i}", benchmark="x", format="multiple_choice",
            prompt="...", answer="A", choices=None, answer_index=None,
        )
        for i in range(40)
    ]
    r = option_permutation_test(_MemorizedOrderModel(), _spec(), good + bad, min_samples=50)
    assert not r.prerequisites_met
    assert r.evidence["n_valid"] == 40
    assert r.evidence["n_total"] == 80


# ----------------------------- behavior ----------------------------- #


def test_clean_model_leak_fraction_near_zero():
    """均匀 logprob → 无离群 → leak_fraction ≈ 0 → CLEAN。"""
    qs = [_mcq(i, n_opt=4) for i in range(60)]
    r = option_permutation_test(_UniformModel(), _spec(), qs, min_samples=50, seed=7)
    assert r.prerequisites_met
    assert r.signal is not None
    assert r.signal < 0.05
    assert r.verdict_hint == Verdict.CLEAN


def test_memorized_model_leak_fraction_high():
    """原序 logprob 离群 → 大量题判泄漏 → leak_fraction 高 → DIRTY。"""
    qs = [_mcq(i, n_opt=4) for i in range(60)]
    r = option_permutation_test(_MemorizedOrderModel(), _spec(), qs, min_samples=50, seed=7)
    assert r.prerequisites_met
    assert r.signal is not None
    assert r.signal > 0.8
    assert r.verdict_hint == Verdict.DIRTY


def test_clean_vs_memorized_separation():
    qs = [_mcq(i, n_opt=4) for i in range(60)]
    clean = option_permutation_test(_UniformModel(), _spec(), qs, min_samples=50, seed=7)
    dirty = option_permutation_test(_MemorizedOrderModel(), _spec(), qs, min_samples=50, seed=7)
    assert dirty.signal > clean.signal + 0.5


def test_samples_permutations_when_factorial_too_large():
    """选项数多（n! > max_permutations）时随机采样，恒含原序，仍能检出记忆。"""
    qs = [_mcq(i, n_opt=6) for i in range(60)]   # 6! = 720 > 24
    r = option_permutation_test(
        _MemorizedOrderModel(), _spec(), qs,
        max_permutations=24, min_samples=50, seed=7,
    )
    assert r.prerequisites_met
    assert r.evidence["mean_perms_per_q"] == 24
    assert r.signal > 0.8   # 原序恒被采样 → 仍离群


def test_evidence_contains_required_fields():
    qs = [_mcq(i, n_opt=4) for i in range(60)]
    r = option_permutation_test(_MemorizedOrderModel(), _spec(), qs, min_samples=50, seed=7)
    ev = r.evidence
    for k in (
        "n_questions", "n_valid", "max_permutations", "mean_perms_per_q",
        "primary_threshold", "leak_fraction", "leak_fraction_by_threshold",
        "leak_score", "alpha_dirty", "alpha_suspect",
    ):
        assert k in ev, f"evidence missing {k}"
    assert ev["n_questions"] == 60
    assert ev["leak_score"] == r.signal
    assert ev["leak_fraction"] == r.signal
    # 三档阈值都在 by_threshold 里
    for t in ("-0.2", "-0.17", "-0.15"):
        assert t in ev["leak_fraction_by_threshold"]


def test_custom_outlier_threshold_recorded():
    qs = [_mcq(i, n_opt=4) for i in range(60)]
    r = option_permutation_test(
        _MemorizedOrderModel(), _spec(), qs, min_samples=50, seed=7,
        outlier_threshold=-0.15,
    )
    assert r.evidence["primary_threshold"] == -0.15
    assert "-0.15" in r.evidence["leak_fraction_by_threshold"]


# ----------------------------- signal direction ----------------------------- #


def test_signal_alignment_with_diff_matrix_direction():
    """信号方向 higher_is_dirtier：泄漏率越高越可疑。"""
    from model_contamination.attribution.diff_matrix import signal_direction_for
    assert signal_direction_for("perm_option") == "higher_is_dirtier"
