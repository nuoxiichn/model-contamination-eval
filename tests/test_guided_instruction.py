"""Guided Instruction 单测。

mock model 控制 generate 输出，验证：
- 前置不满足（无 GENERATE / 样本不足 / 切分失败）→ prerequisites_met=False
- guided 复刻原文 / general 乱说 → p_value < 0.01 → DIRTY
- guided 与 general 同样乱说 → p_value 较大 → CLEAN
- ROUGE-L 与 LCS 边界
"""

from __future__ import annotations

import numpy as np
import pytest

from model_contamination.models.base import Capability, ModelInterface
from model_contamination.shared.guided_instruction import (
    _general_prompt,
    _guided_prompt,
    _lcs_length,
    _single_sided_paired_t_test,
    _split_words,
    guided_instruction_test,
    rouge_l_f1,
)
from model_contamination.types import BenchmarkQuestion, BenchmarkSpec, Verdict


def _spec(name: str = "gsm8k") -> BenchmarkSpec:
    return BenchmarkSpec(
        name=name, family="t", format="math_cot", language="en",
        variants=[], applicable_methods=["guided"],
        trustworthiness_default=Verdict.SUSPECT,
        data_source="hf", data_id="x/y",
    )


def _q(i: int) -> BenchmarkQuestion:
    return BenchmarkQuestion(
        id=f"q-{i}", benchmark="gsm8k", format="math_cot",
        prompt=f"sample number {i} the quick brown fox jumps over the lazy dog answer",
        answer="42",
    )


# ----------------------------- mock backends ----------------------------- #


class _NoGenerateModel(ModelInterface):
    name = "logprobs-only"
    stage_tag = "base"

    def supports(self, cap: Capability) -> bool:
        return cap == Capability.LOGPROBS

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        raise AssertionError("should not be called")


class _GenericGenerateModel(ModelInterface):
    """generate 永远输出固定字符串，guided 与 general 表现一致 → ΔROUGE ≈ 0。"""

    name = "generic"
    stage_tag = "base"

    def supports(self, cap: Capability) -> bool:
        return cap == Capability.GENERATE

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        return "i have no idea what comes next here random words"


class _RecallOriginalModel(ModelInterface):
    """guided prompt 含 benchmark 名 → 返回真实后缀；general prompt → 乱写。

    模拟"模型见过 benchmark：被名字唤醒后能复刻原文"。
    """

    name = "recall"
    stage_tag = "base"

    def __init__(self, questions: list[BenchmarkQuestion], benchmark_name: str) -> None:
        # 把 (prefix → reference) 建索引；guided prompt 含 prefix 字串就能回忆
        self._lookup = {}
        for q in questions:
            full = q.prompt
            words = full.split()
            cut = len(words) // 2
            prefix = " ".join(words[:cut])
            reference = " ".join(words[cut:])
            self._lookup[prefix] = reference
        self._benchmark_name = benchmark_name

    def supports(self, cap: Capability) -> bool:
        return cap == Capability.GENERATE

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        is_guided = self._benchmark_name in prompt
        for prefix, ref in self._lookup.items():
            if prefix in prompt:
                if is_guided:
                    return ref
                # general：输出与 ref 完全不同的乱码
                return "totally unrelated tokens that share nothing with reference"
        return ""


# ----------------------------- prerequisites ----------------------------- #


def test_returns_inconclusive_without_generate():
    qs = [_q(i) for i in range(40)]
    r = guided_instruction_test(_NoGenerateModel(), _spec(), qs)
    assert not r.prerequisites_met
    assert r.verdict_hint == Verdict.INCONCLUSIVE
    assert r.signal is None
    assert "GENERATE" in (r.error or "")


def test_returns_inconclusive_when_too_few_samples():
    qs = [_q(i) for i in range(5)]
    r = guided_instruction_test(_GenericGenerateModel(), _spec(), qs, min_samples=30)
    assert not r.prerequisites_met


def test_returns_inconclusive_on_bad_prefix_ratio():
    qs = [_q(i) for i in range(40)]
    r = guided_instruction_test(_GenericGenerateModel(), _spec(), qs, prefix_ratio=0.0)
    assert not r.prerequisites_met


# ----------------------------- behavior ----------------------------- #


def test_recall_model_yields_small_pvalue_and_dirty():
    """guided 复刻原文 → ROUGE 大；general 乱写 → ROUGE 小 → ΔROUGE > 0 → p << 0.05。"""
    qs = [_q(i) for i in range(40)]
    spec = _spec("gsm8k")
    model = _RecallOriginalModel(qs, benchmark_name=spec.name)
    r = guided_instruction_test(model, spec, qs, min_samples=30)
    assert r.prerequisites_met
    assert r.signal is not None
    assert r.signal < 0.01
    assert r.verdict_hint == Verdict.DIRTY
    assert r.evidence["mean_rouge_guided"] > r.evidence["mean_rouge_general"]
    assert r.evidence["mean_delta"] > 0


def test_generic_model_yields_clean():
    """guided/general 都给同一句 → ΔROUGE = 0 → p_value 远大于 alpha_suspect → CLEAN。"""
    qs = [_q(i) for i in range(40)]
    r = guided_instruction_test(_GenericGenerateModel(), _spec(), qs, min_samples=30)
    assert r.prerequisites_met
    assert r.signal is not None
    # ΔROUGE 恒等于 0 → std=0 → 我们约定返回 1.0（mean<=0 时）
    assert r.signal >= 0.5
    assert r.verdict_hint == Verdict.CLEAN
    assert r.evidence["mean_delta"] == pytest.approx(0.0)


# ----------------------------- prompts ----------------------------- #


def test_guided_prompt_contains_benchmark_name():
    spec = _spec("my-bench")
    p = _guided_prompt(spec, "test", "first half")
    assert "my-bench" in p
    assert "first half" in p
    assert "test" in p.lower()


def test_general_prompt_omits_benchmark_name():
    p = _general_prompt("first half")
    assert "first half" in p
    # 一般化 prompt 不应含任何 benchmark 名（这里用 'gsm8k' 做反向检查）
    assert "gsm8k" not in p.lower()


# ----------------------------- ROUGE-L / LCS ----------------------------- #


def test_lcs_basic():
    assert _lcs_length(["a", "b", "c"], ["a", "b", "c"]) == 3
    assert _lcs_length(["a", "b", "c", "d"], ["b", "d", "a"]) == 2  # 'b','d' 子序列
    assert _lcs_length([], ["a"]) == 0
    assert _lcs_length(["a"], []) == 0


def test_rouge_identical_is_one():
    assert rouge_l_f1("a b c d", "a b c d") == 1.0


def test_rouge_disjoint_is_zero():
    assert rouge_l_f1("a b c", "x y z") == 0.0


def test_rouge_partial_match():
    # hyp="a b c x"(4), ref="a b c"(3), LCS=3 → p=3/4, r=1, F1=6/7
    assert rouge_l_f1("a b c x", "a b c") == pytest.approx(6 / 7)


def test_rouge_handles_empty():
    assert rouge_l_f1("", "a b") == 0.0
    assert rouge_l_f1("a b", "") == 0.0


# ----------------------------- splitting ----------------------------- #


def test_split_words_keeps_one_token_each_side():
    p, s = _split_words("only two", 0.5)
    # len=2 < 4 → 直接返回空，避免噪声
    assert (p, s) == ("", "")


def test_split_words_preserves_words():
    p, s = _split_words("a b c d e f g h", 0.5)
    assert p == "a b c d"
    assert s == "e f g h"


# ----------------------------- t-test ----------------------------- #


def test_paired_t_test_zero_mean():
    p = _single_sided_paired_t_test(np.array([0.1, -0.2, 0.05, -0.05]))
    assert 0.3 < p < 0.7


def test_paired_t_test_positive_mean():
    p = _single_sided_paired_t_test(np.full(20, 0.3))
    # std≈0 + mean>0 → 极强 H1 信号；scipy 可能返回 1e-300 级别而非显式 0，接受。
    assert p < 1e-6


def test_paired_t_test_negative_constant():
    p = _single_sided_paired_t_test(np.full(20, -0.3))
    assert p > 1.0 - 1e-6
