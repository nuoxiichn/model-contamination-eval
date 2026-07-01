"""SPV-MIA 单测。

不依赖真模型；mock 模型按 completion 文本内容返回 logprobs，验证：
- 前置不满足（target / reference 无 LOGPROBS）
- 参数非法（mask_ratio 越界 / n_neighbors=0）
- paraphraser 未实装分支
- 样本不足 → INCONCLUSIVE
- 无 control → mean_only 模式 + INCONCLUSIVE
- member（target 记得 + reference 不记得）vs 完全 unseen control → AUC > 0.7 DIRTY
- 完全随机两边 → AUC ≈ 0.5 CLEAN
- MC 题型（短 completion）→ 全 NaN → INCONCLUSIVE
- paraphrase 确定性（同 seed 同输入 → 同输出）
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from model_contamination.models.base import Capability, ModelInterface
from model_contamination.stage_sft.spv_mia import (
    _DEFAULT_WORD_POOL,
    _paraphrase,
    spv_mia,
)
from model_contamination.types import BenchmarkQuestion, BenchmarkSpec, Verdict


# ----------------------------- helpers ----------------------------- #


def _spec(name: str = "gsm8k", fmt: str = "math_cot") -> BenchmarkSpec:
    return BenchmarkSpec(
        name=name, family="t", format=fmt, language="en",
        variants=[], applicable_methods=["spv_mia"],
        trustworthiness_default=Verdict.SUSPECT,
        data_source="hf", data_id="x/y",
    )


def _q(i: int, benchmark: str = "gsm8k", answer: str | None = None) -> BenchmarkQuestion:
    # 默认 answer 故意写长（>= 4 词）以满足 _MIN_COMPLETION_WORDS
    ans = answer if answer is not None else f"the answer is {2 * i} indeed"
    return BenchmarkQuestion(
        id=f"{benchmark}-{i}", benchmark=benchmark, format="math_cot",
        prompt=f"What is {i}+{i}?", answer=ans,
    )


# ----------------------------- mock backends ----------------------------- #


class _NoLogprobsModel(ModelInterface):
    name = "blackbox"
    stage_tag = "sft"

    def supports(self, cap: Capability) -> bool:
        return cap == Capability.GENERATE

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        return ""


class _HashLogprobsModel(ModelInterface):
    """logprobs 由 (prompt, completion) 的稳定哈希驱动，每 token 独立 logp。

    member_set 中的 completion 整体抬高 logp（模拟"记住了"）。
    paraphrase 改了 token → 哈希变 → 退回 baseline。
    """

    def __init__(self, name: str, stage: str, member_set: set[str] | None = None, memorize_boost: float = 3.0):
        self.name = name
        self.stage_tag = stage
        self._member_set = member_set or set()
        self._memorize_boost = memorize_boost

    def supports(self, cap: Capability) -> bool:
        return cap in {Capability.LOGPROBS, Capability.GENERATE}

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        return ""

    def logprobs(self, prompt: str, completion: str) -> np.ndarray:
        digest = hashlib.sha256((prompt + "|" + completion).encode()).digest()[:4]
        rng = np.random.default_rng(int.from_bytes(digest, "big"))
        # token 数取 completion 词数（粗略代理）
        n = max(1, len(completion.split()))
        # baseline log p 约 [-5, -1]（自然对数下"较低概率"）
        lp = rng.uniform(-5.0, -1.0, size=n)
        if completion in self._member_set:
            lp = lp + self._memorize_boost
        return lp


# ----------------------------- prerequisites ----------------------------- #


def test_target_no_logprobs_returns_inconclusive():
    target = _NoLogprobsModel()
    ref = _HashLogprobsModel("ref", "base")
    r = spv_mia(target, ref, _spec(), [_q(i) for i in range(30)])
    assert r.prerequisites_met is False
    assert r.verdict_hint == Verdict.INCONCLUSIVE
    assert r.signal is None
    assert "target" in (r.error or "").lower() or "LOGPROBS" in (r.error or "")


def test_reference_no_logprobs_returns_inconclusive():
    target = _HashLogprobsModel("target", "sft")
    ref = _NoLogprobsModel()
    r = spv_mia(target, ref, _spec(), [_q(i) for i in range(30)])
    assert r.prerequisites_met is False
    assert r.verdict_hint == Verdict.INCONCLUSIVE


# ----------------------------- param validation ----------------------------- #


@pytest.mark.parametrize("mask_ratio", [-0.1, 0.0, 1.0, 1.5])
def test_invalid_mask_ratio(mask_ratio):
    target = _HashLogprobsModel("t", "sft")
    ref = _HashLogprobsModel("r", "base")
    r = spv_mia(target, ref, _spec(), [_q(i) for i in range(30)], mask_ratio=mask_ratio)
    assert r.prerequisites_met is False
    assert "mask_ratio" in (r.error or "")


@pytest.mark.parametrize("n", [0, -1])
def test_invalid_n_neighbors(n):
    target = _HashLogprobsModel("t", "sft")
    ref = _HashLogprobsModel("r", "base")
    r = spv_mia(target, ref, _spec(), [_q(i) for i in range(30)], n_neighbors=n)
    assert r.prerequisites_met is False
    assert "n_neighbors" in (r.error or "")


def test_too_few_samples():
    target = _HashLogprobsModel("t", "sft")
    ref = _HashLogprobsModel("r", "base")
    r = spv_mia(target, ref, _spec(), [_q(i) for i in range(5)], min_samples=30)
    assert r.prerequisites_met is False
    assert "samples" in (r.error or "").lower()


def test_unimplemented_paraphraser():
    target = _HashLogprobsModel("t", "sft")
    ref = _HashLogprobsModel("r", "base")
    with pytest.raises(NotImplementedError):
        # paraphraser 是 Literal，但运行时不会拦；只在调用 _paraphrase 时抛
        # 直接走 _paraphrase 验证
        _paraphrase("foo bar baz qux", 0.5, "t5", np.random.default_rng(0))  # type: ignore[arg-type]


# ----------------------------- mean_only mode ----------------------------- #


def test_no_control_returns_mean_only():
    target = _HashLogprobsModel("t", "sft")
    ref = _HashLogprobsModel("r", "base")
    questions = [_q(i) for i in range(40)]
    r = spv_mia(target, ref, _spec(), questions, control_questions=None, n_neighbors=3)
    assert r.prerequisites_met is True
    assert r.verdict_hint == Verdict.INCONCLUSIVE
    assert r.signal is not None
    assert r.evidence["mode"] == "mean_only"
    assert "target_mean_delta_pv" in r.evidence
    assert r.evidence["n_target"] > 0


# ----------------------------- AUC: member separation ----------------------------- #


def test_member_target_vs_unseen_control_yields_high_auc():
    """target 记住所有 questions 的 completion；reference 谁都不记得。

    paraphrase 后 completion 变 → target 退回 baseline logp → pv_target 显著负。
    control_questions 谁都没见 → pv_target ≈ 0 ≈ pv_ref。
    AUC(−Δpv_target_set vs −Δpv_control_set) 应远高于 0.5。
    """
    questions = [_q(i) for i in range(60)]
    member_set = {q.answer for q in questions}
    target = _HashLogprobsModel("t", "sft", member_set=member_set, memorize_boost=5.0)
    ref = _HashLogprobsModel("r", "base", member_set=set(), memorize_boost=0.0)

    # control 用不同 question pool（不在 member_set 里）
    controls = [_q(i, answer=f"control answer number {i} here") for i in range(100, 160)]

    r = spv_mia(target, ref, _spec(), questions, control_questions=controls, n_neighbors=5)
    assert r.prerequisites_met is True
    assert r.evidence["mode"] == "auc"
    assert r.signal is not None and r.signal > 0.70
    assert r.verdict_hint == Verdict.DIRTY
    # Sanity：member 的 Δpv 应显著负，control 的应接近 0
    assert r.evidence["target_mean_delta_pv"] < r.evidence["control_mean_delta_pv"]
    assert r.evidence["delta_of_delta_pv"] < 0


def test_random_both_yields_auc_near_half():
    """target 和 reference 都不记得任何 question → Δpv ≈ 0 两边 → AUC ≈ 0.5。"""
    questions = [_q(i) for i in range(60)]
    controls = [_q(i, answer=f"control answer number {i} here") for i in range(100, 160)]
    target = _HashLogprobsModel("t", "sft", member_set=set())
    ref = _HashLogprobsModel("r", "base", member_set=set())
    r = spv_mia(target, ref, _spec(), questions, control_questions=controls, n_neighbors=5)
    assert r.prerequisites_met is True
    assert r.signal is not None
    assert 0.35 < r.signal < 0.65, f"expected AUC ≈ 0.5, got {r.signal}"
    assert r.verdict_hint in {Verdict.CLEAN, Verdict.SUSPECT}


# ----------------------------- MC degrade ----------------------------- #


def test_mc_single_letter_answers_degrade_to_inconclusive():
    """MC 题型 completion 是单字母 → < _MIN_COMPLETION_WORDS → 全 NaN → INCONCLUSIVE。"""
    qs = [
        BenchmarkQuestion(
            id=f"mc-{i}", benchmark="mmlu-pro", format="multiple_choice",
            prompt=f"Q{i}?", answer="B",
            choices=["alpha", "beta", "gamma", "delta"],
            answer_index=1,
        )
        for i in range(40)
    ]
    target = _HashLogprobsModel("t", "sft")
    ref = _HashLogprobsModel("r", "base")
    r = spv_mia(target, ref, _spec(name="mmlu-pro", fmt="multiple_choice"), qs, min_samples=30)
    assert r.prerequisites_met is False
    assert r.verdict_hint == Verdict.INCONCLUSIVE
    assert "finite" in (r.error or "").lower() or "samples" in (r.error or "").lower()
    # 早 return 也要落 evidence 诊断（不是 None），下游打印/CSV 不会 KeyError
    assert r.evidence is not None
    assert r.evidence.get("degraded_reason") == "too_few_finite_scores"
    assert r.evidence.get("n_input") == 40
    assert r.evidence.get("n_target") == 0


# ----------------------------- math_cot: raw['full_answer'] 走 SFT 训过的 CoT ----------------------------- #


def test_math_cot_uses_full_answer_from_raw():
    """GSM8K 类 math_cot：归一化 loader 把 answer 提成最终数字 "3"，完整 CoT 在
    raw['full_answer']。SFT 训的是 CoT+answer 全文，SPV-MIA 必须用 full_answer
    才能在 paraphrase 时撼动 token 序列产生有意义的 Δpv。
    """
    full = "First we compute 2/2=1. So total = 2+1=3 bolts of fabric. #### 3"
    questions = [
        BenchmarkQuestion(
            id=f"g-{i}", benchmark="gsm8k", format="math_cot",
            prompt=f"Trivial problem {i}?", answer="3",  # 1 词
            raw={"full_answer": full},
        )
        for i in range(40)
    ]
    # member_set 用 full_answer 文本（_HashLogprobsModel 按 completion 文本计 boost）
    member_set = {full}
    target = _HashLogprobsModel("t", "sft", member_set=member_set, memorize_boost=5.0)
    ref = _HashLogprobsModel("r", "base", member_set=set())

    r = spv_mia(target, ref, _spec(name="gsm8k", fmt="math_cot"), questions, n_neighbors=3)
    # 全部题都应产生 finite Δpv（不再因 1-word answer 被过滤）
    assert r.prerequisites_met is True
    assert r.evidence["mode"] == "mean_only"
    assert r.evidence["n_target"] == 40
    # member 全集 → mean(Δpv) 应显著 < 0（target 记住 full_answer，paraphrase 后退回 baseline）
    assert r.evidence["target_mean_delta_pv"] < -1.0


def test_math_cot_falls_back_to_answer_when_raw_missing():
    """raw 不含 full_answer 时回退到 q.answer，保留兼容路径。"""
    questions = [
        BenchmarkQuestion(
            id=f"g-{i}", benchmark="gsm8k", format="math_cot",
            prompt=f"Q{i}?",
            answer=f"the long form answer number {i} indeed",  # >= 4 词，可 paraphrase
            raw={},
        )
        for i in range(40)
    ]
    target = _HashLogprobsModel("t", "sft")
    ref = _HashLogprobsModel("r", "base")
    r = spv_mia(target, ref, _spec(name="gsm8k", fmt="math_cot"), questions, n_neighbors=3)
    assert r.prerequisites_met is True
    assert r.evidence["n_target"] == 40


# ----------------------------- paraphrase 确定性 ----------------------------- #


def test_paraphrase_deterministic_with_same_seed():
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    text = "the quick brown fox jumps over the lazy dog tonight"
    out1 = _paraphrase(text, 0.3, "random", rng1)
    out2 = _paraphrase(text, 0.3, "random", rng2)
    assert out1 == out2
    # 确实改了内容
    assert out1 != text
    # 词数保持
    assert len(out1.split()) == len(text.split())


def test_paraphrase_pool_used():
    """随机替换出的词必须在 _DEFAULT_WORD_POOL 内。"""
    rng = np.random.default_rng(0)
    text = "alpha bravo charlie delta echo foxtrot"
    original_words = set(text.split())
    out = _paraphrase(text, 0.5, "random", rng)
    new_words = [w for w in out.split() if w not in original_words]
    pool_set = set(_DEFAULT_WORD_POOL)
    for w in new_words:
        assert w in pool_set, f"replacement word {w!r} not in default pool"


def test_paraphrase_too_short_returns_text_unchanged():
    rng = np.random.default_rng(0)
    text = "two words"
    out = _paraphrase(text, 0.5, "random", rng)
    assert out == text  # 词数 < _MIN_COMPLETION_WORDS
