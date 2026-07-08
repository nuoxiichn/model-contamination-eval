"""CoDeC 单测。

不依赖真模型；用 mock model 控制 logprobs，验证：
- 前置不满足（无 logprobs / 样本 < 2）→ prerequisites_met=False
- "seen 模型"（有 context 时 logp 下降）→ signal≈1.0，DIRTY
- "unseen 模型"（有 context 时 logp 上升）→ signal≈0.0，CLEAN
- skip_first_tokens：短样本被跳过并计入 n_skipped
- 同 seed 两次调用结果一致
"""

from __future__ import annotations

import numpy as np

from model_contamination.models.base import Capability, ModelInterface
from model_contamination.shared.codec import codec_detect
from model_contamination.types import BenchmarkQuestion, BenchmarkSpec, Verdict


def _spec(name: str = "pile-arxiv") -> BenchmarkSpec:
    return BenchmarkSpec(
        name=name, family="t", format="open_generation", language="en",
        variants=[], applicable_methods=["codec"],
        trustworthiness_default=Verdict.SUSPECT,
        data_source="hf", data_id="x/y",
    )


def _qs(n: int, tokens_each: int = 20) -> list[BenchmarkQuestion]:
    """每条 prompt 是 tokens_each 个用空格分隔的词（近似 tokens_each 个 token）。"""
    out = []
    for i in range(n):
        words = " ".join(f"w{i}_{t}" for t in range(tokens_each))
        out.append(
            BenchmarkQuestion(
                id=f"q-{i}", benchmark="pile-arxiv", format="open_generation",
                prompt=words, answer="",
            )
        )
    return out


# ----------------------------- mock models ----------------------------- #


class _NoLogprobModel(ModelInterface):
    name = "blackbox"
    stage_tag = "base"

    def supports(self, cap: Capability) -> bool:
        return cap == Capability.GENERATE

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        return ""


class _ContextEffectModel(ModelInterface):
    """按 completion 的空白切分成「token」，每 token 给固定 base logp。

    有 context（prompt 非空）时对每 token 施加 delta：
        delta < 0 → 模拟 seen（加 context 降低置信度）
        delta > 0 → 模拟 unseen（加 context 提高置信度）
    logp 长度 = completion 的「词」数，忠实模拟 logprobs 只返回 completion token。
    """

    name = "ctx"
    stage_tag = "base"

    def __init__(self, delta: float):
        self._delta = delta

    def supports(self, cap: Capability) -> bool:
        return cap in {Capability.LOGPROBS, Capability.GENERATE}

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        return ""

    def logprobs(self, prompt: str, completion: str) -> np.ndarray:
        n = len(completion.split())
        base = np.full(n, -2.0, dtype=np.float64)
        if prompt.strip():  # in-context 调用
            base = base + self._delta
        return base


# ----------------------------- prerequisites ----------------------------- #


def test_inconclusive_without_logprobs():
    r = codec_detect(_NoLogprobModel(), _spec(), _qs(10))
    assert not r.prerequisites_met
    assert r.verdict_hint == Verdict.INCONCLUSIVE
    assert r.signal is None
    assert "logprobs" in (r.error or "")


def test_inconclusive_with_single_sample():
    r = codec_detect(_ContextEffectModel(-1.0), _spec(), _qs(1))
    assert not r.prerequisites_met
    assert r.verdict_hint == Verdict.INCONCLUSIVE


class _NaNModel(ModelInterface):
    """logprobs 返回 NaN，模拟 OPT bf16 finetune 后前向数值崩溃。"""

    name = "nan"
    stage_tag = "base"

    def supports(self, cap: Capability) -> bool:
        return cap in {Capability.LOGPROBS, Capability.GENERATE}

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        return ""

    def logprobs(self, prompt: str, completion: str) -> np.ndarray:
        return np.full(len(completion.split()), np.nan, dtype=np.float64)


def test_nan_logprobs_inconclusive_not_silent_clean():
    """NaN logprobs 必须返回 INCONCLUSIVE，绝不能静默变成 signal=0.0 CLEAN。

    回归：NaN<0 == False → 旧代码 frac_negative=mean(NaN<0)=0.0 假报 CLEAN，
    违反红线「禁止悄悄回退到不可信结果」。现改为 _avg_logp 剔除非有限值。
    """
    r = codec_detect(_NaNModel(), _spec(), _qs(30))
    assert not r.prerequisites_met
    assert r.verdict_hint == Verdict.INCONCLUSIVE
    assert r.signal is None
    assert "non-finite" in (r.error or "")


# ----------------------------- behavior ----------------------------- #


def test_seen_model_high_score_dirty():
    """context 使 logp 一致下降 → 所有 Δ<0 → signal≈1.0，DIRTY。"""
    r = codec_detect(_ContextEffectModel(-1.0), _spec(), _qs(30), seed=7)
    assert r.prerequisites_met
    assert r.signal == 1.0
    assert r.verdict_hint == Verdict.DIRTY
    assert r.evidence["mean_delta"] < 0


def test_unseen_model_low_score_clean():
    """context 使 logp 一致上升 → 所有 Δ>0 → signal≈0.0，CLEAN。"""
    r = codec_detect(_ContextEffectModel(+1.0), _spec(), _qs(30), seed=7)
    assert r.prerequisites_met
    assert r.signal == 0.0
    assert r.verdict_hint == Verdict.CLEAN
    assert r.evidence["mean_delta"] > 0


def test_keep_deltas_absent_by_default_present_on_request():
    """keep_deltas=False（默认）不带 deltas；True 时带每样本 Δ 数组。"""
    r_off = codec_detect(_ContextEffectModel(-1.0), _spec(), _qs(30), seed=7)
    assert "deltas" not in r_off.evidence

    r_on = codec_detect(
        _ContextEffectModel(-1.0), _spec(), _qs(30), seed=7, keep_deltas=True
    )
    deltas = r_on.evidence["deltas"]
    assert len(deltas) == r_on.evidence["n_samples"]
    # signal = Δ<0 比例，应与 deltas 一致
    frac_neg = sum(d < 0 for d in deltas) / len(deltas)
    assert frac_neg == r_on.signal


def test_skip_first_tokens_counts_short_samples():
    """全部样本词数 <= skip_first_tokens → 无有效 Δ → INCONCLUSIVE。"""
    r = codec_detect(
        _ContextEffectModel(-1.0), _spec(), _qs(20, tokens_each=8),
        skip_first_tokens=10,
    )
    assert not r.prerequisites_met
    assert "too short" in (r.error or "")


def test_partial_skip_reports_n_skipped():
    """混合长短样本：短的进 n_skipped，长的进 n_samples。"""
    qs = _qs(10, tokens_each=20) + _qs(5, tokens_each=5)
    # _qs 的 id 会重复，重打 id 以免歧义
    qs = [
        BenchmarkQuestion(id=f"q{i}", benchmark="b", format="open_generation",
                          prompt=q.prompt, answer="")
        for i, q in enumerate(qs)
    ]
    r = codec_detect(_ContextEffectModel(-1.0), _spec(), qs, skip_first_tokens=10)
    assert r.prerequisites_met
    assert r.evidence["n_skipped"] == 5
    assert r.evidence["n_samples"] == 10


def test_seed_reproducible():
    qs = _qs(30)
    r1 = codec_detect(_ContextEffectModel(-0.5), _spec(), qs, seed=123)
    r2 = codec_detect(_ContextEffectModel(-0.5), _spec(), qs, seed=123)
    assert r1.signal == r2.signal
    assert r1.evidence["mean_delta"] == r2.evidence["mean_delta"]


def test_evidence_fields_present():
    r = codec_detect(_ContextEffectModel(-1.0), _spec(), _qs(30))
    for key in (
        "n_samples", "n_skipped", "n_context", "n_seeds",
        "skip_first_tokens", "mean_delta", "frac_negative", "threshold_source",
    ):
        assert key in r.evidence
    assert r.evidence["threshold_source"] == "paper_A.5_uncalibrated"
