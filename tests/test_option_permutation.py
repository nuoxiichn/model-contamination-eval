"""option_permutation 单测。

Mock model 模拟两种行为：
- 干净模型（追内容）：识别正确选项的语义，shuffle 后仍命中正确字母
- 污染模型（追位置）：记住"题面 → 字母 A"，shuffle 后还是输出 A
"""

from __future__ import annotations

import numpy as np
import pytest

from model_contamination.models.base import Capability, ModelInterface
from model_contamination.stage_base.option_permutation import (
    _LETTERS,
    _predict_letter,
    option_permutation_test,
)
from model_contamination.types import BenchmarkQuestion, BenchmarkSpec, Verdict


def _spec(name: str = "mmlu-pro", fmt: str = "multiple_choice") -> BenchmarkSpec:
    return BenchmarkSpec(
        name=name, family="t", format=fmt, language="en",
        variants=[], applicable_methods=["perm_option"],
        trustworthiness_default=Verdict.SUSPECT,
        data_source="hf", data_id="x/y",
    )


def _mcq(i: int, n_opt: int = 4, answer_idx: int = 0) -> BenchmarkQuestion:
    """选项内容里嵌入题号 + 选项编号，方便 mock model 解析。

    用 [QID=N] 作为题号锚点，避免和 "Question:" 前缀混淆。
    """
    return BenchmarkQuestion(
        id=f"q-{i}", benchmark="mmlu-pro", format="multiple_choice",
        prompt=f"[QID={i}] pick the right one",
        answer=_LETTERS[answer_idx],
        choices=[f"q{i}_opt{j}" for j in range(n_opt)],
        answer_index=answer_idx,
    )


# ----------------------------- mock models ----------------------------- #


class _NoLogprobModel(ModelInterface):
    name = "blackbox"
    stage_tag = "sft"

    def supports(self, cap: Capability) -> bool:
        return cap == Capability.GENERATE

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        return ""


class _ContentTrackingModel(ModelInterface):
    """干净模型：识别 prompt 中正确选项的内容，给对应字母高 logp。

    通过解析 prompt 找出哪个选项以 'q{i}_opt{answer_idx}' 命名的内容 ——
    用一个外部 dict 记 'q{i} 的正确内容是 opt0'，与 questions 的 answer_idx 对齐。
    """

    name = "content_tracker"
    stage_tag = "sft"

    def __init__(self, correct_content_by_q: dict[int, str]) -> None:
        # {q_id: 正确选项的 content 字符串}
        self.correct = correct_content_by_q

    def supports(self, cap: Capability) -> bool:
        return cap in {Capability.LOGPROBS, Capability.GENERATE}

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        return ""

    def logprobs(self, prompt: str, completion: str) -> np.ndarray:
        # completion 是 " A" / " B" / ...
        letter = completion.strip()
        # 解析 prompt 找该 letter 行的选项内容
        target_line = None
        for line in prompt.split("\n"):
            if line.startswith(f"{letter}. "):
                target_line = line[len(f"{letter}. "):]
                break
        if target_line is None:
            return np.array([-10.0], dtype=np.float64)
        # 解析 prompt 取 q 编号：找 [QID=N]
        import re
        m = re.search(r"\[QID=(\d+)\]", prompt)
        if m is None:
            return np.array([-10.0], dtype=np.float64)
        qnum = int(m.group(1))
        correct_content = self.correct.get(qnum)
        if target_line == correct_content:
            return np.array([-0.1], dtype=np.float64)
        return np.array([-5.0], dtype=np.float64)


class _PositionStickyModel(ModelInterface):
    """污染模型：永远偏好字母 A（位置 0），不看内容。

    模拟 "SFT 训练时正确答案在 A 的题被多 epoch 训过，模型把 prompt-shape → A 固化"。
    """

    name = "position_sticky"
    stage_tag = "sft"

    def supports(self, cap: Capability) -> bool:
        return cap in {Capability.LOGPROBS, Capability.GENERATE}

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        return ""

    def logprobs(self, prompt: str, completion: str) -> np.ndarray:
        letter = completion.strip()
        return np.array([-0.1 if letter == "A" else -5.0], dtype=np.float64)


# ----------------------------- prerequisites ----------------------------- #


def test_rejects_non_mc_benchmark():
    qs = [_mcq(i) for i in range(60)]
    with pytest.raises(ValueError, match="multiple_choice"):
        option_permutation_test(
            _PositionStickyModel(),
            _spec(name="gsm8k", fmt="math_cot"),
            qs,
        )


def test_returns_inconclusive_without_logprobs():
    qs = [_mcq(i) for i in range(60)]
    r = option_permutation_test(_NoLogprobModel(), _spec(), qs, min_samples=50)
    assert not r.prerequisites_met
    assert r.verdict_hint == Verdict.INCONCLUSIVE
    assert r.signal is None
    assert "logprobs" in (r.error or "")


def test_returns_inconclusive_when_too_few_samples():
    qs = [_mcq(i) for i in range(10)]
    r = option_permutation_test(
        _PositionStickyModel(), _spec(), qs, min_samples=50,
    )
    assert not r.prerequisites_met
    assert r.signal is None
    assert "valid MC questions" in (r.error or "")


def test_filters_questions_without_choices():
    """缺 choices 或 answer_index 的题不计入有效样本。"""
    good = [_mcq(i) for i in range(40)]
    bad = [
        BenchmarkQuestion(
            id=f"bad-{i}", benchmark="x", format="multiple_choice",
            prompt="...", answer="A", choices=None, answer_index=None,
        )
        for i in range(40)
    ]
    r = option_permutation_test(
        _PositionStickyModel(), _spec(), good + bad,
        min_samples=50,
    )
    assert not r.prerequisites_met
    assert r.evidence["n_valid"] == 40
    assert r.evidence["n_total"] == 80


# ----------------------------- behavior ----------------------------- #


def test_clean_model_signal_near_zero():
    """ContentTracking 模型 shuffle 不影响识别 → signal ≈ 0。"""
    qs = [_mcq(i, n_opt=4, answer_idx=i % 4) for i in range(60)]
    correct = {i: f"q{i}_opt{i % 4}" for i in range(60)}
    r = option_permutation_test(
        _ContentTrackingModel(correct), _spec(), qs,
        n_permutations=5, min_samples=50, seed=7,
    )
    assert r.prerequisites_met
    assert r.signal is not None
    assert abs(r.signal) < 0.05
    assert r.verdict_hint == Verdict.CLEAN
    # 原序和打乱都接近 100% 正确
    assert r.evidence["mean_acc_original"] > 0.95
    assert r.evidence["mean_acc_permuted"] > 0.95


def test_position_sticky_model_signal_high():
    """永远输出 A 的模型：原序里 answer_index=0 的题 → 100% 正确
    （因为正确字母总是 A）；shuffle 后正确字母位置变了，模型还输出 A → 大概率错。
    signal ≈ acc_orig - acc_perm 应明显大于 0.05。
    """
    # 全部 answer_index=0，原序下 sticky model 100% 正确；shuffle 后正确字母被随机分布
    qs = [_mcq(i, n_opt=4, answer_idx=0) for i in range(60)]
    r = option_permutation_test(
        _PositionStickyModel(), _spec(), qs,
        n_permutations=10, min_samples=50, seed=7,
    )
    assert r.prerequisites_met
    assert r.signal is not None
    # 原序 100%, shuffle 后约 1/4（正确字母变到 A 的概率）→ signal ≈ 0.75
    assert r.evidence["mean_acc_original"] == 1.0
    assert r.evidence["mean_acc_permuted"] < 0.4
    assert r.signal > 0.5
    assert r.verdict_hint == Verdict.DIRTY


def test_evidence_contains_required_fields():
    qs = [_mcq(i, n_opt=4, answer_idx=0) for i in range(60)]
    r = option_permutation_test(
        _PositionStickyModel(), _spec(), qs,
        n_permutations=3, min_samples=50, seed=7,
    )
    ev = r.evidence
    for k in (
        "n_questions", "n_permutations",
        "mean_acc_original", "mean_acc_permuted",
        "leak_score", "alpha_dirty", "alpha_suspect",
    ):
        assert k in ev, f"evidence missing {k}"
    assert ev["n_questions"] == 60
    assert ev["n_permutations"] == 3
    assert ev["leak_score"] == r.signal


# ----------------------------- helper ----------------------------- #


def test_predict_letter_argmax_logp():
    """_predict_letter 应返回 logp 最大的字母。"""
    qs = [_mcq(0, n_opt=4, answer_idx=2)]
    correct = {0: "q0_opt2"}
    model = _ContentTrackingModel(correct)
    pred = _predict_letter(model, qs[0].prompt, qs[0].choices)
    # opt2 在位置 C → mock model 给 C 高 logp
    assert pred == "C"


def test_signal_alignment_with_diff_matrix_direction():
    """信号方向与 diff_matrix.signal_direction_for("perm_option") 对齐：
    higher_is_dirtier，clean=0，sticky 应远大于 clean。
    """
    from model_contamination.attribution.diff_matrix import signal_direction_for
    assert signal_direction_for("perm_option") == "higher_is_dirtier"
    # clean 信号 < sticky 信号 已在上面测试覆盖


def test_next_token_logprobs_default_fallback_matches_logprobs():
    """未 override next_token_logprobs 的 mock 后端应走 base 类默认 fallback，
    结果等价于逐 candidate sum(logprobs)。这条覆盖 base.py 新加的默认实现。
    """
    model = _PositionStickyModel()  # 只实现 logprobs，next_token_logprobs 走 fallback
    prompt = "Question: foo\nA. x\nAnswer:"
    candidates = [" A", " B", " C"]
    got = model.next_token_logprobs(prompt, candidates)
    ref = np.array(
        [float(np.sum(model.logprobs(prompt, c))) for c in candidates],
        dtype=np.float64,
    )
    assert got.shape == (3,)
    assert np.allclose(got, ref)
    # PositionSticky 里 " A" 得 -0.1，其他 -5.0 → " A" argmax
    assert int(np.argmax(got)) == 0
