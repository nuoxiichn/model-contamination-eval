"""evaluator 单测。

不依赖 transformers / 真模型，用一个 mock ModelInterface 验证 dispatch / scoring
逻辑；纯函数 extract_math_answer 直接测。
"""

from __future__ import annotations

import numpy as np
import pytest

from model_contamination.models.base import Capability, ModelInterface
from model_contamination.shared.evaluator import (
    evaluate_accuracy,
    extract_math_answer,
)
from model_contamination.types import BenchmarkQuestion


# ----------------------------- mock model ----------------------------- #


class _MockLogprobModel(ModelInterface):
    """对每个 prompt+completion 返回固定 logp，便于精确控制 MC scoring。"""

    def __init__(self, completion_to_meanlogp: dict[str, float]) -> None:
        self.name = "mock-lp"
        self.stage_tag = "base"
        self._scores = completion_to_meanlogp

    def supports(self, cap: Capability) -> bool:
        return cap in {Capability.LOGPROBS, Capability.GENERATE}

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        return ""

    def logprobs(self, prompt: str, completion: str) -> np.ndarray:
        # 完整 completion 字符串作为 key；返回 1 token，便于 mean=该值
        return np.array([self._scores.get(completion, -10.0)], dtype=np.float64)


class _MockGenModel(ModelInterface):
    """只支持 generate；按 prompt 前缀返回预设输出。"""

    def __init__(self, responses: list[str]) -> None:
        self.name = "mock-gen"
        self.stage_tag = "base"
        self._responses = list(responses)
        self._i = 0

    def supports(self, cap: Capability) -> bool:
        return cap == Capability.GENERATE

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        out = self._responses[self._i % len(self._responses)]
        self._i += 1
        return out

    def logprobs(self, prompt: str, completion: str) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


# ----------------------------- extract_math_answer ----------------------------- #


def test_extract_boxed_simple():
    assert extract_math_answer("The answer is \\boxed{42}.") == "42"


def test_extract_boxed_nested():
    assert extract_math_answer("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


def test_extract_fallback_last_number():
    # 没有 boxed，取最后一个数
    assert extract_math_answer("Step 1: 3. Step 2: 7. So 19.") == "19"


def test_extract_handles_decimals_and_negatives():
    assert extract_math_answer("ans -3.14") == "-3.14"


def test_extract_empty():
    assert extract_math_answer("no number here") == ""


# ----------------------------- MC dispatch ----------------------------- #


def _mc_q(choices: list[str], answer_index: int) -> BenchmarkQuestion:
    return BenchmarkQuestion(
        id="t",
        benchmark="t",
        format="multiple_choice",
        prompt="Q?",
        choices=choices,
        answer="ABCDEFG"[answer_index],
        answer_index=answer_index,
    )


def test_mc_picks_highest_logprob_choice():
    q = _mc_q(["red", "blue", "green"], answer_index=1)
    # 让 " blue" 拿最高 logprob
    model = _MockLogprobModel({" red": -5.0, " blue": -1.0, " green": -3.0})
    acc = evaluate_accuracy(model, [q])
    assert acc == 1.0


def test_mc_wrong_pick_drops_accuracy():
    q = _mc_q(["red", "blue", "green"], answer_index=1)
    model = _MockLogprobModel({" red": -1.0, " blue": -5.0, " green": -3.0})
    assert evaluate_accuracy(model, [q]) == 0.0


def test_mc_blackbox_fallback():
    """模型不支持 logprobs → 让它直接生成字母。"""
    q = _mc_q(["red", "blue", "green"], answer_index=2)
    model = _MockGenModel(["C. green"])
    assert evaluate_accuracy(model, [q]) == 1.0


class _FakeChatTokenizer:
    """最小 chat tokenizer 模拟：apply_chat_template 返回 user/assistant 标记的字符串。"""

    def apply_chat_template(
        self, messages: list[dict], tokenize: bool = False, add_generation_prompt: bool = True
    ) -> str:
        user = messages[0]["content"]
        return f"<|user|>{user}<|assistant|>"


class _MockChatLogprobModel(_MockLogprobModel):
    """支持 logprobs + 有 chat template：验证路由强制走黑盒。"""

    def __init__(self, gen_output: str) -> None:
        super().__init__({})
        self._tokenizer = _FakeChatTokenizer()
        self._gen = gen_output

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        return self._gen


def test_mc_chat_model_forces_blackbox_over_logprobs():
    """有 chat template 的模型（即便支持 logprobs）也应走黑盒字母输出路径。

    根因：`mean-logprob over raw choice text` 与 chat SFT 训练分布不匹配，信号被噪声
    淹没（v3 smoke: mmlu_heavy × mmlu-pro acc_orig=0.1）。
    """
    q = _mc_q(["red", "blue", "green"], answer_index=1)
    # logprob 侧故意让 " red" 最高——若路由错误会答 A（错）；黑盒返回 "B" → 应答对
    model = _MockChatLogprobModel(gen_output=" B")
    model._scores = {" red": -1.0, " blue": -5.0, " green": -3.0}
    assert evaluate_accuracy(model, [q]) == 1.0


# ----------------------------- math_cot dispatch ----------------------------- #


def _math_q(answer: str) -> BenchmarkQuestion:
    return BenchmarkQuestion(
        id="m", benchmark="m", format="math_cot",
        prompt="2+2=?", answer=answer,
    )


def test_math_cot_correct_boxed():
    model = _MockGenModel(["After thinking, \\boxed{4}."])
    assert evaluate_accuracy(model, [_math_q("4")]) == 1.0


def test_math_cot_numeric_equivalence():
    """gold='0.5' 与 pred='1/2' 应视为相等。"""
    model = _MockGenModel(["So the answer is \\boxed{1/2}."])
    assert evaluate_accuracy(model, [_math_q("0.5")]) == 1.0


def test_math_cot_wrong_drops_accuracy():
    model = _MockGenModel(["\\boxed{5}"])
    assert evaluate_accuracy(model, [_math_q("4")]) == 0.0


# ----------------------------- edge cases ----------------------------- #


def test_empty_questions_returns_zero():
    model = _MockGenModel([""])
    assert evaluate_accuracy(model, []) == 0.0


def test_unknown_format_raises():
    q = BenchmarkQuestion(
        id="x", benchmark="x", format="code_completion",  # type: ignore[arg-type]
        prompt="def f():", answer="pass",
    )
    model = _MockGenModel(["pass"])
    with pytest.raises(NotImplementedError):
        evaluate_accuracy(model, [q])
