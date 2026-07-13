"""Benchmark loader 单测。

不依赖具体数据集，覆盖：
- normalizer 函数对单行 row 的归一化逻辑（不真下数据）
- _extract_boxed latex 解析
- 'TBD' / 未注册 normalizer / 未知 data_source 的错误路径
"""

from __future__ import annotations

import pytest

from model_contamination.benchmarks.loader import (
    _extract_boxed,
    _normalize_gsm8k,
    _normalize_math,
    _normalize_math_500,
    _normalize_mmlu_cf,
    _normalize_mmlu_pro,
    load_questions,
)
from model_contamination.types import BenchmarkSpec, Verdict


def _spec(name: str, fmt: str = "math_cot", source: str = "hf", data_id: str = "x/y") -> BenchmarkSpec:
    return BenchmarkSpec(
        name=name, family="t", format=fmt, language="en",
        variants=[], applicable_methods=[],
        trustworthiness_default=Verdict.SUSPECT,
        data_source=source, data_id=data_id,
    )


# ---------- normalizers ---------- #

def test_normalize_gsm8k_extracts_final_answer():
    row = {"question": "2+3=?", "answer": "Step 1...\n#### 5"}
    q = _normalize_gsm8k(row, 0, _spec("gsm8k"))
    assert q.format == "math_cot"
    assert q.prompt == "2+3=?"
    assert q.answer == "5"
    assert q.full_answer.endswith("#### 5")


def test_normalize_gsm8k_handles_comma_and_negatives():
    row = {"question": "q", "answer": "blah\n#### -1,234"}
    q = _normalize_gsm8k(row, 0, _spec("gsm8k"))
    assert q.answer == "-1234"


def test_normalize_math_extracts_boxed():
    row = {
        "problem": "Solve x.",
        "solution": "We get \\boxed{42}.",
        "level": "Level 3",
        "type": "Algebra",
    }
    q = _normalize_math(row, 7, _spec("math"))
    assert q.id == "math-7"
    assert q.answer == "42"
    assert q.raw["level"] == "Level 3"


def test_normalize_math_500_passthrough_answer():
    row = {"problem": "p", "answer": "  3.14  ", "level": 5, "subject": "Algebra",
           "solution": "First expand, then \\boxed{3.14}."}
    q = _normalize_math_500(row, 0, _spec("math-500"))
    assert q.answer == "3.14"
    # CoT 全文提到顶层 full_answer（SPV-MIA 等 MIA 方法读固定字段，不再掉进 raw）
    assert q.full_answer == "First expand, then \\boxed{3.14}."


def test_normalize_math_populates_full_answer():
    row = {"problem": "Solve x.", "solution": "We get \\boxed{42} after two steps.",
           "level": "Level 3", "type": "Algebra"}
    q = _normalize_math(row, 0, _spec("math"))
    assert q.answer == "42"
    assert q.full_answer == "We get \\boxed{42} after two steps."


def test_normalize_mmlu_pro_letter_and_index():
    row = {
        "question": "Capital of France?",
        "options": ["Berlin", "Madrid", "Paris", "Rome"],
        "answer_index": 2,
        "category": "geo",
        "src": "wiki",
    }
    q = _normalize_mmlu_pro(row, 0, _spec("mmlu-pro", fmt="multiple_choice"))
    assert q.format == "multiple_choice"
    assert q.choices == ["Berlin", "Madrid", "Paris", "Rome"]
    assert q.answer == "C"
    assert q.answer_index == 2


def test_normalize_mmlu_cf_columns():
    row = {
        "Question": "q?", "A": "a", "B": "b", "C": "c", "D": "d",
        "Answer": "B", "Subject": "math", "Category": "stem",
    }
    q = _normalize_mmlu_cf(row, 3, _spec("mmlu-cf", fmt="multiple_choice"))
    assert q.answer == "B"
    assert q.answer_index == 1
    assert q.choices == ["a", "b", "c", "d"]


# ---------- _extract_boxed ---------- #

def test_extract_boxed_simple():
    assert _extract_boxed("ans is \\boxed{42}.") == "42"


def test_extract_boxed_nested_braces():
    assert _extract_boxed("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


def test_extract_boxed_absent():
    assert _extract_boxed("no boxed here") is None


# ---------- load_questions error paths ---------- #

def test_load_questions_rejects_tbd():
    spec = _spec("aime-2025", data_id="TBD")
    with pytest.raises(ValueError, match="data_id 未确定"):
        load_questions(spec)


def test_load_questions_rejects_unknown_normalizer():
    spec = _spec("ifeval", source="hf", data_id="google/IFEval")
    with pytest.raises(NotImplementedError, match="尚未注册 normalizer"):
        load_questions(spec)


def test_load_questions_unknown_source():
    spec = _spec("x", source="weird", data_id="x")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="未知 data_source"):
        load_questions(spec)
