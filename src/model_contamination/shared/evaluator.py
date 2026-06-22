"""跨方法共用的"在 model 上跑分"工具。

接口契约：
    evaluate_accuracy(model, questions) -> float ∈ [0, 1]

按 question.format dispatch：
    multiple_choice  → 对每个 choice 算 mean token logprob，argmax 与 answer_index 比
    math_cot         → generate CoT，抽取 \\boxed{...} 或最后一个数字，与 answer 比

family_diff / oren / sanity-check 都直接调这里。

第一版只做 MC + math_cot，其他 format raise NotImplementedError。
"""

from __future__ import annotations

import re

import numpy as np

from model_contamination.models.base import Capability, ModelInterface
from model_contamination.types import BenchmarkQuestion


def evaluate_accuracy(
    model: ModelInterface,
    questions: list[BenchmarkQuestion],
    *,
    max_tokens: int = 512,
) -> float:
    """返回 model 在 questions 上的 accuracy（0–1）。"""
    if not questions:
        return 0.0
    correct = 0
    for q in questions:
        if q.format == "multiple_choice":
            ok = _score_multiple_choice(model, q)
        elif q.format == "math_cot":
            ok = _score_math_cot(model, q, max_tokens=max_tokens)
        else:
            raise NotImplementedError(
                f"evaluator 暂未支持 format='{q.format}'（题目 {q.id}）"
            )
        correct += int(ok)
    return correct / len(questions)


# ----------------------------- multiple_choice ----------------------------- #


def _score_multiple_choice(model: ModelInterface, q: BenchmarkQuestion) -> bool:
    """对每个 choice 算 token-level mean logprob，argmax 即预测答案。

    用 mean logprob 而非 sum，避免长选项被惩罚。前置：模型支持 LOGPROBS；
    黑盒模型（仅 generate）走 fallback：让模型直接答字母再 string compare。
    """
    if q.choices is None or q.answer_index is None:
        return False
    if model.supports(Capability.LOGPROBS):
        prompt = _mc_prompt(q)
        scores: list[float] = []
        for choice in q.choices:
            completion = f" {choice}"
            lp = model.logprobs(prompt, completion)
            scores.append(float(np.mean(lp)) if lp.size else float("-inf"))
        pred = int(np.argmax(scores))
        return pred == q.answer_index

    # 黑盒回退：让模型生成一个字母
    prompt = _mc_prompt(q) + "\nAnswer:"
    out = model.generate(prompt, max_tokens=4, temperature=0.0).strip()
    return out[:1].upper() == q.answer


def _mc_prompt(q: BenchmarkQuestion) -> str:
    assert q.choices is not None
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(q.choices))
    return f"{q.prompt}\n{body}"


# ----------------------------- math_cot ----------------------------- #

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


def _score_math_cot(model: ModelInterface, q: BenchmarkQuestion, *, max_tokens: int) -> bool:
    prompt = f"Question: {q.prompt}\nAnswer:"
    out = model.generate(prompt, max_tokens=max_tokens, temperature=0.0)
    pred = extract_math_answer(out)
    return _math_answers_equal(pred, q.answer)


def extract_math_answer(text: str) -> str:
    """优先抽 \\boxed{...}；否则取文本中最后一个数字。"""
    # boxed
    i = text.find("\\boxed{")
    if i >= 0:
        depth = 0
        start = i + len("\\boxed{")
        for j in range(start, len(text)):
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                if depth == 0:
                    return text[start:j].strip()
                depth -= 1
    # 最后一个数字
    matches = _NUM_RE.findall(text)
    return matches[-1].replace(",", "") if matches else ""


def _math_answers_equal(pred: str, gold: str) -> bool:
    if not pred:
        return False
    p = pred.strip().replace(",", "").replace(" ", "")
    g = gold.strip().replace(",", "").replace(" ", "")
    if p == g:
        return True
    # 数值比较（兼容 3 vs 3.0 / 1/2 vs 0.5）
    try:
        return abs(_as_float(p) - _as_float(g)) < 1e-6
    except ValueError:
        return False


def _as_float(s: str) -> float:
    if "/" in s:
        a, b = s.split("/", 1)
        return float(a) / float(b)
    return float(s)
