"""跨方法共用的"在 model 上跑分"工具。

接口契约：
    evaluate_accuracy(model, questions) -> float ∈ [0, 1]

按 question.format dispatch：
    multiple_choice  → 对每个 choice 算 mean token logprob，argmax 与 answer_index 比
    math_cot         → generate CoT，抽取 \\boxed{...} 或最后一个数字，与 answer 比

family_diff / paraphrase 都直接调这里。

第一版只做 MC + math_cot，其他 format raise NotImplementedError。

## Prompt 模板策略

**SFT ckpt 只有在 chat template 下才能进入训练分布**——用 `Question: ... Answer:` 硬模板评测 SFT ckpt，模型进不了训练表征，输出乱七八糟，acc 假阴。

因此：模型 tokenizer 若能 `apply_chat_template`，则走 chat 分支（SFT 训练模板同源）；否则回退硬模板（兼容 base / mock model）。

MC 路径的 LOGPROBS 分支（`mean-logprob over raw choice text`）假设模型把 choice
文本作为自然 continuation 输出——对 chat SFT ckpt 完全不成立（它训练成"输出字母"）。
所以当模型有 chat template 时强制走黑盒生成路径（要模型输出字母），与训练分布同源。
LOGPROBS 分支仅在 base 类模型（无 chat template）保留。
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


# ----------------------------- prompt template ----------------------------- #


def _wrap_chat(model: ModelInterface, user_content: str) -> str:
    """若 model tokenizer 有 chat template，wrap 到 assistant 生成起点；否则返回 None 由上层 fallback。"""
    tokenizer = getattr(model, "_tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return ""
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return ""


# ----------------------------- multiple_choice ----------------------------- #


def _score_multiple_choice(model: ModelInterface, q: BenchmarkQuestion) -> bool:
    """对每个 choice 算 token-level mean logprob，argmax 即预测答案。

    路由：
    - 有 chat template（SFT / chat 类）→ **强制**走黑盒（要模型输出字母），
      LOGPROBS 分支下的 `mean-logprob over raw choice text` 与 chat 训练分布不匹配，
      信号被噪声淹没（v3 smoke: mmlu_heavy × mmlu-pro acc_orig=0.1，SFT ckpt 泛化不该这么低）。
    - 无 chat template 且支持 LOGPROBS（base 类）→ 走 mean-logprob 打分路径。
    - 其余（黑盒仅 generate）→ 让模型直接答字母再 string compare。
    """
    if q.choices is None or q.answer_index is None:
        return False

    chat = _wrap_chat(model, _mc_prompt(q))
    if chat:
        # chat 模型强制黑盒：要模型输出字母
        prompt = chat + "The answer is"
        out = model.generate(prompt, max_tokens=4, temperature=0.0).strip()
        return out[:1].upper() == q.answer

    if model.supports(Capability.LOGPROBS):
        prompt = _mc_prompt(q)
        scores: list[float] = []
        for choice in q.choices:
            completion = f" {choice}"
            lp = model.logprobs(prompt, completion)
            scores.append(float(np.mean(lp)) if lp.size else float("-inf"))
        pred = int(np.argmax(scores))
        return pred == q.answer_index

    # 黑盒仅 generate 的降级路径
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
    """generate CoT + 抽取数值答案。

    模板策略：SFT ckpt tokenizer 通常带 chat template → wrap 到 Qwen3/Llama chat
    起点（与训练时 alpaca+template 分布同源）；无 chat template（base / mock）
    走 `Question: ... Answer:` 硬模板。

    抽取兼容 SFT 训练 output 格式 `... #### {N}`（`_NUM_RE.findall(...)[-1]`
    自然拿到 N）与 base 模型 CoT 里的 `\\boxed{...}` / 最后数字。
    """
    chat = _wrap_chat(model, q.prompt)
    prompt = chat if chat else f"Question: {q.prompt}\nAnswer:"
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
