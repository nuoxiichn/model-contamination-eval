"""paraphrase_stress 单测。

覆盖：
- 前置：不支持 GENERATE / paraphraser 未实装 / n_paraphrases 非法 / min_samples
- format 过滤（只支持 multiple_choice / math_cot）
- 短 prompt 跳过（< _MIN_PROMPT_WORDS）
- rewriter 确定性（同 seed 同 rewriter → 同 output）
- rewriter 至少改点什么（output != input，除非无可用规则）
- 干净 mock model（识别语义）→ signal ≈ 0，CLEAN
- 污染 mock model（记原始 prompt token）→ signal > 阈值，DIRTY
- prerequisites_met 分支返回结构完整
- Cache 命中：第二次调用时 rewrite 被跳过（读缓存）
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from model_contamination.models.base import Capability, ModelInterface
from model_contamination.shared.paraphrase_stress import (
    _MIN_PROMPT_WORDS,
    _rule_paraphrase,
    paraphrase_stress_test,
)
from model_contamination.types import BenchmarkQuestion, BenchmarkSpec, Verdict

# ----------------------------- helpers ----------------------------- #


def _spec(name: str = "mmlu-pro", fmt: str = "multiple_choice") -> BenchmarkSpec:
    return BenchmarkSpec(
        name=name, family="t", format=fmt, language="en",
        variants=[], applicable_methods=["paraphrase"],
        trustworthiness_default=Verdict.SUSPECT,
        data_source="hf", data_id="x/y",
    )


def _mcq(i: int, n_opt: int = 4, answer_idx: int = 0) -> BenchmarkQuestion:
    """MC 题：prompt 足够长，可被规则改写。"""
    return BenchmarkQuestion(
        id=f"mc-{i}",
        benchmark="mmlu-pro",
        format="multiple_choice",
        prompt=(
            f"This is question number {i}. However, please choose the correct "
            "option from the list. Because it is important."
        ),
        answer="ABCDEFGHIJ"[answer_idx],
        choices=[f"content_{i}_opt{j}" for j in range(n_opt)],
        answer_index=answer_idx,
    )


def _mathq(i: int) -> BenchmarkQuestion:
    return BenchmarkQuestion(
        id=f"m-{i}",
        benchmark="gsm8k",
        format="math_cot",
        prompt=(
            f"What is {i} plus {i}? Please compute step by step and "
            "provide the numeric answer."
        ),
        answer=str(2 * i),
    )


# ----------------------------- mock models ----------------------------- #


class _NoGenerateModel(ModelInterface):
    name = "no_gen"
    stage_tag = "sft"

    def supports(self, cap: Capability) -> bool:
        return cap == Capability.LOGPROBS  # 只支持 logprobs，不支持 generate

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        raise RuntimeError("should not be called")


class _CleanContentModel(ModelInterface):
    """干净模型：能识别 prompt 中"question number N"里的 N，
    只要 N 还在 prompt 里就正确回答（对改写鲁棒）。

    - MC: 给正确选项 letter 高 logprob
    - math_cot: generate 输出正确数字
    """

    name = "clean_content"
    stage_tag = "sft"

    def supports(self, cap: Capability) -> bool:
        return cap in {Capability.GENERATE, Capability.LOGPROBS}

    def _extract_qnum(self, prompt: str) -> int | None:
        # 匹配 "question number N" / "What is N plus N"
        m = re.search(r"question number (\d+)", prompt, re.IGNORECASE)
        if m:
            return int(m.group(1))
        m = re.search(r"what is (\d+) plus", prompt, re.IGNORECASE)
        if m:
            return int(m.group(1))
        return None

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        n = self._extract_qnum(prompt)
        if n is None:
            return "0"
        return f"The answer is {2 * n}."

    def logprobs(self, prompt: str, completion: str) -> np.ndarray:
        # 找到正确 answer_index：解析 prompt 中的选项行 "A. content_N_opt{answer_idx}"
        # question number N 是题号；对应的正确选项 letter 依赖上层 answer_idx，
        # 但 mock 不知道 answer_idx——用启发式：如果 completion 里的 letter 对应的
        # 选项行内容里含 "opt0"，就给高 logprob（干净模型追内容 opt0）
        letter = completion.strip()
        target_line = None
        for line in prompt.split("\n"):
            if line.strip().startswith(f"{letter}. "):
                target_line = line.strip()[len(f"{letter}. "):]
                break
        if target_line is None:
            return np.array([-5.0], dtype=np.float64)
        # 判是不是正确的 opt（约定测试里 answer_idx=0 → 正确内容含 "opt0"）
        # 这里 answer_idx 从题目 answer 字段获取——间接：字母 A 对应 opt0
        # 简化：如果内容含 "opt0" → 高 logp
        if "opt0" in target_line:
            return np.array([-0.1], dtype=np.float64)
        return np.array([-5.0], dtype=np.float64)


class _MemorizedPromptModel(ModelInterface):
    """污染模型：记住"原始 prompt 完全一致"→ 输出正确答案；
    prompt 有任何字符扰动 → 输出错答案（chance 水平）。
    """

    name = "memorized"
    stage_tag = "sft"

    def __init__(self, known_prompts: set[str]) -> None:
        self._known = known_prompts

    def supports(self, cap: Capability) -> bool:
        return cap in {Capability.GENERATE, Capability.LOGPROBS}

    def _prompt_seen(self, prompt: str) -> bool:
        # 精确匹配（prompt 里嵌了 evaluator 加的 "A. ..." 行）——比较去掉 MC body 部分
        # 简化：只要 prompt 里包含任一 known prompt 的核心串，就算见过
        return any(kp in prompt for kp in self._known)

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        if self._prompt_seen(prompt):
            # 从 "What is N plus N" 里提出 N，正确输出 2*N
            m = re.search(r"what is (\d+) plus", prompt, re.IGNORECASE)
            if m:
                return f"The answer is {2 * int(m.group(1))}."
            return "0"
        return "-999"  # 未见过 → 错答案

    def logprobs(self, prompt: str, completion: str) -> np.ndarray:
        letter = completion.strip()
        if self._prompt_seen(prompt):
            # 见过：正确字母 A（约定 answer_idx=0）
            if letter == "A":
                return np.array([-0.1], dtype=np.float64)
            return np.array([-5.0], dtype=np.float64)
        # 未见过：全部 letter 均匀
        return np.array([-2.0], dtype=np.float64)


# ----------------------------- prerequisites ----------------------------- #


def test_returns_inconclusive_without_generate():
    qs = [_mcq(i) for i in range(30)]
    r = paraphrase_stress_test(
        _NoGenerateModel(), _spec(), qs, min_samples=20,
    )
    assert not r.prerequisites_met
    assert r.verdict_hint == Verdict.INCONCLUSIVE
    assert r.signal is None
    assert "GENERATE" in (r.error or "") or "generate" in (r.error or "").lower()


def test_unsupported_paraphraser():
    qs = [_mcq(i) for i in range(30)]
    r = paraphrase_stress_test(
        _CleanContentModel(), _spec(), qs, paraphraser="llm",  # type: ignore[arg-type]
        min_samples=20,
    )
    assert not r.prerequisites_met
    assert "not implemented" in (r.error or "").lower() or "llm" in (r.error or "")


@pytest.mark.parametrize("n", [0, -1])
def test_invalid_n_paraphrases(n):
    qs = [_mcq(i) for i in range(30)]
    r = paraphrase_stress_test(
        _CleanContentModel(), _spec(), qs, n_paraphrases=n, min_samples=20,
    )
    assert not r.prerequisites_met
    assert "n_paraphrases" in (r.error or "")


def test_too_few_samples():
    qs = [_mcq(i) for i in range(5)]
    r = paraphrase_stress_test(
        _CleanContentModel(), _spec(), qs, min_samples=20,
    )
    assert not r.prerequisites_met
    assert "samples" in (r.error or "").lower()


def test_filters_unsupported_format():
    """format 不在 {multiple_choice, math_cot} 的 sample 被过滤。"""
    qs = [
        BenchmarkQuestion(
            id=f"cloze-{i}", benchmark="x", format="cloze",
            prompt="fill in the blank please here now", answer="foo",
        )
        for i in range(30)
    ]
    r = paraphrase_stress_test(
        _CleanContentModel(), _spec(fmt="cloze"), qs, min_samples=20,
    )
    assert not r.prerequisites_met
    assert r.evidence["n_valid"] == 0


# ----------------------------- rewriter properties ----------------------------- #


def test_rule_paraphrase_deterministic_with_same_seed():
    text = "This is a question. However, please choose one option because it matters."
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    out1 = _rule_paraphrase(text, rng1)
    out2 = _rule_paraphrase(text, rng2)
    assert out1 == out2


def test_rule_paraphrase_changes_text():
    """长文本经过 rule paraphrase 至少要变一点（非全等）。"""
    text = "This is a question. However, please choose one option because it matters."
    rng = np.random.default_rng(0)
    # 采 5 次，至少有一次输出与原文不同
    outs = [_rule_paraphrase(text, rng) for _ in range(5)]
    assert any(o != text for o in outs)


def test_short_prompt_skipped():
    """< _MIN_PROMPT_WORDS 的 prompt 会被跳过（不参与评测）。"""
    long = [_mathq(i) for i in range(1, 21)]
    short = [
        BenchmarkQuestion(
            id=f"s-{i}", benchmark="gsm8k", format="math_cot",
            prompt="hi",  # 1 word
            answer="0",
        )
        for i in range(10)
    ]
    r = paraphrase_stress_test(
        _CleanContentModel(), _spec(name="gsm8k", fmt="math_cot"), long + short,
        min_samples=20, n_paraphrases=2, seed=1,
    )
    # 20 有效 + 10 skipped_short = 30 valid，但 10 无法 rewrite
    assert r.evidence["n_skipped_short"] == 10
    assert r.evidence["n_scorable"] == 20
    _ = _MIN_PROMPT_WORDS  # just reference to avoid lint


def test_mc_benchmark_skipped_as_unsupported():
    """MC 题型已确认结构性失效 → paraphrase 对 MC benchmark 返回 skip（不作为检测信号）。"""
    qs = [_mcq(i) for i in range(30)]
    r = paraphrase_stress_test(
        _CleanContentModel(), _spec(), qs, min_samples=20, n_paraphrases=3, seed=1,
    )
    assert r.prerequisites_met is False
    assert r.verdict_hint == Verdict.INCONCLUSIVE
    assert r.signal is None
    assert r.evidence["n_mc"] == 30
    assert "multiple_choice" in (r.error or "")


# ----------------------------- behavior signal direction ----------------------------- #


def test_clean_model_signal_near_zero_math():
    """干净模型识别 prompt 里的数字，改写 wrapper / 标点后仍能答对 → signal 很小。"""
    qs = [_mathq(i) for i in range(1, 25)]
    r = paraphrase_stress_test(
        _CleanContentModel(), _spec(name="gsm8k", fmt="math_cot"), qs,
        n_paraphrases=3, min_samples=20, seed=7,
    )
    assert r.prerequisites_met
    assert r.signal is not None
    # 干净模型 acc 原始 = 1.0；改写后仍能提取 "N plus N" → 仍 1.0
    assert r.evidence["acc_original"] >= 0.95
    assert r.signal < 0.10
    assert r.verdict_hint in {Verdict.CLEAN, Verdict.SUSPECT}


def test_contaminated_model_signal_high_math():
    """污染模型只在原始 prompt 上答对；任何改写破坏原字符串 → acc 掉。"""
    qs = [_mathq(i) for i in range(1, 25)]
    # known_prompts 用每题原始 prompt
    known = {q.prompt for q in qs}
    model = _MemorizedPromptModel(known)
    r = paraphrase_stress_test(
        model, _spec(name="gsm8k", fmt="math_cot"), qs,
        n_paraphrases=5, min_samples=20, seed=7,
    )
    assert r.prerequisites_met
    assert r.signal is not None
    # 原始 acc 应接近 1；改写 acc 显著下降
    assert r.evidence["acc_original"] >= 0.9
    assert r.evidence["acc_worst_case"] <= 0.3
    assert r.signal >= 0.5
    assert r.verdict_hint == Verdict.DIRTY


def test_evidence_contains_required_fields():
    qs = [_mathq(i) for i in range(1, 25)]
    r = paraphrase_stress_test(
        _CleanContentModel(), _spec(name="gsm8k", fmt="math_cot"), qs,
        n_paraphrases=3, min_samples=20, seed=1,
    )
    ev = r.evidence
    for k in (
        "n_input", "n_valid", "n_scorable", "n_paraphrases",
        "paraphraser", "seed",
        "acc_original", "acc_paraphrases", "acc_worst_case",
        "acc_mean_paraphrases", "worst_case_drop", "mean_drop",
        "alpha_dirty", "alpha_suspect",
    ):
        assert k in ev, f"missing evidence key: {k}"
    assert len(ev["acc_paraphrases"]) == 3
    assert ev["worst_case_drop"] == pytest.approx(r.signal)


# ----------------------------- cache ----------------------------- #


def test_cache_writes_and_reads(tmp_path: Path):
    """第一次跑写 cache；第二次跑对同 seed / n / paraphraser 应命中。"""
    qs = [_mathq(i) for i in range(1, 25)]
    cache_dir = tmp_path / "para_cache"

    # 首跑
    r1 = paraphrase_stress_test(
        _CleanContentModel(), _spec(name="gsm8k", fmt="math_cot"), qs,
        n_paraphrases=3, min_samples=20, seed=42, cache_dir=cache_dir,
    )
    assert r1.prerequisites_met

    # 缓存文件确实生成
    files = list(cache_dir.glob("*.json"))
    assert len(files) == 1
    stored = json.loads(files[0].read_text(encoding="utf-8"))
    assert len(stored) == r1.evidence["n_scorable"]
    for q in qs:
        assert q.id in stored
        assert len(stored[q.id]) == 3

    # 篡改 cache 内容：把所有改写替换成原 prompt——这样如果 rewrite 走的是 cache，
    # 那么"改写=原文"，acc 应与原始一致；如果没走 cache，重新采样会再产生扰动。
    tampered = {q.id: [q.prompt] * 3 for q in qs}
    files[0].write_text(json.dumps(tampered), encoding="utf-8")

    # 二跑同参数
    r2 = paraphrase_stress_test(
        _CleanContentModel(), _spec(name="gsm8k", fmt="math_cot"), qs,
        n_paraphrases=3, min_samples=20, seed=42, cache_dir=cache_dir,
    )
    assert r2.prerequisites_met
    # 改写=原 prompt 时，clean model 的 acc_worst_case 应等于 acc_original
    assert r2.evidence["acc_worst_case"] == pytest.approx(r2.evidence["acc_original"])
    assert r2.signal == pytest.approx(0.0)


def test_cache_key_includes_seed_and_n(tmp_path: Path):
    qs = [_mathq(i) for i in range(1, 25)]
    cache_dir = tmp_path / "para_cache"

    paraphrase_stress_test(
        _CleanContentModel(), _spec(name="gsm8k", fmt="math_cot"), qs,
        n_paraphrases=3, min_samples=20, seed=1, cache_dir=cache_dir,
    )
    paraphrase_stress_test(
        _CleanContentModel(), _spec(name="gsm8k", fmt="math_cot"), qs,
        n_paraphrases=3, min_samples=20, seed=2, cache_dir=cache_dir,
    )
    paraphrase_stress_test(
        _CleanContentModel(), _spec(name="gsm8k", fmt="math_cot"), qs,
        n_paraphrases=5, min_samples=20, seed=1, cache_dir=cache_dir,
    )
    # 3 组独立 cache 文件
    files = list(cache_dir.glob("*.json"))
    assert len(files) == 3
