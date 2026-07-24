"""Paraphrase Stress Test（arXiv:2510.08616）。

## 原理

同义改写 prompt，观察准确率下降幅度。
    - 干净模型：识别的是语义 → 改写后 acc 基本不变
    - 污染模型：记住的是 prompt 的 token 序列 → 改写后 acc 显著下降

    signal = acc_orig − min_i(acc_paraphrase_i)   (worst-case drop, ∈ [-1, 1])

论文 Table 1：已知污染对照集上典型 gap 10–30%，clean 模型 <5%。

## 黑盒方法

只依赖 `Capability.GENERATE`；不需要 logprobs。

## 适用范围（v2 收窄）

**只用于 math_cot（生成式 CoT，如 GSM8K / MATH）**。MC 题型已确认结构性失效
（experiments/2026-06-27_sft_contam_gt/notes.md E0.1/E0.3），不再作为 MC benchmark
的检测信号，遇到 MC 为主的 benchmark 直接返回 `prerequisites_met=False` 标 skip。
定位：**低权重参考信号**，需 clean ckpt 对照解读，不单独定裁决。

## Rewriter 策略（v1: rule-based）

v1 保持"纯 Python、无外部依赖、确定性"，用一组表面语法变换制造多个 paraphrase：

1. synonym_replace       — 高频虚词/连词的同义/等价词替换（"however" ↔ "but"）
2. punctuation_shuffle   — 标点符号增删（多加一个 "." / 去掉多余空格）
3. case_perturb          — 大小写扰动（首字母大小写切换 / 全角空格插入）
4. hedge_wrap            — 前后加"Please answer:" / "Answer this question:" 等 wrapper
5. clause_reorder        — 简单句法：把 "if X then Y" 改成 "Y, if X" 之类

目标是让改写后的 prompt **语义等价**、只在表面 token 上变化。
若模型只是把原始 token 序列记忆了，这些扰动足以让准确率坍塌；语义理解的模型则不受影响。

**v1 已知失效场景**（如同 SPV-MIA 在 MC 上的失效）：
    - 极短 prompt（<3 词）— 没什么可扰动的
    - IFEval 类"指令遵循"题 — 改写会破坏指令本身语义（论文原文亦承认）

## v2 待实装（Phase 3）

`paraphraser` 参数已预留，未来切到：
    - "self": target.generate 一次自我 paraphrase（易 leak，需 second-model）
    - "llm":  外部 LLM 改写（GPT-4-mini / DeepSeek 等）

Phase 3 才做，v1 只 `paraphraser="rule"`。

## 缓存

改写文本按 `(benchmark_name, model_name, seed, n, paraphraser)` 索引缓存到
`cache_dir/paraphrase/{benchmark}_{model_name}_{seed}_{n}_{paraphraser}.json`。
重跑同参数不重新采样改写（决定性种子已保证同 seed 同 rewriter → 同输出，
但 IO cache 避免重启 Python 时的 latency）。

## 接口契约

    paraphrase_stress_test(model, benchmark, samples, *, ...) -> DetectionResult

- 前三位置参数 `(model, benchmark, samples)` 与项目其他方法一致。
- samples 内部是 `list[BenchmarkQuestion]`。
- 未满足 `Capability.GENERATE` → INCONCLUSIVE + prerequisites_met=False。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

import numpy as np

from model_contamination.models.base import Capability, ModelInterface
from model_contamination.shared.evaluator import (
    _score_math_cot,
    _score_multiple_choice,
)
from model_contamination.types import (
    BenchmarkQuestion,
    BenchmarkSpec,
    DetectionResult,
    Stage,
    Verdict,
)

Paraphraser = Literal["rule"]  # v2 add "self" / "llm"

# verdict 阈值：与 perm_option 一致，acc drop 幅度
_ALPHA_DIRTY = 0.20      # worst-case acc drop >= 20% → DIRTY
_ALPHA_SUSPECT = 0.05    # >= 5% → SUSPECT

_MIN_PROMPT_WORDS = 3    # 少于此，规则改写基本无作用；标 skip
_DEFAULT_CACHE_DIR = Path("cache/paraphrase")


# ----------------------------- 规则改写字典 ----------------------------- #

# 高频虚词 / 副词 / 连词的等价替换。左右可互换（双向）。
# 只包含语义几乎完全等价的 pair，避免语义漂移。
_SYNONYM_PAIRS: list[tuple[str, str]] = [
    ("however", "but"),
    ("therefore", "so"),
    ("because", "since"),
    ("although", "though"),
    ("also", "as well"),
    ("often", "frequently"),
    ("usually", "typically"),
    ("very", "quite"),
    ("many", "numerous"),
    ("few", "several"),
    ("small", "little"),
    ("large", "big"),
    ("begin", "start"),
    ("end", "finish"),
    ("show", "demonstrate"),
    ("use", "utilize"),
    ("help", "assist"),
    ("find", "locate"),
    ("get", "obtain"),
    ("make", "create"),
    ("in order to", "to"),
    ("in addition", "also"),
    ("for example", "for instance"),
    ("such as", "like"),
    ("Please", "Kindly"),
]

_HEDGE_WRAPPERS: list[tuple[str, str]] = [
    ("Question: ", ""),
    ("", "\nThink step by step."),
    ("Please answer this question. ", ""),
    ("Consider the following: ", ""),
    ("", " Provide your answer."),
]


# ----------------------------- 主入口 ----------------------------- #


def paraphrase_stress_test(
    model: ModelInterface,
    benchmark: BenchmarkSpec,
    samples: list[BenchmarkQuestion],
    *,
    n_paraphrases: int = 5,
    paraphraser: Paraphraser = "rule",
    worst_case_drop_threshold: float = _ALPHA_DIRTY,
    suspect_threshold: float = _ALPHA_SUSPECT,
    min_samples: int = 20,
    seed: int = 42,
    cache_dir: Path | str | None = None,
    max_tokens: int = 512,
) -> DetectionResult:
    """对 benchmark 跑 Paraphrase Stress Test。

    signal = acc_orig − min_i(acc_paraphrase_i)   (worst-case drop)

    Args:
        model: 被检 checkpoint。必须支持 Capability.GENERATE。
        benchmark: 静态 spec。
        samples: 归一化后的 BenchmarkQuestion 列表。
        n_paraphrases: 每题生成多少个改写。论文 5 已足够检出污染。
        paraphraser: v1 仅 "rule"；"self" / "llm" TODO Phase 3。
        worst_case_drop_threshold: DIRTY 阈值（默认 0.20，与 perm_option 一致）。
        suspect_threshold: SUSPECT 阈值（默认 0.05）。
        min_samples: 主集最小题数；低于则 INCONCLUSIVE。
        seed: rewriter 随机种子（+ cache key 一部分）。
        cache_dir: 改写文本磁盘缓存目录。None → cache/paraphrase/。
        max_tokens: math_cot 生成时的 max_tokens（透传 evaluator）。

    Returns:
        DetectionResult(method="paraphrase", ...)
    """
    stage = Stage(model.stage_tag)

    if not model.supports(Capability.GENERATE):
        return DetectionResult(
            method="paraphrase", stage=stage, benchmark=benchmark.name,
            signal=None, verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            error=f"{model.name} does not support GENERATE; paraphrase stress requires it.",
        )

    if paraphraser != "rule":
        return DetectionResult(
            method="paraphrase", stage=stage, benchmark=benchmark.name,
            signal=None, verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            error=(
                f"paraphraser={paraphraser!r} not implemented in v1; only 'rule' supported. "
                "TODO Phase 3: 'self' (target self-paraphrase) / 'llm' (external rewriter)."
            ),
        )

    if n_paraphrases < 1:
        return DetectionResult(
            method="paraphrase", stage=stage, benchmark=benchmark.name,
            signal=None, verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            error=f"n_paraphrases must be >= 1, got {n_paraphrases}.",
        )

    if len(samples) < min_samples:
        return DetectionResult(
            method="paraphrase", stage=stage, benchmark=benchmark.name,
            signal=None, verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            error=f"Need >= {min_samples} samples, got {len(samples)}.",
            evidence={"n_input": len(samples)},
        )

    # Paraphrase 只用于 math_cot（生成式 CoT）。MC 题型已确认结构性失效
    # （见 experiments/2026-06-27_sft_contam_gt/notes.md E0.1/E0.3：SFT ckpt 训成
    # "输出字母"，raw choice text 非训练分布，signal 被噪声淹没），不再作为 MC benchmark
    # 的检测信号。遇到 MC 为主的 benchmark 直接标 skip，而非跑出低信号再忽略。
    supported_formats = {"math_cot"}
    valid = [q for q in samples if q.format in supported_formats]
    if len(valid) < min_samples:
        n_mc = sum(1 for q in samples if q.format == "multiple_choice")
        reason = (
            f"Only {len(valid)} math_cot samples; need >= {min_samples}."
        )
        if n_mc >= len(samples) - len(valid):
            reason = (
                f"benchmark 以 multiple_choice 为主（{n_mc}/{len(samples)}）；"
                "paraphrase 对 MC 结构性失效，不作为 MC benchmark 的检测信号"
                f"（math_cot 有效样本仅 {len(valid)} < {min_samples}）。"
            )
        return DetectionResult(
            method="paraphrase", stage=stage, benchmark=benchmark.name,
            signal=None, verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            error=reason,
            evidence={"n_input": len(samples), "n_valid": len(valid), "n_mc": n_mc},
        )

    cache = _CacheHandle(
        cache_dir=Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR,
        benchmark_name=benchmark.name,
        model_name=model.name,
        seed=seed,
        n_paraphrases=n_paraphrases,
        paraphraser=paraphraser,
    )

    rng = np.random.default_rng(seed)

    # 生成改写。每题 n_paraphrases 个，写入/读自 cache。
    paraphrased_by_q: dict[str, list[str]] = {}
    n_skipped_short = 0
    for q in valid:
        cached = cache.get(q.id)
        if cached is not None and len(cached) == n_paraphrases:
            paraphrased_by_q[q.id] = cached
            continue
        if len(q.prompt.split()) < _MIN_PROMPT_WORDS:
            paraphrased_by_q[q.id] = []  # 短题跳过
            n_skipped_short += 1
            continue
        variants: list[str] = []
        for _ in range(n_paraphrases):
            variants.append(_rule_paraphrase(q.prompt, rng))
        paraphrased_by_q[q.id] = variants
        cache.put(q.id, variants)
    cache.flush()

    # 只保留有 >=1 个有效改写的题
    scored: list[BenchmarkQuestion] = [q for q in valid if paraphrased_by_q[q.id]]
    if len(scored) < min_samples // 2:
        return DetectionResult(
            method="paraphrase", stage=stage, benchmark=benchmark.name,
            signal=None, verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            error=(
                f"Only {len(scored)} samples have valid paraphrases "
                f"(too many short prompts <{_MIN_PROMPT_WORDS} words)."
            ),
            evidence={
                "n_input": len(samples),
                "n_valid": len(valid),
                "n_scorable": len(scored),
                "n_skipped_short": n_skipped_short,
            },
        )

    # 评测：原题 acc + n 个 paraphrase 各自的 acc
    print(
        f"[paraphrase] {benchmark.name} ({model.name}): "
        f"scoring {len(scored)} orig samples...",
        flush=True,
    )
    acc_orig = _score_batch(model, scored, max_tokens=max_tokens)
    print(f"[paraphrase]   acc_orig={acc_orig:.3f}", flush=True)

    acc_paras: list[float] = []
    for k in range(n_paraphrases):
        para_qs: list[BenchmarkQuestion] = []
        for q in scored:
            variants = paraphrased_by_q[q.id]
            # 若 <k+1 个 variants（几乎不发生，规则改写总产 n 个），用 wrap-around
            new_prompt = variants[k % len(variants)]
            para_qs.append(_swap_prompt(q, new_prompt))
        print(
            f"[paraphrase]   scoring paraphrase {k + 1}/{n_paraphrases}...",
            flush=True,
        )
        acc_k = _score_batch(model, para_qs, max_tokens=max_tokens)
        acc_paras.append(acc_k)
        print(f"[paraphrase]     acc={acc_k:.3f}", flush=True)

    worst_case_acc = min(acc_paras)
    mean_acc_paras = float(np.mean(acc_paras))
    signal = acc_orig - worst_case_acc

    if signal >= worst_case_drop_threshold:
        verdict = Verdict.DIRTY
    elif signal >= suspect_threshold:
        verdict = Verdict.SUSPECT
    else:
        verdict = Verdict.CLEAN

    return DetectionResult(
        method="paraphrase", stage=stage, benchmark=benchmark.name,
        signal=float(signal), verdict_hint=verdict, prerequisites_met=True,
        evidence={
            "n_input": len(samples),
            "n_valid": len(valid),
            "n_scorable": len(scored),
            "n_skipped_short": n_skipped_short,
            "n_paraphrases": n_paraphrases,
            "paraphraser": paraphraser,
            "seed": seed,
            "acc_original": float(acc_orig),
            "acc_paraphrases": [float(a) for a in acc_paras],
            "acc_worst_case": float(worst_case_acc),
            "acc_mean_paraphrases": mean_acc_paras,
            "worst_case_drop": float(signal),
            "mean_drop": float(acc_orig - mean_acc_paras),
            "alpha_dirty": worst_case_drop_threshold,
            "alpha_suspect": suspect_threshold,
        },
    )


# ----------------------------- 内部工具 ----------------------------- #


def _swap_prompt(q: BenchmarkQuestion, new_prompt: str) -> BenchmarkQuestion:
    """构造新 BenchmarkQuestion，只改 prompt，其他字段（answer/choices/...）保留。"""
    return BenchmarkQuestion(
        id=q.id, benchmark=q.benchmark, format=q.format,
        prompt=new_prompt, answer=q.answer,
        choices=q.choices, answer_index=q.answer_index,
        raw=q.raw,
    )


def _score_batch(
    model: ModelInterface,
    qs: list[BenchmarkQuestion],
    *,
    max_tokens: int,
) -> float:
    """dispatch by format，返回 accuracy ∈ [0, 1]。

    与 shared/evaluator.evaluate_accuracy 语义一致，但内联以避免多次遍历。
    每 5 题打一次 progress，避免 math_cot 长 generate 时看似 hang。
    """
    if not qs:
        return 0.0
    correct = 0
    n = len(qs)
    for i, q in enumerate(qs):
        if q.format == "multiple_choice":
            ok = _score_multiple_choice(model, q)
        elif q.format == "math_cot":
            ok = _score_math_cot(model, q, max_tokens=max_tokens)
        else:
            # valid 过滤已保证 format 落在 {mc, math_cot}；这里防御性 skip
            continue
        correct += int(bool(ok))
        if (i + 1) % 5 == 0 or (i + 1) == n:
            print(
                f"[paraphrase]       progress {i + 1}/{n} (correct={correct})",
                flush=True,
            )
    return correct / len(qs)


# ----------------------------- rule paraphraser ----------------------------- #


_WORD_RE = re.compile(r"\b\w+\b")


def _rule_paraphrase(text: str, rng: np.random.Generator) -> str:
    """规则式改写：从 5 类变换里随机采若干个应用。

    确保：
    - 至少应用一次变换（若可用）
    - 输出与输入不同（若可能）
    - 语义等价（只做同义词/标点/包装/大小写扰动）
    """
    ops = [
        _op_synonym_replace,
        _op_punctuation_shuffle,
        _op_case_perturb,
        _op_hedge_wrap,
        _op_clause_reorder,
    ]
    # 随机采 1-3 个操作
    n_ops = int(rng.integers(1, 4))
    chosen_idx = rng.choice(len(ops), size=n_ops, replace=False)
    out = text
    for i in chosen_idx:
        out = ops[i](out, rng)
    # 兜底：若无变化，强制加个空格 / 追加句号，保证 token 序列变了
    if out == text:
        out = out.rstrip() + "."
    return out


def _op_synonym_replace(text: str, rng: np.random.Generator) -> str:
    """随机挑一个能匹配上的 pair，做单次替换（保留大小写敏感的常见形式）。"""
    # 随机顺序遍历 pairs
    order = rng.permutation(len(_SYNONYM_PAIRS))
    for i in order:
        a, b = _SYNONYM_PAIRS[i]
        # 随机方向
        if rng.random() < 0.5:
            a, b = b, a
        # word-boundary 匹配（case-insensitive but 首字母跟随原来）
        pattern = re.compile(rf"\b{re.escape(a)}\b", re.IGNORECASE)
        m = pattern.search(text)
        if m:
            replacement = _match_case(m.group(0), b)
            return pattern.sub(replacement, text, count=1)
    return text


def _match_case(original: str, replacement: str) -> str:
    """让 replacement 大致跟随 original 的首字母大小写。"""
    if not original or not replacement:
        return replacement
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def _op_punctuation_shuffle(text: str, rng: np.random.Generator) -> str:
    """加或去掉一些无关标点/空格。"""
    kind = int(rng.integers(0, 4))
    if kind == 0:
        return text.rstrip(".!?") + "?" if text.rstrip().endswith("?") else text.rstrip() + "."
    if kind == 1:
        return re.sub(r"  +", " ", text)  # 多空格合并
    if kind == 2:
        return text.replace(", ", " , ", 1) if ", " in text else text
    return re.sub(r"([.!?])\s+", r"\1  ", text, count=1)  # 一次性双空格


def _op_case_perturb(text: str, rng: np.random.Generator) -> str:
    """把首个单词的首字母翻转大小写（"What" ↔ "what"）。"""
    if not text:
        return text
    words = text.split(" ", 1)
    head = words[0]
    if not head:
        return text
    # 翻转首字母大小写
    if head[0].isupper():
        new_head = head[0].lower() + head[1:]
    elif head[0].islower():
        new_head = head[0].upper() + head[1:]
    else:
        return text
    return new_head + (" " + words[1] if len(words) > 1 else "")


def _op_hedge_wrap(text: str, rng: np.random.Generator) -> str:
    """前后加/减 wrapper 语。"""
    prefix, suffix = _HEDGE_WRAPPERS[int(rng.integers(0, len(_HEDGE_WRAPPERS)))]
    return f"{prefix}{text}{suffix}"


def _op_clause_reorder(text: str, rng: np.random.Generator) -> str:
    """简单的 "If X, Y" ↔ "Y, if X" / 单句"A. B." 交换。

    保守：只在明确匹配 pattern 时才改，否则原样返回。
    """
    # "If X, Y." → "Y, if X."
    m = re.match(r"^If ([^,]+), (.+?)([.!?])?$", text)
    if m and rng.random() < 0.7:
        x, y, end = m.group(1), m.group(2), m.group(3) or ""
        return f"{y[0].upper()}{y[1:]}, if {x}{end}"
    # 单段两个句子交换（"A. B." → "B. A."）
    parts = re.split(r"(?<=[.!?])\s+", text)
    if len(parts) == 2 and rng.random() < 0.5:
        return f"{parts[1]} {parts[0]}"
    return text


# ----------------------------- Cache 处理 ----------------------------- #


class _CacheHandle:
    """按 (benchmark, model, seed, n, paraphraser) 一个 json，key=q.id → list[str]。

    load-once, put-in-memory, flush-once。避免每题一个文件的 IO overhead。
    """

    def __init__(
        self,
        *,
        cache_dir: Path,
        benchmark_name: str,
        model_name: str,
        seed: int,
        n_paraphrases: int,
        paraphraser: str,
    ) -> None:
        self.cache_dir = cache_dir
        safe_model = model_name.replace("/", "_").replace(":", "_")
        fname = (
            f"{benchmark_name}_{safe_model}_s{seed}_"
            f"n{n_paraphrases}_{paraphraser}.json"
        )
        self.path = cache_dir / fname
        self._data: dict[str, list[str]] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._data = {}

    def get(self, q_id: str) -> list[str] | None:
        v = self._data.get(q_id)
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            return v
        return None

    def put(self, q_id: str, variants: list[str]) -> None:
        self._data[q_id] = variants
        self._dirty = True

    def flush(self) -> None:
        if not self._dirty:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)
        self._dirty = False


__all__ = ["paraphrase_stress_test"]
