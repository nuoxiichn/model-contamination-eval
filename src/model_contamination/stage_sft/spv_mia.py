"""SPV-MIA（Self-calibrated Probabilistic Variation MIA）。

Fu et al. NeurIPS 2024, [arXiv:2311.06062](https://arxiv.org/abs/2311.06062)。
源码参考：tsinghua-fib-lab/NeurIPS2024_SPV-MIA。

## 论文算法两个模块

### PDC（Practical Difficulty Calibration）
self-prompt approach：用 target LLM 生成 reference dataset Dself，
拿 Dself 微调一个 reference model θ̂，校准信号 = m_θ(x) − m_θ̂(x)。

### PVA（Probabilistic Variation Assessment）
membership 信号不是 raw probability，而是 x 是否处于 p_θ 的局部极大值。
二阶方向导的对称差分近似：

    ẽp_θ(x) ≈ (1/2N) · Σ_n [p_θ(x⁺_n) + p_θ(x⁻_n) − 2·p_θ(x)]

其中 x⁺_n 和 x⁻_n 是 x 的一对对称 paraphrase。member 样本的 ẽp_θ(x)
显著负（局部最大值附近，两边都低），non-member 接近 0。

## v1 对本项目场景的简化

我们的目标：LoRA SFT 场景内部模型 GT（同源 base + SFT ckpt 都在手）。

| 论文模块 | v1 简化 | 理由 |
|---|---|---|
| PDC self-prompt 训 θ̂ | **直接用 base ckpt 当 reference** | LoRA 场景 base 是天然 reference；微调 self-prompt 模型工程量过大且 base 已经满足"无 member 信息"的需求 |
| PVA 对称扰动 x⁺/x⁻ 镜像对 | **one-sided neighbor**：只生成 x' | 严格镜像需 embedding 域算 antipode，实装复杂；single-side 仍是 neighbor attack，论文 Table 1 AUC≈0.62-0.65 |
| 语义域 T5 mask-fill | **word-level random substitution** | 避开 T5 vocab mismatch（数学/代码）+ 单测确定性 + 0 model 依赖 |

v1 公式（每条样本 x = (prompt, completion)，completion 是被训的"答案"部分）：

    pv_m(x) = mean_n [log p_m(x'_n) − log p_m(x)]
    score(x) = pv_target(x) − pv_reference(x)

x'_n 是把 completion 的 mask_ratio 比例的"词"（whitespace 切）随机替换。

member x（被 target 训过、被 reference 没训过）：log p_target(x) 高、log p_target(x'_n) 低 →
pv_target(x) 很负；log p_reference(x) 与 log p_reference(x'_n) 差不多 → pv_reference(x) 接近 0；
**Δpv(x) << 0**。non-member：两个模型 pv 都接近 0，Δpv ≈ 0。

→ 把 `−score` 当 membership probability 的代理（越大越像 member）。

## v1 不做（TODO）

- 完整 self-prompt 训练（需 LoRA 微调流水线 + reference dataset 生成）
- 对称镜像扰动 x⁺/x⁻ 配对
- T5 / target.generate 做语义 paraphrase（v2 增加 `paraphraser="self"` 分支）

## 接口契约（与 stub 一致 + 与 mink_plus_plus 对齐）

输入 target_model + reference_model（都需 LOGPROBS）+ spec + questions [+ control_questions]。
输出 DetectionResult(method="spv_mia", signal=AUC 或 mean_Δpv)。
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from model_contamination.models.base import Capability, ModelInterface
from model_contamination.types import (
    BenchmarkQuestion,
    BenchmarkSpec,
    DetectionResult,
    Stage,
    Verdict,
)

# AUC 阈值与 mink_plus_plus 同套（一致性）
_AUC_DIRTY = 0.70
_AUC_SUSPECT = 0.60

# 最小可 paraphrase 的 completion 词数。少于此触发降级（MC 单字母答案直接 INCONCLUSIVE）
_MIN_COMPLETION_WORDS = 4

# 默认 word pool（高频英文词 + 标点）。用于 word-level random substitution。
# 长度 ~100 足以让 paraphrase 不退化到"全空"或"全相同"。
_DEFAULT_WORD_POOL = ["the", "a", "an", "of", "and", "or", "to", "in", "on", "at", "by", "for", "from", "with", "as", "is", "are", "was", "were", "be", "been", "being", "this", "that", "these", "those", "it", "its", "they", "them", "their", "he", "she", "him", "her", "his", "hers", "we", "us", "our", "ours", "you", "your", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "however", "therefore", "because", "although", "since", "while", "when", "where", "why", "how", "what", "which", "who", "if", "then", "than", "such", "so", "but", "yet", "still", "not", "no", "yes", "maybe", "perhaps", "possibly", "good", "bad", "small", "large", "new", "old", "high", "low", "first", "last", "same", "different", ".", ",", ";", ":", "?", "!", "-"]

Paraphraser = Literal["random"]  # v2 add "self" / "t5"


def spv_mia(
    target_model: ModelInterface,
    reference_model: ModelInterface,
    benchmark: BenchmarkSpec,
    questions: list[BenchmarkQuestion],
    *,
    n_neighbors: int = 10,
    mask_ratio: float = 0.20,
    control_questions: list[BenchmarkQuestion] | None = None,
    min_samples: int = 30,
    paraphraser: Paraphraser = "random",
    seed: int = 42,
) -> DetectionResult:
    """SPV-MIA v1 主入口。

    target_model: SFT 后的模型（被检 ckpt）
    reference_model: 同源 base（LoRA 场景：不加 adapter）。stub 文档里说"上一版本
        SFT ckpt"也行，但 v1 默认推荐 base——它对 SFT 数据完全 unseen，差分干净。
    n_neighbors: 每条样本生成多少 paraphrase（论文 default 较大；v1 设 10 以平衡精度与延迟）
    mask_ratio: paraphrase 时把 completion 的多少比例词替换为 random pool 词
    control_questions: 同分布、假定未见过的对照集。给了 → AUC 模式；不给 → mean_only。
    min_samples: 主集最小题数；低于则 INCONCLUSIVE。
    paraphraser: v1 仅 "random"；v2 加 "self" / "t5"。
    seed: paraphrase 随机性的种子（确保 reproducibility）

    signal 语义：
        AUC 模式：二分类 AUC(−Δpv on target_questions vs −Δpv on control_questions)；
            >=0.70 DIRTY / >=0.60 SUSPECT / else CLEAN。
        mean_only 模式：mean(Δpv)；verdict_hint INCONCLUSIVE（无对照不能 AUC）。
    """
    stage = Stage(target_model.stage_tag)

    if not target_model.supports(Capability.LOGPROBS):
        return DetectionResult(
            method="spv_mia", stage=stage, benchmark=benchmark.name,
            signal=None, verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            error=f"{target_model.name} does not support LOGPROBS; SPV-MIA requires per-token log p on target.",
        )
    if not reference_model.supports(Capability.LOGPROBS):
        return DetectionResult(
            method="spv_mia", stage=stage, benchmark=benchmark.name,
            signal=None, verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            error=f"{reference_model.name} does not support LOGPROBS; SPV-MIA requires per-token log p on reference.",
        )

    if not 0.0 < mask_ratio < 1.0:
        return DetectionResult(
            method="spv_mia", stage=stage, benchmark=benchmark.name,
            signal=None, verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            error=f"mask_ratio must be in (0, 1), got {mask_ratio}.",
        )
    if n_neighbors < 1:
        return DetectionResult(
            method="spv_mia", stage=stage, benchmark=benchmark.name,
            signal=None, verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            error=f"n_neighbors must be >= 1, got {n_neighbors}.",
        )

    if len(questions) < min_samples:
        return DetectionResult(
            method="spv_mia", stage=stage, benchmark=benchmark.name,
            signal=None, verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            error=f"Need >= {min_samples} samples, got {len(questions)}.",
        )

    rng = np.random.default_rng(seed)

    target_scores = _score_questions(
        target_model, reference_model, questions, n_neighbors, mask_ratio, paraphraser, rng,
    )
    n_input = len(questions)
    n_finite = int(np.isfinite(target_scores).sum())
    target_scores = _filter_finite(target_scores)
    if len(target_scores) < min_samples // 2:
        return DetectionResult(
            method="spv_mia", stage=stage, benchmark=benchmark.name,
            signal=None, verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            evidence={
                "mode": "degraded",
                "degraded_reason": "too_few_finite_scores",
                "n_input": n_input,
                "n_target": n_finite,
                "n_neighbors": n_neighbors,
                "mask_ratio": mask_ratio,
                "paraphraser": paraphraser,
                "seed": seed,
                "note": (
                    "Likely cause: completion too short (< _MIN_COMPLETION_WORDS=4) "
                    "for paraphrase, e.g. MC single-letter answers or 1-word numeric. "
                    "math_cot loaders should populate q.full_answer with the CoT."
                ),
            },
            error=f"Only {n_finite} finite scores out of {n_input} questions (likely too-short completions).",
        )

    if control_questions is None or len(control_questions) == 0:
        return DetectionResult(
            method="spv_mia", stage=stage, benchmark=benchmark.name,
            signal=float(np.mean(target_scores)),
            verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=True,
            evidence={
                "mode": "mean_only",
                "n_neighbors": n_neighbors,
                "mask_ratio": mask_ratio,
                "paraphraser": paraphraser,
                "seed": seed,
                "n_target": int(len(target_scores)),
                "target_mean_delta_pv": float(np.mean(target_scores)),
                "target_median_delta_pv": float(np.median(target_scores)),
                "target_std_delta_pv": float(np.std(target_scores, ddof=1)) if len(target_scores) > 1 else 0.0,
                "target_delta_pv": target_scores.tolist(),
                "note": (
                    "No control set; AUC unavailable. mean(Δpv) only as relative cross-ckpt signal. "
                    "Member x should have Δpv << 0 (target memorized more than reference)."
                ),
            },
            error=None,
        )

    control_scores = _score_questions(
        target_model, reference_model, control_questions, n_neighbors, mask_ratio, paraphraser, rng,
    )
    n_ctrl_input = len(control_questions)
    n_ctrl_finite = int(np.isfinite(control_scores).sum())
    control_scores = _filter_finite(control_scores)
    if len(control_scores) < 2:
        return DetectionResult(
            method="spv_mia", stage=stage, benchmark=benchmark.name,
            signal=None, verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            evidence={
                "mode": "degraded",
                "degraded_reason": "too_few_finite_control_scores",
                "n_input": n_ctrl_input,
                "n_control": n_ctrl_finite,
                "n_target": int(len(target_scores)),
                "target_mean_delta_pv": float(np.mean(target_scores)),
                "n_neighbors": n_neighbors,
                "mask_ratio": mask_ratio,
                "paraphraser": paraphraser,
                "seed": seed,
            },
            error=f"Only {n_ctrl_finite} finite control scores; need >= 2.",
        )

    # Member 标签：target_questions 是疑似 member，control_questions 是 non-member。
    # Δpv 越负 → 越像 member（target 在 x 上局部最大值越尖）。
    # AUC 算"target 的 −Δpv 是否系统性大于 control 的 −Δpv"。
    auc = _auc_target_vs_control(-target_scores, -control_scores)
    if auc >= _AUC_DIRTY:
        verdict = Verdict.DIRTY
    elif auc >= _AUC_SUSPECT:
        verdict = Verdict.SUSPECT
    else:
        verdict = Verdict.CLEAN

    return DetectionResult(
        method="spv_mia", stage=stage, benchmark=benchmark.name,
        signal=float(auc), verdict_hint=verdict, prerequisites_met=True,
        evidence={
            "mode": "auc",
            "n_neighbors": n_neighbors,
            "mask_ratio": mask_ratio,
            "paraphraser": paraphraser,
            "seed": seed,
            "n_target": int(len(target_scores)),
            "n_control": int(len(control_scores)),
            "target_mean_delta_pv": float(np.mean(target_scores)),
            "control_mean_delta_pv": float(np.mean(control_scores)),
            "delta_of_delta_pv": float(np.mean(target_scores) - np.mean(control_scores)),
            "target_delta_pv": target_scores.tolist(),
            "control_delta_pv": control_scores.tolist(),
        },
        error=None,
    )


# ----------------------------- core scoring ----------------------------- #


def _score_questions(
    target_model: ModelInterface,
    reference_model: ModelInterface,
    questions: list[BenchmarkQuestion],
    n_neighbors: int,
    mask_ratio: float,
    paraphraser: Paraphraser,
    rng: np.random.Generator,
) -> np.ndarray:
    scores = np.empty(len(questions), dtype=np.float64)
    for i, q in enumerate(questions):
        scores[i] = _spv_sample_score(
            target_model, reference_model, q, n_neighbors, mask_ratio, paraphraser, rng,
        )
    return scores


def _spv_sample_score(
    target_model: ModelInterface,
    reference_model: ModelInterface,
    q: BenchmarkQuestion,
    n_neighbors: int,
    mask_ratio: float,
    paraphraser: Paraphraser,
    rng: np.random.Generator,
) -> float:
    """单样本 Δpv(x) 计算。

    completion 太短（< _MIN_COMPLETION_WORDS）→ paraphrase 无意义 → NaN（上层过滤）。
    """
    prompt, completion = _split_prompt_completion(q)
    if len(completion.split()) < _MIN_COMPLETION_WORDS:
        return float("nan")

    logp_target_x = _sum_logp(target_model, prompt, completion)
    logp_ref_x = _sum_logp(reference_model, prompt, completion)
    if not (np.isfinite(logp_target_x) and np.isfinite(logp_ref_x)):
        return float("nan")

    pv_target = 0.0
    pv_ref = 0.0
    n_valid = 0
    for _ in range(n_neighbors):
        x_prime = _paraphrase(completion, mask_ratio, paraphraser, rng)
        if x_prime == completion:
            continue
        logp_target_x_prime = _sum_logp(target_model, prompt, x_prime)
        logp_ref_x_prime = _sum_logp(reference_model, prompt, x_prime)
        if not (np.isfinite(logp_target_x_prime) and np.isfinite(logp_ref_x_prime)):
            continue
        pv_target += logp_target_x_prime - logp_target_x
        pv_ref += logp_ref_x_prime - logp_ref_x
        n_valid += 1

    if n_valid == 0:
        return float("nan")
    pv_target /= n_valid
    pv_ref /= n_valid
    return float(pv_target - pv_ref)


def _sum_logp(model: ModelInterface, prompt: str, completion: str) -> float:
    """sum_i log p_m(t_i | t_<i) over completion tokens。空 completion → NaN。

    用 sum 而非 mean 是为了与论文 p_θ(x) = Π p(t_i|t_<i) 对齐——pv 在 log 空间下是
    sum 的差，所以 mean 与 sum 不会影响 sign，但 sum 与论文公式直接对应。
    """
    lp = model.logprobs(prompt, completion)
    if lp.size == 0:
        return float("nan")
    return float(np.sum(lp))


def _split_prompt_completion(q: BenchmarkQuestion) -> tuple[str, str]:
    """SPV-MIA 信号在 answer tokens 上（SFT loss 也只算 response），故：
        prompt    = "Question: ...\\nAnswer: "
        completion = 答案文本

    math_cot 题型（GSM8K / MATH / MATH-500 等）：SFT 训的是完整 CoT+answer 全文
    （见 experiments/.../prep_bench_to_sft.py），所以 SPV-MIA 必须用 CoT 全文才能看到
    memorization。归一化 loader 已把 CoT 统一提到顶层 `q.full_answer`（各 benchmark
    上游列名不同，由各自 normalizer 映射），这里**只读这一个固定字段、不猜 raw key**。
    `full_answer` 为 None/空（该题型无 CoT）时才回退到 q.answer 作兼容。

    MC 题型 completion 是单字母（如 "B"），后续被 _MIN_COMPLETION_WORDS 过滤
    为 NaN，不参与 AUC。这是 SPV-MIA 在 MC 上的已知失效模式（短答案无法形成稳定邻居扰动）。
    """
    if q.format == "math_cot":
        cot = q.full_answer
        completion = cot if isinstance(cot, str) and cot.strip() else q.answer
        return f"Question: {q.prompt}\nAnswer: ", completion
    if q.format == "multiple_choice" and q.choices:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(q.choices))
        return f"Question: {q.prompt}\n{body}\nAnswer: ", q.answer
    return f"Question: {q.prompt}\nAnswer: ", q.answer


def _paraphrase(text: str, mask_ratio: float, paraphraser: Paraphraser, rng: np.random.Generator) -> str:
    """v1: word-level random substitution。

    把 text 按空白切词，随机 mask_ratio 比例的词位替换为 _DEFAULT_WORD_POOL 中的随机词。
    保留原文长度（词数不变），保留非被替换词位。
    """
    if paraphraser != "random":
        raise NotImplementedError(
            f"paraphraser={paraphraser!r} not implemented in v1; only 'random' supported. "
            "TODO v2: 'self' (target.generate) / 't5' (semantic mask-fill)."
        )
    words = text.split()
    n = len(words)
    if n < _MIN_COMPLETION_WORDS:
        return text
    n_mask = max(1, int(round(mask_ratio * n)))
    indices = rng.choice(n, size=n_mask, replace=False)
    for i in indices:
        words[i] = rng.choice(_DEFAULT_WORD_POOL)
    return " ".join(words)


def _filter_finite(arr: np.ndarray) -> np.ndarray:
    return arr[np.isfinite(arr)]


def _auc_target_vs_control(target: np.ndarray, control: np.ndarray) -> float:
    """二分类 AUC：target 标 1（疑似 member），control 标 0。

    与 mink_plus_plus._auc_target_vs_control 同一公式（Mann–Whitney U / n_t / n_c）。
    重复以避免私函数跨模块引用；几行代码不值得引入 shared util。
    """
    n_t = len(target)
    n_c = len(control)
    if n_t == 0 or n_c == 0:
        return 0.5
    wins = 0.0
    for t in target:
        wins += np.sum(t > control) + 0.5 * np.sum(t == control)
    return float(wins / (n_t * n_c))
