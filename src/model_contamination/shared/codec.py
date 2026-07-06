"""CoDeC — Contamination Detection via Context (Zawalski et al., arXiv:2510.27055).

原理：对候选数据集 D 的每个样本 x，比较模型在两种情况下对 x 的平均 log-likelihood：
    baseline    : 无 in-context 样本，仅以 "\\n\\n" 作 prefix 算 x 各 token 的 avg logp
                  （用分隔符而非空串，避免无 BOS 的 tokenizer 对空 prompt 返回 0 token）
    in-context  : 从 D∖{x} 随机采 n 个样本，用 "\\n\\n" 连接 prepend 到 x 前，
                  仍只在 x 自身 token 上算 avg logp
    Δ(x) = logprob_in-context(x) − logprob_baseline(x)

直觉：模型见过的数据，额外同分布 context 无新增信息且会扰动 memorization →
置信度下降（Δ<0）；未见数据像 few-shot 一样受益 → 置信度上升（Δ>0）。
数据集分数 S_CoDeC = Δ<0 样本的比例，∈[0,1]，越高越可疑。

接口契约：
    输入 model (灰盒，需 logprobs) + benchmark + 已加载 questions
    输出 DetectionResult(method="codec", signal=frac_negative ∈ [0,1])
    signal > 0.80 → DIRTY（论文 §A.5 经验阈值，未 calibrate）；0.60–0.80 → SUSPECT

输入要求：灰盒，只需 model.logprobs(prompt, completion)（每 token log p）。
    只看 Δ 的符号，与 logit 尺度无关 → 天然模型无关。

信号含义：signal = 数据集中「加 context 反而降低置信度」的样本比例。
    这是「对 memorized 先验的依赖度」，不等价于严格训练成员；训练在
    改写 / 蒸馏 / 同分布数据上也会拉高（论文 §A.5.4）。

已知失效场景（论文 §C.7 / §A.3.2）：
    - 高度重复数据集（同一文本重复多次）→ 分数被人为压低到 0。
    - 多个不相关来源混合 → context 无关，分数被人为抬高到 ~50%（如 MMLU 多主题）。
    - 极端多样、样本间无共享结构的数据 → 未见也可能到 60%。
    - 格式标签（Question:/Answer: 等）会干扰分数；suspect 文本应尽量用
      「训练时一定原样出现」的纯文本（这里默认只取题面 prompt）。
    ⇒ 中间区间分数须跨多个模型对照解读，不能单模型下绝对裁决（对齐仓库红线）。

论文默认：n_context=1, n_seeds=5, 忽略 x 前 10 个 token, 每数据集 ~1000 样本。
"""

from __future__ import annotations

import numpy as np

from model_contamination.models.base import Capability, ModelInterface
from model_contamination.types import (
    BenchmarkQuestion,
    BenchmarkSpec,
    DetectionResult,
    Stage,
    Verdict,
)

_SEPARATOR = "\n\n"


def codec_detect(
    model: ModelInterface,
    benchmark: BenchmarkSpec,
    questions: list[BenchmarkQuestion],
    *,
    n_context: int = 1,
    n_seeds: int = 5,
    skip_first_tokens: int = 10,
    dirty_threshold: float = 0.80,
    suspect_threshold: float = 0.60,
    min_samples: int = 100,
    seed: int = 42,
) -> DetectionResult:
    """对给定 benchmark 跑 CoDeC。

    n_context: 每次 prepend 的 context 样本数（论文默认 1）。
    n_seeds:   随机 context 有方差，对 n_seeds 个 seed 的 Δ 取平均（论文默认 5）。
    skip_first_tokens: 忽略 x 前若干 token，消除 text 切换的边界效应（论文默认 10）。
    min_samples: 论文称 100 样本即稳定；低于此仍跑但在 evidence 标注低置信。

    每样本需要 (1 + n_seeds) 次 logprobs 前向；数据集分数是 Δ<0 的比例。
    """
    stage = Stage(model.stage_tag)

    if not model.supports(Capability.LOGPROBS):
        return DetectionResult(
            method="codec", stage=stage, benchmark=benchmark.name,
            signal=None, verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            error=f"{model.name} does not support logprobs; CoDeC requires logprobs.",
        )

    if len(questions) < 2:
        return DetectionResult(
            method="codec", stage=stage, benchmark=benchmark.name,
            signal=None, verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            error=f"CoDeC needs >= 2 samples to sample context, got {len(questions)}.",
        )

    texts = [_codec_text(q) for q in questions]
    rng = np.random.default_rng(seed)

    deltas: list[float] = []
    n_skipped = 0
    for i, x in enumerate(texts):
        # baseline 用 _SEPARATOR 作 prefix（而非空串）：无 in-context 信息，但避免
        # 无 BOS 的 tokenizer（如 Pythia/gpt_neox）对空 prompt 返回 0 token 导致
        # logprobs 崩溃；且与 in-context 版「x 前紧接 \n\n」的分词对齐，Δ 更可比。
        base_lp = _avg_logp(model, prompt=_SEPARATOR, completion=x, skip=skip_first_tokens)
        if base_lp is None:  # x 的有效 token 数 <= skip_first_tokens
            n_skipped += 1
            continue

        seed_deltas: list[float] = []
        for _ in range(n_seeds):
            ctx = _sample_context(texts, exclude=i, n=n_context, rng=rng)
            prompt = _SEPARATOR.join(ctx) + _SEPARATOR
            ic_lp = _avg_logp(model, prompt=prompt, completion=x, skip=skip_first_tokens)
            if ic_lp is not None:
                seed_deltas.append(ic_lp - base_lp)
        if not seed_deltas:
            n_skipped += 1
            continue
        deltas.append(float(np.mean(seed_deltas)))

    if not deltas:
        return DetectionResult(
            method="codec", stage=stage, benchmark=benchmark.name,
            signal=None, verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            error="All samples too short after skip_first_tokens; no CoDeC score.",
        )

    deltas_arr = np.asarray(deltas, dtype=np.float64)
    frac_negative = float(np.mean(deltas_arr < 0.0))

    if frac_negative >= dirty_threshold:
        verdict = Verdict.DIRTY
    elif frac_negative >= suspect_threshold:
        verdict = Verdict.SUSPECT
    else:
        verdict = Verdict.CLEAN

    return DetectionResult(
        method="codec", stage=stage, benchmark=benchmark.name,
        signal=frac_negative, verdict_hint=verdict, prerequisites_met=True,
        evidence={
            "n_samples": len(deltas),
            "n_skipped": n_skipped,
            "n_context": n_context,
            "n_seeds": n_seeds,
            "skip_first_tokens": skip_first_tokens,
            "mean_delta": float(deltas_arr.mean()),
            "frac_negative": frac_negative,
            "low_confidence": len(deltas) < min_samples,
            # positive control 未到位：阈值来自论文经验值，非本仓库 calibration
            "threshold_source": "paper_A.5_uncalibrated",
        },
    )


def _codec_text(q: BenchmarkQuestion) -> str:
    """CoDeC 的 suspect 文本：只取题面纯文本。

    论文 §A.3.2：suspect 数据应聚焦「训练时一定原样出现」的部分，避免
    Question:/Answer: 等格式标签人为改变分数。对纯文本 benchmark（如 Pile
    子集切 chunk），prompt 字段本身就是原文，直接用。
    """
    return q.prompt


def _sample_context(
    texts: list[str], *, exclude: int, n: int, rng: np.random.Generator
) -> list[str]:
    """从 texts∖{texts[exclude]} 随机采 n 个样本（无放回，n 超量时截断）。"""
    candidates = [j for j in range(len(texts)) if j != exclude]
    k = min(n, len(candidates))
    idx = rng.choice(len(candidates), size=k, replace=False)
    return [texts[candidates[j]] for j in idx]


def _avg_logp(
    model: ModelInterface, *, prompt: str, completion: str, skip: int
) -> float | None:
    """completion 各 token 的 avg log p，忽略前 skip 个 token。

    有效 token 数 <= skip 时返回 None（样本过短，无法给出稳定信号）。
    baseline 与 in-context 都调此函数，返回的都是 completion（=x）自身
    token 的 logp，天然对齐可比。
    """
    lp = model.logprobs(prompt=prompt, completion=completion)
    if lp.shape[0] <= skip:
        return None
    return float(np.mean(lp[skip:]))
