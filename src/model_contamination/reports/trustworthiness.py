"""Benchmark 可信度报告（§6.1，对外发布用）。

## 红线

**positive control 未到位前不出绝对红/黄/绿裁决**（CLAUDE.md 红线 + calibration-findings-2026-06-29）。
默认 `aggregate_verdict` 走 ranked-list 模式：只统计强/弱信号阳性数，`Verdict.SUSPECT/DIRTY`
以此为相对排名意义；calibration 完成后 caller 传 `calibrated=True` 才启用论文 default 阈值的
绝对裁决语义。

## 信号分档（按对内部模型 + 数据可见场景的独立价值）

- STRONG_SIGNALS：数据层无法/难以查到的独立方法学
    - `codec`：同分布 context 反事实，当前验证最完整；仍需 positive control 校准
    - `spv_mia`：跨 checkpoint MIA，适合有同源 base/reference 的 SFT 场景
    - `paraphrase`：数学 CoT 的改写压力测试，覆盖部分数据层 n-gram miss 场景

- WEAK_SIGNALS：内部模型 + 数据可见时数据层已能覆盖，模型层辅助验证
    - `perm_option`：MC 位置记忆，数据层能 exact match（见 [[project-scope-internal-model]]）
    - `mink_plus_plus`：pretrain 阶段泄露，pretrain 数据层扫描已覆盖大部分
    - `perm_option`：多选题选项排列异常，适合 MC 专用场景

**MemLens 已删**（Session 12，2026-07-01）：论文的"早期层 shortcut"假设在
本项目场景（Qwen3-1.7B + 200 题 × 5 副本 × 10 epoch）下不成立——前 20 层
target prob 全 <1e-3，唯一区分层是 L28（末层 lm_head 记忆），信号本质与
`_score_multiple_choice` 的 mean-logprob 打分和 SPV-MIA 高度重叠，独立价值
为零。见 memlens-hypothesis-check-2026-07-01 记忆记录。

排序键：strong 阳性数 > 弱阳性数 > raw signal 强度。
"""

from __future__ import annotations

from model_contamination.types import (
    BenchmarkVerdict,
    DetectionResult,
    Stage,
    Verdict,
)

STRONG_SIGNALS = {
    "codec",
    "spv_mia",
    "paraphrase",
}
WEAK_SIGNALS = {
    "perm_option",
    "mink_plus_plus",
}


def _count_positives(results: list[DetectionResult]) -> tuple[int, int, int]:
    """返回 (strong_dirty_or_suspect, weak_dirty_or_suspect, inconclusive_count)。

    阳性口径含 DIRTY + SUSPECT，因为 Phase 1 没 positive control 时二者都是"相对可疑"。
    caller 在 calibrated=True 时可以另外只数 DIRTY。
    """
    strong_pos = 0
    weak_pos = 0
    inconclusive = 0
    for r in results:
        if not r.prerequisites_met or r.verdict_hint == Verdict.INCONCLUSIVE:
            inconclusive += 1
            continue
        positive = r.verdict_hint in (Verdict.DIRTY, Verdict.SUSPECT)
        if r.method in STRONG_SIGNALS and positive:
            strong_pos += 1
        elif r.method in WEAK_SIGNALS and positive:
            weak_pos += 1
    return strong_pos, weak_pos, inconclusive


def aggregate_verdict(
    benchmark: str,
    stage: Stage,
    results: list[DetectionResult],
    calibrated: bool = False,
    red_min_strong: int = 2,
    yellow_min_strong: int = 1,
) -> BenchmarkVerdict:
    """跨方法综合裁决单 benchmark。

    默认 `calibrated=False`：Verdict 是相对排名标签（Phase 1 语义）。
    仅当 positive control 校准完成、caller 显式传 `calibrated=True`，
    才把 Verdict 当作论文 default 阈值下的绝对裁决。
    """
    strong_pos, weak_pos, inconclusive = _count_positives(results)
    total = len(results)

    if total > 0 and inconclusive == total:
        return BenchmarkVerdict(
            benchmark=benchmark,
            stage=stage,
            verdict=Verdict.INCONCLUSIVE,
            strong_signals_positive=0,
            weak_signals_positive=0,
            results=results,
            summary="前置条件全部缺失或 signal 不足，无法裁决",
        )

    if strong_pos >= red_min_strong:
        verdict = Verdict.DIRTY
    elif strong_pos >= yellow_min_strong or weak_pos >= 2:
        verdict = Verdict.SUSPECT
    else:
        verdict = Verdict.CLEAN

    if calibrated:
        prefix = {Verdict.DIRTY: "🔴 红", Verdict.SUSPECT: "🟡 黄", Verdict.CLEAN: "🟢 绿"}
        summary = (
            f"{prefix[verdict]}：{strong_pos} 强 + {weak_pos} 弱信号阳性"
            f"（{inconclusive}/{total} 前置缺失）"
        )
    else:
        summary = (
            f"[相对排名] {strong_pos} 强 + {weak_pos} 弱信号阳性"
            f"（{inconclusive}/{total} 前置缺失，positive control 未到位不出绝对裁决）"
        )

    return BenchmarkVerdict(
        benchmark=benchmark,
        stage=stage,
        verdict=verdict,
        strong_signals_positive=strong_pos,
        weak_signals_positive=weak_pos,
        results=results,
        summary=summary,
    )


def _sortkey_signal_magnitude(r: DetectionResult) -> float:
    if r.signal is None:
        return 0.0
    return abs(r.signal)


def rank_by_suspicion(verdicts: list[BenchmarkVerdict]) -> list[BenchmarkVerdict]:
    """按可疑度降序返回新 list，原 list 不变。

    排序键（依次）：
    1. 强信号阳性数（降序）
    2. 弱信号阳性数（降序）
    3. 最大原始 signal 绝对值（降序，粗略衡量证据强度）
    4. benchmark 名（升序，稳定）
    """
    def key(v: BenchmarkVerdict) -> tuple:
        max_abs_signal = max(
            (_sortkey_signal_magnitude(r) for r in v.results),
            default=0.0,
        )
        return (
            -v.strong_signals_positive,
            -v.weak_signals_positive,
            -max_abs_signal,
            v.benchmark,
        )

    return sorted(verdicts, key=key)


def render_trustworthiness_report(
    verdicts: list[BenchmarkVerdict],
    calibrated: bool = False,
) -> str:
    """渲染对外可读的可信度报告（markdown ranked list）。

    Phase 1 默认输出 ranked list + 各方法 raw signal，不出绝对红/黄/绿裁决。
    `calibrated=True` 时表头切到"可信度报告（绝对裁决）"，其余同。
    """
    if not verdicts:
        return "# Benchmark 可信度报告\n\n（无数据）\n"

    ranked = rank_by_suspicion(verdicts)
    lines: list[str] = []
    title = "Benchmark 可信度报告"
    if calibrated:
        lines.append(f"# {title}（绝对裁决）\n")
    else:
        lines.append(f"# {title}（相对排名 · Phase 1）\n")
        lines.append(
            "> **口径**：positive control 未到位，本报告只输出按可疑度排序的 ranked list "
            "+ 每方法原始 signal。Verdict 列请当作相对排名标签，不是绝对红/黄/绿裁决。\n"
        )

    lines.append("## 排名总表\n")
    lines.append("| 排名 | benchmark | stage | verdict | 强阳性 | 弱阳性 | 摘要 |")
    lines.append("|---:|---|---|---|---:|---:|---|")
    for i, v in enumerate(ranked, start=1):
        lines.append(
            f"| {i} | `{v.benchmark}` | {v.stage.value} | {v.verdict.value} | "
            f"{v.strong_signals_positive} | {v.weak_signals_positive} | {v.summary} |"
        )

    lines.append("\n## 各 benchmark 方法级细节\n")
    for i, v in enumerate(ranked, start=1):
        lines.append(f"### {i}. `{v.benchmark}` ({v.stage.value})\n")
        lines.append(f"- Verdict：**{v.verdict.value}** — {v.summary}")
        lines.append("")
        lines.append("| 方法 | 分档 | signal | hint | 前置 | 错误 |")
        lines.append("|---|---|---:|---|:-:|---|")
        for r in v.results:
            tier = (
                "strong" if r.method in STRONG_SIGNALS
                else "weak" if r.method in WEAK_SIGNALS
                else "other"
            )
            sig_str = f"{r.signal:.4f}" if r.signal is not None else "—"
            ok = "✓" if r.prerequisites_met else "✗"
            err = (r.error or "").replace("|", "\\|")
            lines.append(
                f"| `{r.method}` | {tier} | {sig_str} | {r.verdict_hint.value} | {ok} | {err} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"
