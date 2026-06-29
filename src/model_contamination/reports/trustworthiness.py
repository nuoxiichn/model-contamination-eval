"""Benchmark 可信度报告（§6.1，对外发布用）。

裁决规则：
    🔴 红：≥2 个强信号阳性 → 内部跑但不对外公布
    🟡 黄：仅 1 个强信号或多个弱信号阳性 → 同时披露 worst-case 与原始分数
    🟢 绿：全部正常 → 正常公布
"""

from __future__ import annotations

from model_contamination.types import (
    BenchmarkVerdict,
    DetectionResult,
    Stage,
    Verdict,
)

STRONG_SIGNALS = {"oren", "family_diff", "canary", "paraphrase", "spv_mia"}
WEAK_SIGNALS = {"mink_plus_plus", "perm_option", "log_prober", "ts_guessing"}


def render_trustworthiness_report(
    verdicts: list[BenchmarkVerdict],
) -> str:
    """渲染对外可读的可信度报告（markdown 格式）。"""
    raise NotImplementedError("Phase 1 TODO: 实装报告渲染")


def aggregate_verdict(
    benchmark: str,
    stage: Stage,
    results: list[DetectionResult],
    red_min_strong: int = 2,
    yellow_min: int = 1,
) -> BenchmarkVerdict:
    """根据多方法 DetectionResult 综合裁决单 benchmark。"""
    strong_pos = sum(
        1 for r in results
        if r.method in STRONG_SIGNALS and r.verdict_hint == Verdict.DIRTY
    )
    weak_pos = sum(
        1 for r in results
        if r.method in WEAK_SIGNALS and r.verdict_hint == Verdict.DIRTY
    )

    if strong_pos >= red_min_strong:
        verdict = Verdict.DIRTY
        summary = f"🔴 红：{strong_pos} 个强信号阳性"
    elif strong_pos >= yellow_min or weak_pos >= 2:
        verdict = Verdict.SUSPECT
        summary = f"🟡 黄：{strong_pos} 强 + {weak_pos} 弱信号阳性，需人工复核"
    elif all(not r.prerequisites_met for r in results):
        verdict = Verdict.INCONCLUSIVE
        summary = "前置条件全部缺失，无法裁决"
    else:
        verdict = Verdict.CLEAN
        summary = "🟢 绿：全部信号正常"

    return BenchmarkVerdict(
        benchmark=benchmark,
        stage=stage,
        verdict=verdict,
        strong_signals_positive=strong_pos,
        weak_signals_positive=weak_pos,
        results=results,
        summary=summary,
    )
