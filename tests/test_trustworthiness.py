"""trustworthiness.py 单测：多信号聚合 + ranked list + markdown 渲染。

覆盖：
- STRONG/WEAK 分类正确
- Phase 1 默认 calibrated=False → summary 显式标"相对排名"
- calibrated=True → 红/黄/绿 emoji 前缀
- inconclusive/prerequisites_met=False 的处理
- rank_by_suspicion 排序键
- render_trustworthiness_report markdown 结构合法
"""

from __future__ import annotations

from model_contamination.reports.trustworthiness import (
    STRONG_SIGNALS,
    WEAK_SIGNALS,
    aggregate_verdict,
    rank_by_suspicion,
    render_trustworthiness_report,
)
from model_contamination.types import DetectionResult, Stage, Verdict


def _mk(method: str, signal: float | None, hint: Verdict, ok: bool = True) -> DetectionResult:
    return DetectionResult(
        method=method,
        stage=Stage.SFT,
        benchmark="mmlu-pro",
        signal=signal,
        verdict_hint=hint,
        prerequisites_met=ok,
    )


# ---------------- STRONG / WEAK 分类 ----------------


def test_canary_in_strong():
    assert "canary" in STRONG_SIGNALS




def test_paraphrase_in_strong():
    assert "paraphrase" in STRONG_SIGNALS


def test_perm_option_in_weak():
    assert "perm_option" in WEAK_SIGNALS
    assert "perm_option" not in STRONG_SIGNALS


def test_spv_mia_in_strong():
    assert "spv_mia" in STRONG_SIGNALS


def test_strong_weak_disjoint():
    assert STRONG_SIGNALS.isdisjoint(WEAK_SIGNALS)


# ---------------- aggregate_verdict 语义 ----------------


def test_two_strong_positives_gives_dirty():
    r = [
        _mk("spv_mia", -3.2, Verdict.DIRTY),
        _mk("canary", 0.42, Verdict.DIRTY),
        _mk("perm_option", 0.05, Verdict.CLEAN),
    ]
    v = aggregate_verdict("mmlu-pro", Stage.SFT, r)
    assert v.verdict == Verdict.DIRTY
    assert v.strong_signals_positive == 2
    assert v.weak_signals_positive == 0


def test_one_strong_positive_gives_suspect():
    r = [
        _mk("spv_mia", -2.1, Verdict.SUSPECT),
        _mk("paraphrase", 0.03, Verdict.CLEAN),
    ]
    v = aggregate_verdict("mmlu-pro", Stage.SFT, r)
    assert v.verdict == Verdict.SUSPECT
    assert v.strong_signals_positive == 1


def test_two_weak_positives_gives_suspect():
    r = [
        _mk("perm_option", 0.12, Verdict.SUSPECT),
        _mk("mink_plus_plus", -5.0, Verdict.SUSPECT),
    ]
    v = aggregate_verdict("mmlu-pro", Stage.SFT, r)
    assert v.verdict == Verdict.SUSPECT
    assert v.weak_signals_positive == 2


def test_one_weak_alone_gives_clean():
    r = [
        _mk("perm_option", 0.11, Verdict.SUSPECT),
        _mk("spv_mia", -0.2, Verdict.CLEAN),
    ]
    v = aggregate_verdict("mmlu-pro", Stage.SFT, r)
    assert v.verdict == Verdict.CLEAN


def test_all_prereq_missing_gives_inconclusive():
    r = [
        _mk("spv_mia", None, Verdict.INCONCLUSIVE, ok=False),
        _mk("canary", None, Verdict.INCONCLUSIVE, ok=False),
    ]
    v = aggregate_verdict("mmlu-pro", Stage.SFT, r)
    assert v.verdict == Verdict.INCONCLUSIVE
    assert v.strong_signals_positive == 0


def test_inconclusive_not_counted_as_positive():
    r = [
        _mk("spv_mia", None, Verdict.INCONCLUSIVE, ok=False),
        _mk("canary", 0.42, Verdict.DIRTY),
    ]
    v = aggregate_verdict("mmlu-pro", Stage.SFT, r)
    # 只有 canary 一个 strong 阳性 → SUSPECT
    assert v.strong_signals_positive == 1
    assert v.verdict == Verdict.SUSPECT


# ---------------- calibrated 开关：summary 措辞 ----------------


def test_calibrated_false_marks_relative_ranking():
    r = [_mk("spv_mia", -3.0, Verdict.DIRTY), _mk("canary", 0.4, Verdict.DIRTY)]
    v = aggregate_verdict("x", Stage.SFT, r, calibrated=False)
    assert "相对排名" in v.summary
    assert "positive control" in v.summary


def test_calibrated_true_uses_traffic_light():
    r = [_mk("spv_mia", -3.0, Verdict.DIRTY), _mk("canary", 0.4, Verdict.DIRTY)]
    v = aggregate_verdict("x", Stage.SFT, r, calibrated=True)
    assert "🔴" in v.summary or "红" in v.summary


# ---------------- rank_by_suspicion 排序 ----------------


def test_rank_puts_more_strong_first():
    v_dirty = aggregate_verdict(
        "a",
        Stage.SFT,
        [
            _mk("spv_mia", -3.0, Verdict.DIRTY),
            _mk("canary", 0.42, Verdict.DIRTY),
        ],
    )
    v_clean = aggregate_verdict(
        "b", Stage.SFT, [_mk("spv_mia", -0.1, Verdict.CLEAN)]
    )
    ranked = rank_by_suspicion([v_clean, v_dirty])
    assert ranked[0].benchmark == "a"
    assert ranked[1].benchmark == "b"


def test_rank_tiebreaks_by_signal_magnitude():
    v1 = aggregate_verdict(
        "a", Stage.SFT, [_mk("spv_mia", -1.0, Verdict.SUSPECT)]
    )
    v2 = aggregate_verdict(
        "b", Stage.SFT, [_mk("spv_mia", -5.0, Verdict.SUSPECT)]
    )
    ranked = rank_by_suspicion([v1, v2])
    assert ranked[0].benchmark == "b"  # 更大信号排前


def test_rank_stable_on_name():
    v1 = aggregate_verdict("bench_b", Stage.SFT, [_mk("spv_mia", -0.1, Verdict.CLEAN)])
    v2 = aggregate_verdict("bench_a", Stage.SFT, [_mk("spv_mia", -0.1, Verdict.CLEAN)])
    ranked = rank_by_suspicion([v1, v2])
    assert [v.benchmark for v in ranked] == ["bench_a", "bench_b"]


def test_rank_does_not_mutate_input():
    inp = [
        aggregate_verdict("z", Stage.SFT, [_mk("spv_mia", -0.1, Verdict.CLEAN)]),
        aggregate_verdict(
            "a",
            Stage.SFT,
            [_mk("spv_mia", -3.0, Verdict.DIRTY), _mk("canary", 0.5, Verdict.DIRTY)],
        ),
    ]
    original_order = [v.benchmark for v in inp]
    _ = rank_by_suspicion(inp)
    assert [v.benchmark for v in inp] == original_order


# ---------------- render_trustworthiness_report markdown ----------------


def test_render_empty():
    out = render_trustworthiness_report([])
    assert "Benchmark 可信度报告" in out
    assert "无数据" in out


def test_render_phase1_default_says_relative():
    v = aggregate_verdict(
        "mmlu-pro",
        Stage.SFT,
        [_mk("spv_mia", -3.0, Verdict.DIRTY), _mk("canary", 0.4, Verdict.DIRTY)],
    )
    out = render_trustworthiness_report([v])
    assert "相对排名" in out
    assert "Phase 1" in out
    assert "positive control" in out
    assert "mmlu-pro" in out
    assert "spv_mia" in out
    assert "canary" in out


def test_render_calibrated_switch_title():
    v = aggregate_verdict(
        "x", Stage.SFT, [_mk("spv_mia", -3.0, Verdict.DIRTY)], calibrated=True
    )
    out = render_trustworthiness_report([v], calibrated=True)
    assert "绝对裁决" in out
    assert "Phase 1" not in out or "Phase 1（默认）" not in out


def test_render_shows_all_methods_including_inconclusive():
    v = aggregate_verdict(
        "x",
        Stage.SFT,
        [
            _mk("spv_mia", -3.0, Verdict.DIRTY),
            _mk("canary", None, Verdict.INCONCLUSIVE, ok=False),
        ],
    )
    out = render_trustworthiness_report([v])
    assert "spv_mia" in out
    assert "canary" in out
    # 前置缺失应该有 ✗ 标记
    assert "✗" in out


def test_render_ordered_by_rank():
    v_dirty = aggregate_verdict(
        "high_risk",
        Stage.SFT,
        [_mk("spv_mia", -3.0, Verdict.DIRTY), _mk("canary", 0.5, Verdict.DIRTY)],
    )
    v_clean = aggregate_verdict(
        "low_risk", Stage.SFT, [_mk("spv_mia", -0.05, Verdict.CLEAN)]
    )
    out = render_trustworthiness_report([v_clean, v_dirty])
    # high_risk 应先于 low_risk 出现在总表
    assert out.index("high_risk") < out.index("low_risk")
