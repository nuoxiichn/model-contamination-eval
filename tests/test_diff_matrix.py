"""diff_matrix contrastive 信号单测。

覆盖 Finding 8 红线：raw Δpv 跨 bench 不能直接卡阈值，必须 contrastive 到同 bench
clean ckpt baseline。
"""

from __future__ import annotations

import math

import pytest

from model_contamination.attribution.diff_matrix import (
    ContrastiveSignal,
    compute_contrastive_signals,
    is_dirtier_than_clean,
    signal_direction_for,
)
from model_contamination.types import DetectionResult, Stage, Verdict


def _dr(method: str, benchmark: str, signal: float | None) -> DetectionResult:
    return DetectionResult(
        method=method,
        stage=Stage.SFT,
        benchmark=benchmark,
        signal=signal,
        verdict_hint=Verdict.INCONCLUSIVE,
        prerequisites_met=True if signal is not None else False,
    )


# ----- signal_direction 注册 -----

def test_signal_direction_known_methods():
    assert signal_direction_for("spv_mia") == "lower_is_dirtier"
    assert signal_direction_for("mink_plus_plus") == "lower_is_dirtier"
    assert signal_direction_for("perm_option") == "higher_is_dirtier"
    assert signal_direction_for("oren") == "lower_is_dirtier"


def test_signal_direction_unknown_defaults_lower():
    assert signal_direction_for("brand_new_method") == "lower_is_dirtier"


# ----- baseline: clean = clean → excess=0, ratio=1（不在输出里，因为 clean 自身被跳过）-----

def test_clean_baseline_excluded_from_output():
    """clean ckpt 自己不出现在 contrastive 结果里（无意义 self-baseline）。"""
    results = {
        "clean": [_dr("spv_mia", "gsm8k", -1.20)],
        "gsm8k_heavy": [_dr("spv_mia", "gsm8k", -211.91)],
    }
    sigs = compute_contrastive_signals(results, clean_ckpt="clean")
    target_ckpts = {s.target_ckpt for s in sigs}
    assert "clean" not in target_ckpts
    assert "gsm8k_heavy" in target_ckpts


# ----- calibration v3 真实场景 -----

def test_gsm8k_heavy_v3_case_ratio_strongly_above_one():
    """v3 实测：clean=-1.20, gsm8k_heavy=-211.91 → ratio≈176, excess≈-210.71。"""
    results = {
        "clean": [_dr("spv_mia", "gsm8k", -1.20)],
        "gsm8k_heavy": [_dr("spv_mia", "gsm8k", -211.91)],
    }
    sigs = compute_contrastive_signals(results, clean_ckpt="clean")
    [s] = [s for s in sigs if s.target_ckpt == "gsm8k_heavy"]
    assert s.prerequisites_met
    assert s.ratio_reliable  # |clean|=1.20 > 0.5
    assert s.ratio is not None
    assert s.ratio == pytest.approx(-211.91 / -1.20, rel=1e-6)
    assert s.ratio > 100  # 远超 1 = 重度污染
    assert s.excess == pytest.approx(-210.71, abs=0.01)
    assert s.signal_direction == "lower_is_dirtier"
    assert is_dirtier_than_clean(s) is True


def test_gsm8k_light_v3_case_already_separates():
    """v3 实测：clean=-1.20, gsm8k_light=-13.47 → ratio≈11.2，SPV-MIA 在 light 就 work。"""
    results = {
        "clean": [_dr("spv_mia", "gsm8k", -1.20)],
        "gsm8k_light": [_dr("spv_mia", "gsm8k", -13.47)],
    }
    sigs = compute_contrastive_signals(results, clean_ckpt="clean")
    [s] = sigs
    assert s.ratio == pytest.approx(-13.47 / -1.20, rel=1e-6)
    assert s.ratio > 10
    assert is_dirtier_than_clean(s) is True


# ----- ratio degrade：|clean| < eps -----

def test_ratio_unreliable_when_clean_near_zero():
    """clean Δpv ≈ 0（噪声级）→ ratio 不稳定，标 ratio_reliable=False，但 excess 仍可用。"""
    results = {
        "clean": [_dr("spv_mia", "gsm8k", 0.1)],   # |0.1| < 0.5 eps
        "target": [_dr("spv_mia", "gsm8k", -50.0)],
    }
    sigs = compute_contrastive_signals(results, clean_ckpt="clean")
    [s] = sigs
    assert s.prerequisites_met
    assert s.ratio_reliable is False
    assert s.ratio is None
    assert s.excess == pytest.approx(-50.1, abs=0.01)
    # is_dirtier_than_clean 降级到 excess 判定
    assert is_dirtier_than_clean(s) is True


# ----- 缺 clean ckpt → prerequisites_met=False -----

def test_missing_clean_ckpt_marks_all_unavailable():
    results = {
        "gsm8k_heavy": [_dr("spv_mia", "gsm8k", -211.91)],
        "mmlu_heavy": [_dr("spv_mia", "gsm8k", -6.12)],
    }
    sigs = compute_contrastive_signals(results, clean_ckpt="clean")
    assert all(not s.prerequisites_met for s in sigs)
    assert all(s.reason and "missing" in s.reason for s in sigs)
    assert all(s.excess is None and s.ratio is None for s in sigs)


# ----- target signal 为 None（如 SPV-MIA degraded MC 场景） -----

def test_target_signal_none_marks_inconclusive():
    """U.5 场景：mmlu-pro 在所有 ckpt 上 spv_mia degraded → signal=None。"""
    results = {
        "clean": [_dr("spv_mia", "gsm8k", -1.20)],
        "mmlu_heavy": [_dr("spv_mia", "gsm8k", None)],
    }
    sigs = compute_contrastive_signals(results, clean_ckpt="clean")
    [s] = sigs
    assert s.prerequisites_met is False
    assert "target" in s.reason
    assert s.excess is None


def test_clean_signal_none_marks_inconclusive():
    results = {
        "clean": [_dr("spv_mia", "mmlu-pro", None)],
        "mmlu_heavy": [_dr("spv_mia", "mmlu-pro", -6.0)],
    }
    sigs = compute_contrastive_signals(results, clean_ckpt="clean")
    [s] = sigs
    assert s.prerequisites_met is False
    assert "clean" in s.reason


def test_nan_signals_treated_as_missing():
    results = {
        "clean": [_dr("spv_mia", "gsm8k", float("nan"))],
        "target": [_dr("spv_mia", "gsm8k", -10.0)],
    }
    sigs = compute_contrastive_signals(results, clean_ckpt="clean")
    [s] = sigs
    assert s.prerequisites_met is False


# ----- higher_is_dirtier 方法（perm_option / TS-guessing） -----

def test_higher_is_dirtier_excess_sign():
    """perm_option：偏好率越高越脏。clean=0.3, target=0.9 → excess>0 = 更脏。"""
    results = {
        "clean": [_dr("perm_option", "mmlu-pro", 0.3)],
        "mmlu_heavy": [_dr("perm_option", "mmlu-pro", 0.9)],
    }
    sigs = compute_contrastive_signals(results, clean_ckpt="clean")
    [s] = sigs
    assert s.signal_direction == "higher_is_dirtier"
    assert s.excess == pytest.approx(0.6, abs=1e-6)
    assert s.ratio == pytest.approx(3.0, abs=1e-6)
    assert is_dirtier_than_clean(s) is True


def test_higher_is_dirtier_below_clean_means_cleaner():
    """target 偏好率低于 clean → cleaner（is_dirtier_than_clean=False）。"""
    results = {
        "clean": [_dr("perm_option", "mmlu-pro", 0.5)],
        "target": [_dr("perm_option", "mmlu-pro", 0.3)],
    }
    sigs = compute_contrastive_signals(results, clean_ckpt="clean")
    [s] = sigs
    assert s.excess == pytest.approx(-0.2, abs=1e-6)
    assert is_dirtier_than_clean(s) is False


# ----- 多 (method, benchmark) 组合 -----

def test_multiple_methods_benchmarks_grouped_correctly():
    """同 ckpt 多 (method, benchmark) 各算各的 contrastive。"""
    results = {
        "clean": [
            _dr("spv_mia", "gsm8k", -1.2),
            _dr("spv_mia", "humaneval", -2.13),
            _dr("mink_plus_plus", "gsm8k", -1.72),
        ],
        "gsm8k_heavy": [
            _dr("spv_mia", "gsm8k", -211.91),
            _dr("spv_mia", "humaneval", -6.10),
            _dr("mink_plus_plus", "gsm8k", -14.42),
        ],
    }
    sigs = compute_contrastive_signals(results, clean_ckpt="clean")
    assert len(sigs) == 3
    by_key = {(s.method, s.benchmark): s for s in sigs}
    assert by_key[("spv_mia", "gsm8k")].ratio > 100
    assert by_key[("spv_mia", "humaneval")].excess < 0  # 跨任务退化
    assert by_key[("mink_plus_plus", "gsm8k")].ratio == pytest.approx(-14.42 / -1.72, rel=1e-6)


def test_method_missing_from_target_skipped_silently():
    """target ckpt 没跑某方法 → 不出现在结果里（不报错）。"""
    results = {
        "clean": [
            _dr("spv_mia", "gsm8k", -1.2),
            _dr("perm_option", "mmlu-pro", 0.3),
        ],
        "target": [_dr("spv_mia", "gsm8k", -50.0)],  # 没 perm_option
    }
    sigs = compute_contrastive_signals(results, clean_ckpt="clean")
    assert len(sigs) == 1
    assert sigs[0].method == "spv_mia"


# ----- is_dirtier_than_clean 边界 -----

def test_is_dirtier_returns_none_when_prerequisites_unmet():
    sig = ContrastiveSignal(
        method="spv_mia", benchmark="x", target_ckpt="t", clean_ckpt="c",
        raw_target=None, raw_clean=None, excess=None, ratio=None,
        ratio_reliable=False, signal_direction="lower_is_dirtier",
        prerequisites_met=False, reason="test",
    )
    assert is_dirtier_than_clean(sig) is None


def test_is_dirtier_uses_excess_when_signs_differ():
    """raw_target 与 raw_clean 异号时 ratio 含义反转，必须降级到 excess。"""
    # clean 正、target 负 → ratio<0，方向不明，必须看 excess
    sig = ContrastiveSignal(
        method="spv_mia", benchmark="x", target_ckpt="t", clean_ckpt="c",
        raw_target=-5.0, raw_clean=1.0,
        excess=-6.0, ratio=-5.0,
        ratio_reliable=True, signal_direction="lower_is_dirtier",
        prerequisites_met=True,
    )
    # ratio=-5 不是 ">1"，按 excess<0 + lower_is_dirtier → 更脏
    assert is_dirtier_than_clean(sig) is True
