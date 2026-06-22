from model_contamination.shared.family_diff import (
    cross_stage_family_diff,
    family_diff,
)


def test_island_detection():
    """主 benchmark 远高于同族 → 判为孤岛。"""
    result = family_diff(
        main_benchmark="gsm8k",
        main_score=95.3,
        variant_scores={"gsm1k": 28.7, "gsm-plus": 30.0},
        island_threshold=10.0,
    )
    assert result.is_island
    assert result.delta_avg > 60


def test_no_island_when_aligned():
    """主 benchmark 与同族对齐 → 非孤岛。"""
    result = family_diff(
        main_benchmark="gsm8k",
        main_score=72.0,
        variant_scores={"gsm1k": 70.0, "gsm-plus": 68.0},
        island_threshold=10.0,
    )
    assert not result.is_island


def test_empty_variants():
    result = family_diff("foo", 50.0, {})
    assert not result.is_island
    assert result.delta_avg == 0.0


def test_cross_stage_rl_migration():
    """RL 后变体大涨 → 判为 RL 阶段污染迁移。"""
    deltas = cross_stage_family_diff(
        {
            "sft": {"gsm8k": 95.0, "gsm1k": 28.0},
            "rlhf": {"gsm8k": 96.0, "gsm1k": 88.0},
        }
    )
    assert deltas["gsm1k"] > 50
    assert deltas["gsm8k"] < 5
