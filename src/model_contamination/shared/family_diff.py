"""同族对照 ΔScore（任务孤岛特征诊断）。

任务孤岛特征是 SFT 污染的标志性指纹（arXiv:2601.06103）：
    SFT 污染：只抬被污染任务本身（GSM8K↑，GSM1k 不动）
    RL 污染：把泄漏迁移到同族任务（GSM8K↑，GSM1k 也↑，伪装成真泛化）

接口契约：
    输入 main_score + variant_scores
    输出 FamilyDiffResult，含 delta_avg / delta_max / is_island 判定

Phase 1：实装。这是最简单的信号但效果很强。
"""

from __future__ import annotations

from model_contamination.types import FamilyDiffResult


def family_diff(
    main_benchmark: str,
    main_score: float,
    variant_scores: dict[str, float],
    island_threshold: float = 10.0,
) -> FamilyDiffResult:
    """计算同族对照差值。

    island_threshold 单位与 score 一致（通常是百分点）。
    delta_avg > threshold → 判为任务孤岛（强 SFT 污染嫌疑）。
    """
    if not variant_scores:
        return FamilyDiffResult(
            main_benchmark=main_benchmark,
            main_score=main_score,
            variant_scores={},
            delta_avg=0.0,
            delta_max=0.0,
            is_island=False,
        )

    variant_values = list(variant_scores.values())
    avg = sum(variant_values) / len(variant_values)
    min_v = min(variant_values)

    delta_avg = main_score - avg
    delta_max = main_score - min_v

    return FamilyDiffResult(
        main_benchmark=main_benchmark,
        main_score=main_score,
        variant_scores=dict(variant_scores),
        delta_avg=delta_avg,
        delta_max=delta_max,
        is_island=delta_avg > island_threshold,
    )


def cross_stage_family_diff(
    benchmark_family_scores: dict[str, dict[str, float]],
) -> dict[str, float]:
    """跨阶段对比：检测 RL 阶段是否把污染迁移到同族任务。

    输入：{stage: {benchmark: score}}，例如
        {
            "sft": {"gsm8k": 95.3, "gsm1k": 28.7, "gsm-plus": 30.0},
            "rlhf": {"gsm8k": 96.1, "gsm1k": 89.0, "gsm-plus": 88.0},
        }
    输出：每个变体在 sft→rlhf 间的涨幅
        {"gsm1k": +60.3, "gsm-plus": +58.0}

    判定：变体涨幅显著（>10） → RL 阶段污染迁移（vs SFT 阶段污染是孤岛）。
    """
    if "sft" not in benchmark_family_scores or "rlhf" not in benchmark_family_scores:
        return {}

    sft = benchmark_family_scores["sft"]
    rlhf = benchmark_family_scores["rlhf"]
    return {
        bench: rlhf[bench] - sft[bench]
        for bench in rlhf
        if bench in sft
    }
