"""阶段归因裁决（§5.2 of solution doc）。

输入差分矩阵 + canary 留存率，输出 StageAttribution。
"""

from __future__ import annotations

from model_contamination.attribution.diff_matrix import DiffSignal
from model_contamination.types import StageAttribution


def attribute_stage(
    benchmark: str,
    diffs: list[DiffSignal],
    canary_delta: float | None,
    has_pre_sft_checkpoint: bool,
    has_canary: bool,
) -> StageAttribution:
    """按可信度排序应用归因规则。

    强信号优先：
    1. Canary 留存率差异 → 直接定位阶段
    2. 强信号缺失时才用 ΔMIA 作为辅助。

    无 pre-SFT checkpoint 时 conclusion 强制为 unattributable。
    """
    if not has_pre_sft_checkpoint:
        return StageAttribution(
            benchmark=benchmark,
            has_pre_sft_checkpoint=False,
            has_canary=has_canary,
            delta_score=None,
            delta_mia_auc=None,
            delta_canary=canary_delta,
            conclusion="unattributable",
            confidence="low",
            evidence={"reason": "无 pre-SFT checkpoint，阶段归因不可用"},
        )

    raise NotImplementedError(
        "Phase 2: 实装阶段归因。规则见方案 §5.2。"
        " 关键：强信号一致时无需 ΔMIA；强信号缺失时才用 ΔMIA 作为弱辅助。"
    )
