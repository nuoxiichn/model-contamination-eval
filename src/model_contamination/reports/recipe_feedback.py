"""数据配方反馈报告（§6.3，指导下一轮数据合成）。

按数据源 tag 切片污染分布。这是唯一能反过来指导数据合成的报告。
反馈对象是数据源类别，不是单条样本。
"""

from __future__ import annotations

from model_contamination.types import DetectionResult


def render_recipe_feedback(
    results_by_data_source: dict[str, list[DetectionResult]],
) -> str:
    raise NotImplementedError(
        "Phase 3: 实装配方反馈报告。"
        " 前置：SFT 数据源 tag 系统化。"
    )
