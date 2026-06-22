"""阶段归因报告（§6.2，内部审计用）。"""

from __future__ import annotations

from model_contamination.types import StageAttribution


def render_attribution_report(attributions: list[StageAttribution]) -> str:
    """渲染阶段归因 markdown 报告。"""
    raise NotImplementedError("Phase 2: 实装阶段归因报告渲染")
