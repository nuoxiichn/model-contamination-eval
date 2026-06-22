"""Min-K%++（Zhang et al. 2024, arXiv:2404.02936）。

对 token log-probability 做 z-score 归一化，取最低 K% 的均值。
Min-K%++ 在 Min-K% 基础上提升最多 10 AUC 点。

⚠️ base 阶段实际 AUC ~0.5（MIMIR 控制时间偏移后）。仅作辅助信号，不单独定性。
SFT 阶段配 same-source base reference 时 AUC 0.7+。

Phase 2 实装。
"""

from __future__ import annotations

from typing import Any

from model_contamination.models.base import ModelInterface
from model_contamination.types import BenchmarkSpec, DetectionResult


def mink_plus_plus(
    model: ModelInterface,
    benchmark: BenchmarkSpec,
    samples: list[dict[str, Any]],
    k_ratio: float = 0.2,
    reference_model: ModelInterface | None = None,
) -> DetectionResult:
    raise NotImplementedError(
        "Phase 2: 实装 Min-K%++（arXiv:2404.02936）。"
        " 若提供 reference_model，应做 reference-based 校准（SFT 阶段必须）。"
    )
