"""LogProber（arXiv:2408.14352）。

区分 Q-A / Q- / -A 三种污染类型，分离 prompt 和 answer 的条件概率。
短文本 Q-A 上 AUC > 0.9。灰盒。

Phase 2 实装。
"""

from __future__ import annotations

from typing import Any

from model_contamination.models.base import ModelInterface
from model_contamination.types import BenchmarkSpec, DetectionResult


def log_prober(
    model: ModelInterface,
    benchmark: BenchmarkSpec,
    samples: list[dict[str, Any]],
) -> DetectionResult:
    raise NotImplementedError(
        "Phase 2: 实装 LogProber。"
        " 分别测 P(Q)、P(A|Q)、P(A)，判定污染落在哪一环。"
    )
