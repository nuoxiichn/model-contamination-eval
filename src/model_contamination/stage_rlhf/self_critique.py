"""Self-Critique（arXiv:2510.09259）。

利用 entropy 和 token-level dependency 在 RL 后阶段检测 SFT 污染。
在 dual-contamination 场景 AUC 提升 55%。
RLHF 后唯一仍可用的实例级 MIA 类方法。

Phase 3 实装。
"""

from __future__ import annotations

from typing import Any

from model_contamination.models.base import ModelInterface
from model_contamination.types import BenchmarkSpec, DetectionResult


def self_critique(
    model: ModelInterface,
    benchmark: BenchmarkSpec,
    samples: list[dict[str, Any]],
) -> DetectionResult:
    raise NotImplementedError(
        "Phase 3: 实装 Self-Critique。"
        " 让模型评分自身输出，利用 entropy 与 token dependency 检测污染。"
    )
