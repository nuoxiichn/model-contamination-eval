"""Paraphrase Stress Test（arXiv:2510.08616）。

同义改写 prompt，观察准确率下降幅度。
SFT 直接优化 P(response|prompt)，prompt 改写打击 SFT 最敏感维度。
已知污染对照集上典型 gap 10–30%。黑盒。

Phase 2 实装。
"""

from __future__ import annotations

from typing import Any

from model_contamination.models.base import ModelInterface
from model_contamination.types import BenchmarkSpec, DetectionResult


def paraphrase_stress_test(
    model: ModelInterface,
    benchmark: BenchmarkSpec,
    samples: list[dict[str, Any]],
    n_paraphrases: int = 5,
    paraphraser: str = "rule",   # rule / gpt-4-mini / local model
    worst_case_drop_threshold: float = 0.15,
) -> DetectionResult:
    raise NotImplementedError(
        "Phase 2: 实装 Paraphrase Stress Test。"
        " 步骤：对每条 prompt 生成 n 个改写，对模型分数取 worst-case，"
        " 与原始分数对比。drop > threshold → 阳性。"
    )
