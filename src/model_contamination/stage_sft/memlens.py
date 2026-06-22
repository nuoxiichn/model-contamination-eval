"""MemLens（arXiv:2509.20909）。

用 logit lens 分析答案 token 在各层的概率轨迹。
污染样本在早期层就锁定答案（shortcut reasoning），干净样本是逐层证据累积。

关键优势：对 rephrasing 和 perturbation 稳健（其他方法的痛点）。
白盒方法：需要 hidden_states + 共享 unembedding 矩阵。

Phase 3 实装（自研模型专用）。
"""

from __future__ import annotations

from typing import Any

from model_contamination.models.base import Capability, ModelInterface
from model_contamination.types import BenchmarkSpec, DetectionResult, Stage, Verdict


def memlens_probe(
    model: ModelInterface,
    benchmark: BenchmarkSpec,
    samples: list[dict[str, Any]],
    layer_range: tuple[int, int] = (0, 16),
    shortcut_threshold: float = 0.5,
) -> DetectionResult:
    if not model.supports(Capability.LOGITS_LENS):
        return DetectionResult(
            method="memlens",
            stage=Stage(model.stage_tag),
            benchmark=benchmark.name,
            signal=None,
            verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            error="MemLens 需要 layer_logits 接口（hidden_states + unembedding）",
        )
    raise NotImplementedError(
        "Phase 3: 实装 MemLens。"
        " 对每条样本提取答案 token 在 layer_range 内的 prob 轨迹，"
        " 早期层 prob > shortcut_threshold 的样本比例 → signal。"
    )
