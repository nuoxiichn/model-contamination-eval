"""SPV-MIA（Fu et al. NeurIPS 2024, arXiv:2311.06062）。

SFT 场景 MIA SOTA：GPT-2 / Wikitext-103 LoRA 10 epoch AUC=0.975，
LLaMA-7B / Wikitext-103 AUC=0.951。

关键前提：同源 base checkpoint 作为 reference。
LoRA 场景 reference 就是"不加 LoRA 权重的同一个模型"。

Phase 2 实装。这是 SFT 阶段实例级主信号。
"""

from __future__ import annotations

from typing import Any

from model_contamination.models.base import Capability, ModelInterface
from model_contamination.types import BenchmarkSpec, DetectionResult, Stage, Verdict


def spv_mia(
    target_model: ModelInterface,
    benchmark: BenchmarkSpec,
    samples: list[dict[str, Any]],
    reference_model: ModelInterface,
    n_neighbors: int = 10,
    noise_scale: float = 0.1,
) -> DetectionResult:
    """SPV-MIA 检测。

    target_model: SFT 后的模型
    reference_model: 同源 base（或上一版本 SFT checkpoint）
    """
    if not target_model.supports(Capability.LOGPROBS):
        return DetectionResult(
            method="spv_mia",
            stage=Stage(target_model.stage_tag),
            benchmark=benchmark.name,
            signal=None,
            verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            error="SPV-MIA requires target_model.logprobs",
        )
    if not reference_model.supports(Capability.LOGPROBS):
        return DetectionResult(
            method="spv_mia",
            stage=Stage(target_model.stage_tag),
            benchmark=benchmark.name,
            signal=None,
            verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            error="SPV-MIA requires reference_model.logprobs",
        )

    raise NotImplementedError(
        "Phase 2: 实装 SPV-MIA。"
        " 步骤：对每条 sample 生成 n_neighbors 个 paraphrase 邻居，"
        " 计算 target_model 与 reference_model 在原文 vs 邻居上的 loss 差，"
        " 用 self-prompt 校准后做成员推断。"
    )
