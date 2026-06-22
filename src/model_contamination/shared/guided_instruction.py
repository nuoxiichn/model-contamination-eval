"""Guided Instruction（Golchin & Surdeanu ICLR 2024, arXiv:2308.14352）。

给模型「数据集名 + 分区类型 + 原始实例前缀」，要求续写；
对照组只给前缀不告知数据集名称。
若 guided 续写与真实后缀的 ROUGE-L / BLEURT 显著高于对照 → 污染。

测试 7 个数据集，准确率 92%–100%。黑盒。
缺点：可能被安全过滤器阻断（自研模型无此问题）。

Phase 2 实装。
"""

from __future__ import annotations

from typing import Any

from model_contamination.models.base import ModelInterface
from model_contamination.types import BenchmarkSpec, DetectionResult


def guided_instruction_test(
    model: ModelInterface,
    benchmark: BenchmarkSpec,
    samples: list[dict[str, Any]],
    prefix_ratio: float = 0.5,
    n_runs: int = 3,
) -> DetectionResult:
    raise NotImplementedError(
        "Phase 2: 实装 Guided Instruction（arXiv:2308.14352）。"
        " 步骤：对每条样本生成 guided 与 non-guided 续写，"
        " 计算与真实后缀的 ROUGE-L 差值，配对 t 检验。"
    )
