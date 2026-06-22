"""选项排列检测（Yang et al. AAAI 2024, arXiv:2305.10403）。

打乱多选题选项顺序，看模型是否偏好原始选项位置。灰盒（需 logprobs），专用于多选题格式。
在 C-Eval 上发现 Qwen 系列泄漏值是其他模型近 10 倍；Qwen2-72B 在 CMB 上 42% 泄漏。

Phase 2 实装。
"""

from __future__ import annotations

from typing import Any

from model_contamination.models.base import ModelInterface
from model_contamination.types import BenchmarkSpec, DetectionResult


def option_permutation_test(
    model: ModelInterface,
    benchmark: BenchmarkSpec,
    samples: list[dict[str, Any]],
) -> DetectionResult:
    if benchmark.format != "multiple_choice":
        raise ValueError(
            f"option_permutation 仅适用 multiple_choice 格式，"
            f"benchmark {benchmark.name} 是 {benchmark.format}"
        )
    raise NotImplementedError(
        "Phase 2: 实装选项排列检测。对每题枚举选项排列，比较模型对原始位置的偏好度。"
    )
