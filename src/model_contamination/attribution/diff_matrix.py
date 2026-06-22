"""差分矩阵：跨阶段对比同方法 signal，输出 ΔScore / ΔMIA / ΔOren_p / ΔCanary。

对应方案 §5.1 差分对象矩阵。
"""

from __future__ import annotations

from dataclasses import dataclass

from model_contamination.types import DetectionResult, Stage


@dataclass
class DiffSignal:
    method: str
    benchmark: str
    base_value: float | None
    sft_value: float | None
    rlhf_value: float | None
    delta_sft: float | None       # sft - base
    delta_rlhf: float | None      # rlhf - sft


def compute_diff_matrix(
    results_by_stage: dict[Stage, list[DetectionResult]],
) -> list[DiffSignal]:
    """跨阶段对齐 (method, benchmark)，输出差分。

    缺某个阶段的 signal 时对应 delta 为 None。
    """
    # 按 (method, benchmark) 索引
    indexed: dict[tuple[str, str], dict[Stage, float | None]] = {}
    for stage, results in results_by_stage.items():
        for r in results:
            key = (r.method, r.benchmark)
            indexed.setdefault(key, {})[stage] = r.signal

    diffs: list[DiffSignal] = []
    for (method, benchmark), stage_values in indexed.items():
        base = stage_values.get(Stage.BASE)
        sft = stage_values.get(Stage.SFT)
        rlhf = stage_values.get(Stage.RLHF)

        delta_sft = sft - base if (sft is not None and base is not None) else None
        delta_rlhf = rlhf - sft if (rlhf is not None and sft is not None) else None

        diffs.append(
            DiffSignal(
                method=method,
                benchmark=benchmark,
                base_value=base,
                sft_value=sft,
                rlhf_value=rlhf,
                delta_sft=delta_sft,
                delta_rlhf=delta_rlhf,
            )
        )
    return diffs
