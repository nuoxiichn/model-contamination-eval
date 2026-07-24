"""差分矩阵：跨阶段对比同方法 signal，输出 ΔScore / ΔMIA。

对应方案 §5.1 差分对象矩阵。

新增 §5.1.b 跨 ckpt contrastive（同阶段内 clean ckpt 锚定）：
源于 calibration v3 / Finding 8（[[calibration-findings-2026-06-29]]）——
SPV-MIA 跨任务训练会让全 bench Δpv 整体偏负，raw 单值不能直接卡阈值。
对策：同 SFT 阶段下 clean ckpt 是 anchor，target ckpt 的可疑度由
`excess = raw_target - raw_clean` 与 `ratio = raw_target / raw_clean` 衡量。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from model_contamination.types import DetectionResult, Stage

# raw_clean 绝对值低于此阈值时 ratio 数值不稳定，标记 ratio_reliable=False
# 各方法 signal 量纲不同，逐方法登记：
#   - SPV-MIA Δpv：v3 clean 基线 ~ -1.2~-2.1，0.5 留安全边
#   - perm_option：leak_fraction ∈ [0,1]，clean 基线 ≈ IsolationForest 假阳率（低），0.05 安全边
#   - MinK-pp median 分数：~ -1 to -15 范围，0.1 安全边
_DEFAULT_RATIO_DENOM_EPS = 0.5
_RATIO_DENOM_EPS: dict[str, float] = {
    "spv_mia": 0.5,
    "mink_plus_plus": 0.1,
    "perm_option": 0.05,
    "paraphrase": 0.05,
}


def _denom_eps_for(method: str) -> float:
    return _RATIO_DENOM_EPS.get(method, _DEFAULT_RATIO_DENOM_EPS)


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


@dataclass
class ContrastiveSignal:
    """同阶段内 target ckpt 相对 clean ckpt 的可疑度。

    用于 SFT 阶段 SPV-MIA / Min-K%++ 等需要 baseline 校准的方法。
    `signal_direction` 编码原始 signal 与 contamination 程度的方向：
      - 'lower_is_dirtier'（SPV-MIA Δpv：越负越像 member）
      - 'higher_is_dirtier'（MIA AUC、perm_option 偏好率：越高越可疑）

    `excess`：raw_target - raw_clean，保留方向（lower_is_dirtier 下 excess<0 = 更脏）
    `ratio`：raw_target / raw_clean，clean 趋零时不稳定，由 ratio_reliable 标记
    `prerequisites_met=False`：target 或 clean 缺 signal / clean ckpt 不存在
    """

    method: str
    benchmark: str
    target_ckpt: str
    clean_ckpt: str
    raw_target: float | None
    raw_clean: float | None
    excess: float | None            # target - clean
    ratio: float | None             # target / clean；clean≈0 时 None
    ratio_reliable: bool            # |raw_clean| >= _RATIO_DENOM_EPS
    signal_direction: str           # 'lower_is_dirtier' | 'higher_is_dirtier'
    prerequisites_met: bool
    reason: str | None = None       # prerequisites_met=False 时填


# 各方法的 signal 方向（contamination 越严重时 signal 往哪个方向走）
_SIGNAL_DIRECTION: dict[str, str] = {
    "spv_mia": "lower_is_dirtier",      # Δpv 越负越像 member
    "mink_plus_plus": "lower_is_dirtier",  # 单题分数越低越像 member
    "perm_option": "higher_is_dirtier",  # 原位置偏好率越高越可疑
    "paraphrase": "lower_is_dirtier",
    "codec": "higher_is_dirtier",
}


def signal_direction_for(method: str) -> str:
    """暴露给 verdict 层；未知方法默认 lower_is_dirtier（与 SPV-MIA 一致）。"""
    return _SIGNAL_DIRECTION.get(method, "lower_is_dirtier")


def compute_contrastive_signals(
    results_by_ckpt: dict[str, list[DetectionResult]],
    clean_ckpt: str,
) -> list[ContrastiveSignal]:
    """对每个 (method, benchmark, target_ckpt != clean_ckpt) 算 contrastive。

    `results_by_ckpt`：{ckpt_name: [DetectionResult, ...]}，每个 ckpt 的 results
    必须来自同一 stage（caller 保证）。
    `clean_ckpt`：作为 baseline 的 ckpt 名（一般是不注入任何 benchmark 的 SFT ckpt）。

    缺 clean ckpt → 全部 target 的 prerequisites_met=False。
    target 或 clean 的 signal 为 None / NaN → 对应 ContrastiveSignal
    prerequisites_met=False，excess/ratio=None，reason 填诊断。
    """
    if clean_ckpt not in results_by_ckpt:
        # 没有 clean baseline，整列不可用
        out: list[ContrastiveSignal] = []
        for ckpt, rs in results_by_ckpt.items():
            for r in rs:
                out.append(
                    ContrastiveSignal(
                        method=r.method,
                        benchmark=r.benchmark,
                        target_ckpt=ckpt,
                        clean_ckpt=clean_ckpt,
                        raw_target=r.signal,
                        raw_clean=None,
                        excess=None,
                        ratio=None,
                        ratio_reliable=False,
                        signal_direction=signal_direction_for(r.method),
                        prerequisites_met=False,
                        reason=f"clean ckpt '{clean_ckpt}' missing from results_by_ckpt",
                    )
                )
        return out

    # 用 (method, benchmark) 索引 clean ckpt 的 signal
    clean_index: dict[tuple[str, str], float | None] = {}
    for r in results_by_ckpt[clean_ckpt]:
        clean_index[(r.method, r.benchmark)] = r.signal

    out: list[ContrastiveSignal] = []
    for ckpt, rs in results_by_ckpt.items():
        if ckpt == clean_ckpt:
            continue
        for r in rs:
            key = (r.method, r.benchmark)
            raw_clean = clean_index.get(key)
            raw_target = r.signal
            direction = signal_direction_for(r.method)

            target_ok = raw_target is not None and not (
                isinstance(raw_target, float) and math.isnan(raw_target)
            )
            clean_ok = raw_clean is not None and not (
                isinstance(raw_clean, float) and math.isnan(raw_clean)
            )

            if not target_ok or not clean_ok:
                missing = []
                if not target_ok:
                    missing.append("target")
                if not clean_ok:
                    missing.append("clean")
                out.append(
                    ContrastiveSignal(
                        method=r.method,
                        benchmark=r.benchmark,
                        target_ckpt=ckpt,
                        clean_ckpt=clean_ckpt,
                        raw_target=raw_target,
                        raw_clean=raw_clean,
                        excess=None,
                        ratio=None,
                        ratio_reliable=False,
                        signal_direction=direction,
                        prerequisites_met=False,
                        reason=f"missing signal: {','.join(missing)}",
                    )
                )
                continue

            excess = raw_target - raw_clean
            denom_eps = _denom_eps_for(r.method)
            ratio_reliable = abs(raw_clean) >= denom_eps
            ratio = (raw_target / raw_clean) if ratio_reliable else None

            out.append(
                ContrastiveSignal(
                    method=r.method,
                    benchmark=r.benchmark,
                    target_ckpt=ckpt,
                    clean_ckpt=clean_ckpt,
                    raw_target=raw_target,
                    raw_clean=raw_clean,
                    excess=excess,
                    ratio=ratio,
                    ratio_reliable=ratio_reliable,
                    signal_direction=direction,
                    prerequisites_met=True,
                )
            )
    return out


def is_dirtier_than_clean(sig: ContrastiveSignal) -> bool | None:
    """便利函数：根据 signal_direction 判断 target 是否比 clean 更脏。

    优先用 ratio（更稳定），ratio 不可用时降级到 excess。
    prerequisites 未满足时返回 None。
    """
    if not sig.prerequisites_met or sig.excess is None:
        return None
    if sig.ratio_reliable and sig.ratio is not None:
        # lower_is_dirtier 下：raw 都是负的 → target 更负 → |target|>|clean| → ratio>1
        # higher_is_dirtier 下：raw 都是正的 → target 更大 → ratio>1
        # 两种情况都是 ratio > 1 表示更脏（前提：raw_clean 与 raw_target 同号且非零）
        same_sign = (sig.raw_target or 0) * (sig.raw_clean or 0) >= 0
        if same_sign:
            return sig.ratio > 1.0
    # 降级到 excess
    if sig.signal_direction == "lower_is_dirtier":
        return sig.excess < 0
    return sig.excess > 0
