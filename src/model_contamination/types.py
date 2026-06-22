"""共用类型定义。

所有检测方法返回 DetectionResult，所有 benchmark 用 BenchmarkSpec 描述。
保持 schema 稳定是跨方法 / 跨阶段做差分归因的前提。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal


class Stage(str, Enum):
    BASE = "base"
    SFT = "sft"
    RLHF = "rlhf"


class Verdict(str, Enum):
    CLEAN = "clean"
    SUSPECT = "suspect"
    DIRTY = "dirty"
    INCONCLUSIVE = "inconclusive"  # 前置条件缺失或信号不足


BenchmarkFormat = Literal[
    "multiple_choice",
    "math_cot",
    "code_completion",
    "open_generation",
    "cloze",
]

MethodTag = Literal[
    "oren",
    "guided",
    "ts_guessing",
    "perm_option",
    "paraphrase",
    "mink_plus_plus",
    "spv_mia",
    "memlens",
    "log_prober",
    "self_critique",
    "canary",
    "family_diff",
]


@dataclass(frozen=True)
class BenchmarkSpec:
    """benchmarks.yaml 中单条 benchmark 的反序列化结果。"""

    name: str
    family: str
    format: BenchmarkFormat
    language: Literal["en", "zh", "multi"]
    variants: list[str]
    applicable_methods: list[str]
    trustworthiness_default: Verdict
    data_source: Literal["hf", "modelscope", "local", "livebench-api"]
    data_id: str
    note: str = ""


@dataclass
class DetectionResult:
    """所有检测方法的统一返回。

    跨方法做差分（ΔScore / ΔMIA / ΔOren_p）时，归因层只读 signal 字段，
    所以同一方法在不同阶段返回的 signal 必须含义一致、可减。
    """

    method: str
    stage: Stage
    benchmark: str
    signal: float | None              # 主要数值信号；前置不满足时为 None
    verdict_hint: Verdict
    prerequisites_met: bool
    evidence: dict[str, Any] = field(default_factory=dict)
    error: str | None = None          # 前置不满足或运行错误时填


@dataclass
class FamilyDiffResult:
    """同族对照 ΔScore。

    任务孤岛特征：main_score 显著高于 variant_scores 平均 → SFT 阶段污染嫌疑。
    """

    main_benchmark: str
    main_score: float
    variant_scores: dict[str, float]
    delta_avg: float                  # main - mean(variants)
    delta_max: float                  # main - min(variants)
    is_island: bool                   # delta_avg > 阈值


@dataclass
class BenchmarkVerdict:
    """单个 benchmark 跨方法综合裁决，给可信度报告用。"""

    benchmark: str
    stage: Stage
    verdict: Verdict
    strong_signals_positive: int
    weak_signals_positive: int
    results: list[DetectionResult]
    summary: str


@dataclass
class StageAttribution:
    """单 benchmark 的阶段归因结果。"""

    benchmark: str
    has_pre_sft_checkpoint: bool
    has_canary: bool
    delta_score: float | None
    delta_oren_p: float | None
    delta_mia_auc: float | None
    delta_canary: float | None
    family_drop_sft: float | None     # SFT 主-同族
    family_drop_rlhf: float | None    # RLHF 主-同族
    conclusion: Literal[
        "pretrain_only",
        "sft_introduced",
        "rl_migrated",
        "joint",
        "unattributable",      # 缺前置 / 信息论上不可分
    ]
    confidence: Literal["high", "medium", "low"]
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelCheckpoint:
    """checkpoint 引用。stage_tag 决定走哪条阶段分支。"""

    path: Path
    stage: Stage
    name: str
    parent_checkpoint: Path | None = None   # 用于差分归因（如 SFT 指向 base）
    data_source_tags: list[str] = field(default_factory=list)  # 数据配方反馈用
