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
    "ts_guessing",
    "perm_option",
    "paraphrase",
    "mink_plus_plus",
    "spv_mia",
    "self_critique",
    "codec",
]


@dataclass(frozen=True)
class BenchmarkQuestion:
    """单条 benchmark 题目，由 loader 从原始数据集归一化得到。

    与 BenchmarkSpec 分离：spec 是 yaml 中的静态元信息（frozen），question 是
    运行时数据；方法层接收 (spec, questions) 两参数。

    归一化原则：保留所有原始字段到 raw，但把"题面 / 选项 / 答案 / 完整解答"提到顶层，
    让方法层不必关心数据集自己的 column 命名。
    """

    id: str
    benchmark: str
    format: BenchmarkFormat
    prompt: str                       # 题面，不含答案
    answer: str                       # 标准化答案字符串（MC: 'A'/'B'...; math: 数值或 latex）
    choices: list[str] | None = None  # 仅 multiple_choice / cloze 有
    answer_index: int | None = None   # MC: 答案在 choices 中的下标（perm_option 用）
    # 完整解答/CoT 原文（math_cot 的 SFT 训练文本）。归一化承诺字段：各 benchmark 上游
    # 列名各异（gsm8k 'answer' 带 #### / math·math-500 'solution'），normalizer 统一映射
    # 到这里，MIA 类方法（SPV-MIA 等）读固定字段、不猜 raw key。无 CoT 的题型为 None。
    full_answer: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


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
    data_subset: str | None = None    # HF dataset config 名（如 gpqa_diamond / mgsm 的 'en'）
    note: str = ""
    # 对齐内部 P1 注册名单：tiers 决定是否进可信度报告主表
    # p1-pretrain / p1-sft 进主表；experimental / control 仅供对照与实验
    tiers: list[str] = field(default_factory=list)
    # pipeline 实际在哪些 stage 上对该 benchmark 跑检测，元素属于 {"base", "sft", "rlhf"}
    stage_targets: list[str] = field(default_factory=list)
    # 是否作为跨阶段 ΔScore 锚点（数学条目里只有 MATH 是 True）
    cross_stage_anchor: bool = False


@dataclass
class DetectionResult:
    """所有检测方法的统一返回。

    跨方法做差分（ΔScore / ΔMIA）时，归因层只读 signal 字段，
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
    delta_score: float | None
    delta_mia_auc: float | None
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
