"""Canary 双阶段注入与召回检测（结构性防御）。

约定见 docs/solutions/模型级污染检测-两阶段复用方案.md §3.4：
    预训练 canary：随机字符串嵌入自然语料，beam search 续写检测
    SFT canary：唯一 (instruction, response) 对，直接 prompt 查询

两组结构必须不同，否则无法区分阶段。
召回率差异 → 直接指向哪个阶段是主要污染源。

Phase 2 实装（依赖训练团队配合在数据准备阶段埋点）。
"""

from __future__ import annotations

from pathlib import Path

from model_contamination.models.base import ModelInterface
from model_contamination.types import DetectionResult


def measure_canary_recall(
    model: ModelInterface,
    canary_path: Path | str,
    canary_type: str,  # "pretrain" | "sft"
) -> DetectionResult:
    raise NotImplementedError(
        "Phase 2: 实装 canary 召回率测量。前置：数据准备阶段已注入对应 canary。"
    )


def generate_pretrain_canaries(
    n: int = 100,
    seed: int = 42,
    output_path: Path | str = "data/canary/pretrain.jsonl",
) -> None:
    """生成预训练 canary（嵌入自然语料的随机字符串）。"""
    raise NotImplementedError("Phase 2: 实装 canary 生成器")


def generate_sft_canaries(
    n: int = 100,
    seed: int = 42,
    output_path: Path | str = "data/canary/sft.jsonl",
) -> None:
    """生成 SFT canary（唯一 instruction-response 对）。"""
    raise NotImplementedError("Phase 2: 实装 canary 生成器")
