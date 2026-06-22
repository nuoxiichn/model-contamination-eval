"""HuggingFace transformers 本地后端。

Phase 2 实装。Phase 1 阶段只需要 stub，pipeline 可以走 mock 模型先打通 plumbing。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from model_contamination.models.base import Capability, ModelInterface


class HFLocalModel(ModelInterface):
    def __init__(self, model_path: str | Path, stage_tag: str, **kwargs):
        self.model_path = Path(model_path)
        self.stage_tag = stage_tag
        self.name = self.model_path.name
        raise NotImplementedError(
            "Phase 2: 实装 HF transformers 后端。需要 lazy import torch/transformers。"
        )

    def supports(self, cap: Capability) -> bool:
        return cap in {
            Capability.LOGPROBS,
            Capability.HIDDEN_STATES,
            Capability.LOGITS_LENS,
            Capability.BATCH,
            Capability.GENERATE,
        }

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        raise NotImplementedError

    def logprobs(self, prompt: str, completion: str) -> np.ndarray:
        raise NotImplementedError
