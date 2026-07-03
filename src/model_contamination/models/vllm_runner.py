"""vLLM 后端（高吞吐，远程机首选）。

Phase 2 实装。vLLM 0.5+ 已支持 logprobs。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from model_contamination.models.base import Capability, ModelInterface


class VLLMModel(ModelInterface):
    def __init__(self, model_path: str | Path, stage_tag: str, **kwargs):
        self.model_path = Path(model_path)
        self.stage_tag = stage_tag
        self.name = self.model_path.name
        raise NotImplementedError("Phase 2: 实装 vLLM 后端")

    def supports(self, cap: Capability) -> bool:
        return cap in {Capability.LOGPROBS, Capability.BATCH, Capability.GENERATE}

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        raise NotImplementedError

    def logprobs(self, prompt: str, completion: str) -> np.ndarray:
        raise NotImplementedError
