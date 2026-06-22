"""模型推理统一接口。

所有检测方法只依赖 ModelInterface，不直接耦合 transformers / vllm / API。
不同后端（HF local、vLLM、远程 API）各自实现这个抽象。

调用方在使用进阶能力（logprobs / hidden_states）前必须先调 supports()
检查，以便方法层能 graceful degrade 而不是崩。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class Capability(str, Enum):
    """模型后端可能不支持的能力。"""

    LOGPROBS = "logprobs"              # 给定 prompt + completion，返回每 token 的 log p
    HIDDEN_STATES = "hidden_states"    # 各层 hidden states（MemLens 需要）
    LOGITS_LENS = "logits_lens"        # 中间层 unembedding 后的 logits（MemLens 严格需要）
    BATCH = "batch"                    # 支持批量推理
    GENERATE = "generate"              # 自由生成（黑盒方法必需）


class ModelInterface(ABC):
    """统一推理接口。具体后端继承本类。"""

    name: str
    stage_tag: str          # base / sft / rlhf
    n_layers: int | None = None  # MemLens / 探针需要

    @abstractmethod
    def supports(self, cap: Capability) -> bool:
        """检查后端是否支持某能力。不支持的方法应 graceful skip。"""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        """文本生成。所有后端必须实现。"""

    def batch_generate(
        self, prompts: list[str], max_tokens: int = 256, temperature: float = 0.0
    ) -> list[str]:
        """默认顺序回退；支持批量的后端应覆写。"""
        return [self.generate(p, max_tokens=max_tokens, temperature=temperature) for p in prompts]

    def logprobs(self, prompt: str, completion: str) -> np.ndarray:
        """返回 completion 各 token 的 log p。

        约定：长度 = tokenizer(completion) token 数；以自然对数为底。
        前置：supports(Capability.LOGPROBS) is True，否则抛 NotImplementedError。
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement logprobs; "
            "check supports(Capability.LOGPROBS) before calling."
        )

    def hidden_states(self, prompt: str) -> np.ndarray | None:
        """返回 shape=(n_layers+1, seq_len, hidden_dim) 的 hidden states。

        默认 None；HF local backend 应覆写。
        """
        return None

    def layer_logits(self, prompt: str, target_token: str) -> np.ndarray | None:
        """各层经 unembedding 后 target_token 的 logit / probability 轨迹（MemLens 用）。

        默认 None；只在 hidden_states + 共享 unembedding 可访问时实现。
        """
        return None
