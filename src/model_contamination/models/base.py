"""模型推理统一接口。

所有检测方法只依赖 ModelInterface，不直接耦合 transformers / vllm / API。
不同后端（HF local、vLLM、远程 API）各自实现这个抽象。

调用方在使用进阶能力（logprobs / token_logprob_stats）前必须先调 supports()
检查，以便方法层能 graceful degrade 而不是崩。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class Capability(str, Enum):
    """模型后端可能不支持的能力。"""

    LOGPROBS = "logprobs"              # 给定 prompt + completion，返回每 token 的 log p
    TOKEN_DIST_STATS = "token_dist_stats"  # 每个位置全 vocab 分布的均值/方差（Min-K%++ 需要）
    BATCH = "batch"                    # 支持批量推理
    GENERATE = "generate"              # 自由生成（黑盒方法必需）


class ModelInterface(ABC):
    """统一推理接口。具体后端继承本类。"""

    name: str
    stage_tag: str          # base / sft / rlhf

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

    def next_token_logprobs(
        self, prompt: str, candidate_completions: list[str]
    ) -> np.ndarray:
        """一次评估同一 prompt 下多个 completion 的 log p 之和。

        用于 perm_option 这类"同题多候选"场景（10 个字母只做 1 次 forward
        而不是 10 次）。返回 shape=(len(candidate_completions),)，每格是
        对应 completion 各 token log p 之和（≈ logprobs(prompt, c).sum()）。

        默认实现：逐 candidate 回退到 logprobs()（正确但慢）。
        HFLocalModel 等具体后端应 override，对全为单 token 的 candidates
        做 1 次 forward + 词表 lookup。

        前置：supports(Capability.LOGPROBS) is True。
        """
        out = np.empty(len(candidate_completions), dtype=np.float64)
        for i, c in enumerate(candidate_completions):
            out[i] = float(np.sum(self.logprobs(prompt, c)))
        return out

    def token_logprob_stats(self, prompt: str, completion: str) -> dict[str, np.ndarray]:
        """返回 completion 每个 token 位置的 (chosen_logp, mu, sigma)。

        定义（Min-K%++, Zhang et al. 2024, arXiv:2404.02936）：
            对位置 i，模型在 x_<i 下的分布 p_i(v)
            chosen_logp[i] = log p_i(x_i)
            mu[i]    = E_{v~p_i}[log p_i(v)] = Σ_v p_i(v) log p_i(v)   (= -H(p_i))
            sigma[i] = sqrt(Var_{v~p_i}[log p_i(v)])

        返回 dict 含三个长度同 chosen logp 的 np.ndarray（自然对数）。

        前置：supports(Capability.TOKEN_DIST_STATS) is True。
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement token_logprob_stats; "
            "check supports(Capability.TOKEN_DIST_STATS) before calling."
        )
