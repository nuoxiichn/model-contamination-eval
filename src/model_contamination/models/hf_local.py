"""HuggingFace transformers 本地后端。

实现 ModelInterface 全部抽象方法。lazy import torch / transformers，避免
仅做 metadata 操作的代码也吃这些重 dep。

关键约定：
- logprobs(prompt, completion) 通过 "再编码一次 prompt+completion" 拿到对齐
  offset；不假设 tokenizer(completion) 与 tokenizer(prompt+completion) 的尾部
  token 一致（BPE 边界问题）
- generate temperature=0 走 greedy（do_sample=False）
- 默认 padding_side='left'，便于左 padding 的 generation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from model_contamination.models.base import Capability, ModelInterface


class HFLocalModel(ModelInterface):
    """transformers.AutoModelForCausalLM 后端。"""

    def __init__(
        self,
        model_path: str | Path,
        stage_tag: str,
        *,
        device: str | None = None,
        dtype: str | None = None,
        name: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "需要 `uv sync --extra hf` 才能用 HFLocalModel"
            ) from e

        self.model_path = Path(model_path) if not str(model_path).startswith("http") else model_path
        self.stage_tag = stage_tag
        self.name = name or (
            self.model_path.name if isinstance(self.model_path, Path) else str(self.model_path)
        )

        self._torch = torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            self._dtype = torch.float16 if self._device.startswith("cuda") else torch.float32
        else:
            self._dtype = getattr(torch, dtype)

        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), trust_remote_code=trust_remote_code
        )
        if self._tokenizer.pad_token_id is None:
            # 多数 causal LM tokenizer 无 pad token；复用 eos 保证 batch 推理可跑
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        self._model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=self._dtype,
            trust_remote_code=trust_remote_code,
        ).to(self._device)
        self._model.eval()

        cfg = self._model.config
        self._max_position = (
            getattr(cfg, "max_position_embeddings", None)
            or getattr(cfg, "n_positions", None)
            or getattr(cfg, "n_ctx", None)
        )

    # ----------------------------- capabilities ----------------------------- #

    def supports(self, cap: Capability) -> bool:
        return cap in {
            Capability.GENERATE,
            Capability.BATCH,
            Capability.LOGPROBS,
            Capability.TOKEN_DIST_STATS,
        }

    # ----------------------------- generation ----------------------------- #

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        return self.batch_generate([prompt], max_tokens=max_tokens, temperature=temperature)[0]

    def batch_generate(
        self, prompts: list[str], max_tokens: int = 256, temperature: float = 0.0
    ) -> list[str]:
        if not prompts:
            return []
        # 留 max_tokens 给生成；prompts 超长则左截断（保尾部，含 question 末尾的 "Answer:"）
        max_in = None
        if self._max_position is not None:
            max_in = max(1, self._max_position - max_tokens)
        enc = self._tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=max_in is not None,
            max_length=max_in,
        )
        input_ids = enc["input_ids"].to(self._device)
        attn = enc["attention_mask"].to(self._device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if temperature == 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature

        with self._torch.no_grad():
            out = self._model.generate(input_ids=input_ids, attention_mask=attn, **gen_kwargs)
        # 去掉 prompt 部分（左 padding，所以从 input_ids.shape[1] 起就是生成内容）
        gen_only = out[:, input_ids.shape[1] :]
        return self._tokenizer.batch_decode(gen_only, skip_special_tokens=True)

    # ----------------------------- logprobs ----------------------------- #

    def logprobs(self, prompt: str, completion: str) -> np.ndarray:
        """返回 completion 各 token 的 log p（自然对数）。

        正确处理 BPE 边界：编码 prompt 和 prompt+completion 两次，差额位置即
        completion 的 token 范围。模型在位置 i 输出的 logits 预测的是位置 i+1
        的 token，所以 completion 的 logprob 取 logits[:, prompt_len-1:-1]。
        """
        prompt_ids = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
        full_ids = self._tokenizer(
            prompt + completion, return_tensors="pt", add_special_tokens=True
        )["input_ids"]
        prompt_len = prompt_ids.shape[1]
        full_len = full_ids.shape[1]
        if full_len <= prompt_len:
            return np.zeros(0, dtype=np.float64)

        # 截断超出 context 的部分：保留最后 max_position 个 token（completion 尾部最重要）。
        # 仍保证 prompt 部分至少 1 token 留作 conditioning。
        if self._max_position is not None and full_len > self._max_position:
            drop = full_len - self._max_position
            full_ids = full_ids[:, drop:]
            prompt_len = max(1, prompt_len - drop)
            full_len = full_ids.shape[1]

        full_ids = full_ids.to(self._device)
        with self._torch.no_grad():
            logits = self._model(input_ids=full_ids).logits  # (1, full_len, vocab)
        log_probs = self._torch.log_softmax(logits.float(), dim=-1)

        # 预测位置 i+1 的 token 用位置 i 的 logits；completion token index = [prompt_len, full_len)
        target_ids = full_ids[0, prompt_len:full_len]                  # (n_completion,)
        pred_logits = log_probs[0, prompt_len - 1 : full_len - 1, :]   # (n_completion, vocab)
        token_logp = pred_logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        return token_logp.detach().cpu().numpy().astype(np.float64)

    def next_token_logprobs(
        self, prompt: str, candidate_completions: list[str]
    ) -> np.ndarray:
        """1 次 prompt forward + 词表 lookup 同时算多个 candidate 的 logp 和。

        快路径：所有 candidate 在 prompt 后延续都是同一起始位置的 1 个 token
        （例：" A"/" B" 各 1 token）→ 1 次 prompt forward 后从末位 log_softmax
        里 gather 出各 candidate 的 token id 即可，避免 N 次前向。

        慢路径：任一 candidate 分词后 > 1 token（多字节字符 / 未成子词） →
        回退到 base 类的逐 candidate 循环。

        约定：多 token candidate 的返回值 = 所有 token log p 之和，与 base 类
        默认实现（sum(logprobs)）等价。
        """
        if not candidate_completions:
            return np.zeros(0, dtype=np.float64)

        # 先编码 prompt 与 prompt+candidate，判断每个 candidate 是否为 1-token 延续
        prompt_ids = self._tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        )["input_ids"]
        prompt_len = prompt_ids.shape[1]

        cand_token_ids: list[int | None] = []
        for c in candidate_completions:
            full = self._tokenizer(
                prompt + c, return_tensors="pt", add_special_tokens=True
            )["input_ids"]
            if full.shape[1] == prompt_len + 1:
                cand_token_ids.append(int(full[0, prompt_len].item()))
            else:
                cand_token_ids.append(None)  # multi-token → slow path

        # 若全为单 token：走快路径，1 次 forward
        if all(tid is not None for tid in cand_token_ids):
            # context 守护：prompt 超长时按 logprobs 同样规则左截断
            trimmed_ids = prompt_ids
            if self._max_position is not None and prompt_len > self._max_position:
                trimmed_ids = prompt_ids[:, prompt_len - self._max_position :]
            trimmed_ids = trimmed_ids.to(self._device)
            with self._torch.no_grad():
                logits = self._model(input_ids=trimmed_ids).logits  # (1, L, vocab)
            last_logp = self._torch.log_softmax(logits[0, -1, :].float(), dim=-1)
            picks = self._torch.tensor(cand_token_ids, device=self._device)
            out = last_logp.index_select(0, picks).detach().cpu().numpy().astype(np.float64)
            return out

        # 混合场景：逐 candidate 回退
        out = np.empty(len(candidate_completions), dtype=np.float64)
        for i, c in enumerate(candidate_completions):
            out[i] = float(np.sum(self.logprobs(prompt, c)))
        return out

    # ----------------------------- token-level distribution stats ----------------------------- #

    def token_logprob_stats(self, prompt: str, completion: str) -> dict[str, np.ndarray]:
        """Min-K%++ 用：单次 forward 同时返回 chosen_logp / μ / σ。

        μ_i = Σ_v p_i(v) log p_i(v) ；σ_i = sqrt(Σ_v p_i(v) (log p_i(v) - μ_i)^2)。
        通过 log_softmax 一次性算，不保留 full vocab tensor 在 host，省内存。
        """
        prompt_ids = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
        full_ids = self._tokenizer(
            prompt + completion, return_tensors="pt", add_special_tokens=True
        )["input_ids"]
        prompt_len = prompt_ids.shape[1]
        full_len = full_ids.shape[1]
        if full_len <= prompt_len:
            empty = np.zeros(0, dtype=np.float64)
            return {"chosen_logp": empty, "mu": empty.copy(), "sigma": empty.copy()}

        # 与 logprobs 一致的 context-length 守护：超长保留尾部
        if self._max_position is not None and full_len > self._max_position:
            drop = full_len - self._max_position
            full_ids = full_ids[:, drop:]
            prompt_len = max(1, prompt_len - drop)
            full_len = full_ids.shape[1]

        full_ids = full_ids.to(self._device)
        with self._torch.no_grad():
            logits = self._model(input_ids=full_ids).logits  # (1, full_len, vocab)
            # log_p, p 都保留 fp32 防止 bf16 vocab 求和误差
            log_p = self._torch.log_softmax(logits.float(), dim=-1)
            p = log_p.exp()
            # μ_i = E[log p]；σ_i^2 = E[(log p - μ)^2]
            mu = (p * log_p).sum(dim=-1)                      # (1, full_len)
            var = (p * (log_p - mu.unsqueeze(-1)) ** 2).sum(dim=-1)
            sigma = var.clamp(min=0.0).sqrt()

            # completion 位置：[prompt_len, full_len) ；对应预测位置 [prompt_len-1, full_len-1)
            target_ids = full_ids[0, prompt_len:full_len]
            pred_logp = log_p[0, prompt_len - 1 : full_len - 1, :]
            chosen = pred_logp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            mu_aligned = mu[0, prompt_len - 1 : full_len - 1]
            sigma_aligned = sigma[0, prompt_len - 1 : full_len - 1]

        return {
            "chosen_logp": chosen.detach().cpu().numpy().astype(np.float64),
            "mu": mu_aligned.detach().cpu().numpy().astype(np.float64),
            "sigma": sigma_aligned.detach().cpu().numpy().astype(np.float64),
        }
