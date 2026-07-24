"""HFLocalModel smoke 测试。

用 hf-internal-testing/tiny-random-gpt2（5 层、随机初始化，~5MB）做端到端 wiring 验证。
不验证生成质量，只验证：
- 接口可调
- logprobs 长度对齐 completion token 数
- supports() 与实际能力一致

环境要求：
- transformers / torch 已装
- 能访问 hf-internal-testing/tiny-random-gpt2（默认走 HF_ENDPOINT 设置，国内
  建议 export HF_ENDPOINT=https://hf-mirror.com）

不可用时整文件 skip，不阻塞 CI。
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

if os.environ.get("RUN_HF_SMOKE") != "1":
    pytest.skip(
        "HF model smoke is opt-in; set RUN_HF_SMOKE=1 to download/run it",
        allow_module_level=True,
    )

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from model_contamination.models.base import Capability  # noqa: E402
from model_contamination.models.hf_local import HFLocalModel  # noqa: E402

MODEL_ID = "hf-internal-testing/tiny-random-gpt2"


class _WhitespaceDroppingTokenizer:
    """Minimal tokenizer reproducing DeepSeek's empty ``"\n\n"`` encoding."""

    bos_token_id = 1
    eos_token_id = 2

    def __call__(self, text, *, return_tensors=None, add_special_tokens=True):
        del add_special_tokens
        ids = [3 + (ord(char) % 10) for char in text if not char.isspace()]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}


class _FullSequenceModel:
    config = SimpleNamespace(vocab_size=16, bos_token_id=1, eos_token_id=2)

    def __call__(self, *, input_ids, **kwargs):
        del kwargs
        batch, seq_len = input_ids.shape
        logits = torch.arange(16, dtype=torch.float32).repeat(batch, seq_len, 1)
        return SimpleNamespace(logits=logits)


def _empty_prompt_model() -> HFLocalModel:
    model = object.__new__(HFLocalModel)
    model.name = "empty-prompt-test"
    model._torch = torch
    model._tokenizer = _WhitespaceDroppingTokenizer()
    model._model = _FullSequenceModel()
    model._device = "cpu"
    model._max_position = None
    return model


@pytest.fixture(scope="module")
def tiny_model():
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    try:
        return HFLocalModel(model_path=MODEL_ID, stage_tag="base", device="cpu", dtype="float32")
    except Exception as e:  # pragma: no cover
        pytest.skip(f"tiny model unavailable: {e}")


def test_supports(tiny_model: HFLocalModel) -> None:
    assert tiny_model.supports(Capability.GENERATE)
    assert tiny_model.supports(Capability.BATCH)
    assert tiny_model.supports(Capability.LOGPROBS)
    assert tiny_model.supports(Capability.TOKEN_DIST_STATS)


def test_stage_and_name(tiny_model: HFLocalModel) -> None:
    assert tiny_model.stage_tag == "base"
    assert tiny_model.name  # 非空


def test_generate_returns_string(tiny_model: HFLocalModel) -> None:
    out = tiny_model.generate("Hello", max_tokens=4, temperature=0.0)
    assert isinstance(out, str)


def test_batch_generate_preserves_order(tiny_model: HFLocalModel) -> None:
    outs = tiny_model.batch_generate(["one", "two", "three"], max_tokens=3, temperature=0.0)
    assert len(outs) == 3
    assert all(isinstance(o, str) for o in outs)


def test_logprobs_length_matches_completion_tokens(tiny_model: HFLocalModel) -> None:
    prompt, completion = "The answer is", " 42 indeed"
    lp = tiny_model.logprobs(prompt, completion)
    expected_len = (
        len(tiny_model._tokenizer(prompt + completion, add_special_tokens=True)["input_ids"])
        - len(tiny_model._tokenizer(prompt, add_special_tokens=True)["input_ids"])
    )
    assert lp.shape == (expected_len,)
    # log p 必为 ≤ 0
    assert np.all(lp <= 0.0)


def test_logprobs_empty_completion(tiny_model: HFLocalModel) -> None:
    lp = tiny_model.logprobs("hi", "")
    assert lp.shape == (0,)


def test_whitespace_only_prompt_gets_bos_conditioning_token() -> None:
    """DeepSeek drops ``"\n\n"``; CoDeC/Min-K++ must still score all tokens."""
    model = _empty_prompt_model()
    prompt, completion = "\n\n", "AB"

    lp = model.logprobs(prompt, completion)
    stats = model.token_logprob_stats(prompt, completion)
    sums = model.seq_logprob_sums(prompt, [completion, "CD"])

    assert lp.shape == (2,)
    assert stats["chosen_logp"].shape == (2,)
    assert np.allclose(stats["chosen_logp"], lp)
    assert sums.shape == (2,)
    assert np.isclose(sums[0], lp.sum())


def test_token_logprob_stats_shapes_and_signs(tiny_model: HFLocalModel) -> None:
    """同一次 forward 应返回 chosen_logp / μ / σ 三个等长数组。"""
    prompt, completion = "The answer is", " 42 indeed"
    stats = tiny_model.token_logprob_stats(prompt, completion)
    assert set(stats) == {"chosen_logp", "mu", "sigma"}
    n = stats["chosen_logp"].shape[0]
    assert n > 0
    assert stats["mu"].shape == (n,)
    assert stats["sigma"].shape == (n,)
    # μ = E[log p] ≤ 0；σ ≥ 0
    assert np.all(stats["mu"] <= 1e-6)
    assert np.all(stats["sigma"] >= 0.0)
    # chosen_logp 与 logprobs() 应一致（同一次 forward 抽取的 token logp）
    lp = tiny_model.logprobs(prompt, completion)
    assert np.allclose(stats["chosen_logp"], lp, atol=1e-5)
    # 归一化值有限
    safe_sigma = np.where(stats["sigma"] > 1e-8, stats["sigma"], np.nan)
    normalized = (stats["chosen_logp"] - stats["mu"]) / safe_sigma
    assert np.all(np.isfinite(normalized[~np.isnan(normalized)]))


def test_token_logprob_stats_empty_completion(tiny_model: HFLocalModel) -> None:
    stats = tiny_model.token_logprob_stats("hi", "")
    assert stats["chosen_logp"].shape == (0,)
    assert stats["mu"].shape == (0,)
    assert stats["sigma"].shape == (0,)


def test_next_token_logprobs_batch_matches_loop(tiny_model: HFLocalModel) -> None:
    """快路径（1 forward + lookup）与慢路径（N 次 logprobs）应返回同一组 logp。

    覆盖 perm_option 的核心 speedup 路径：多个单 token candidate 走批量。
    """
    prompt = "The next word is"
    candidates = [" a", " b", " c", " d"]  # 均预期为 1 token 延续
    batched = tiny_model.next_token_logprobs(prompt, candidates)
    assert batched.shape == (len(candidates),)
    reference = np.array(
        [float(np.sum(tiny_model.logprobs(prompt, c))) for c in candidates],
        dtype=np.float64,
    )
    assert np.allclose(batched, reference, atol=1e-5)


def test_next_token_logprobs_multi_token_falls_back(tiny_model: HFLocalModel) -> None:
    """混合场景：一个 candidate 分词后 >1 token → 走慢路径回退，结果仍与 logprobs 一致。"""
    prompt = "Say something:"
    candidates = [" a", " unbelievable"]  # 后者一般 >1 subword
    out = tiny_model.next_token_logprobs(prompt, candidates)
    assert out.shape == (2,)
    ref = np.array(
        [float(np.sum(tiny_model.logprobs(prompt, c))) for c in candidates],
        dtype=np.float64,
    )
    assert np.allclose(out, ref, atol=1e-5)


def test_next_token_logprobs_empty_candidate_list(tiny_model: HFLocalModel) -> None:
    out = tiny_model.next_token_logprobs("prompt", [])
    assert out.shape == (0,)
