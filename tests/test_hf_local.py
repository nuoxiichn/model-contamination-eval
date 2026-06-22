"""HFLocalModel smoke 测试。

用 hf-internal-testing/tiny-random-gpt2（5 层、随机初始化，~5MB）做端到端 wiring 验证。
不验证生成质量，只验证：
- 接口可调
- logprobs 长度对齐 completion token 数
- hidden_states 形状符合契约
- supports() 与实际能力一致

环境要求：
- transformers / torch 已装
- 能访问 hf-internal-testing/tiny-random-gpt2（默认走 HF_ENDPOINT 设置，国内
  建议 export HF_ENDPOINT=https://hf-mirror.com）

不可用时整文件 skip，不阻塞 CI。
"""

from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from model_contamination.models.base import Capability
from model_contamination.models.hf_local import HFLocalModel

MODEL_ID = "hf-internal-testing/tiny-random-gpt2"


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
    assert tiny_model.supports(Capability.HIDDEN_STATES)
    # MemLens 接口尚未与 hf_local 对接，约定 False
    assert not tiny_model.supports(Capability.LOGITS_LENS)


def test_stage_and_name(tiny_model: HFLocalModel) -> None:
    assert tiny_model.stage_tag == "base"
    assert tiny_model.name  # 非空


def test_n_layers_populated(tiny_model: HFLocalModel) -> None:
    assert isinstance(tiny_model.n_layers, int) and tiny_model.n_layers > 0


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


def test_hidden_states_shape(tiny_model: HFLocalModel) -> None:
    hs = tiny_model.hidden_states("hello world")
    assert hs is not None
    # shape: (n_layers+1, seq_len, hidden_dim)
    assert hs.ndim == 3
    assert hs.shape[0] == tiny_model.n_layers + 1
    assert hs.shape[1] >= 1
