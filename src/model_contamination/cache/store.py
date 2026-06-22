"""文件系统缓存：按 (model_name, benchmark_name, input_hash) 索引。

logprobs 是最贵的计算，跨方法可复用：Oren 与 Min-K%++ 都吃同一份 logprobs。
此缓存让差分归因（base vs SFT vs RLHF）和方法迭代不重算。

约定：
    cache_dir/
        ├── {model_name}/
        │   ├── {benchmark_name}/
        │   │   ├── logprobs_{input_hash}.npz
        │   │   └── generate_{input_hash}.jsonl
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


class CacheStore:
    def __init__(self, cache_dir: Path | str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_path(self, model_name: str, benchmark: str, kind: str, input_hash: str) -> Path:
        d = self.cache_dir / _safe(model_name) / _safe(benchmark)
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{kind}_{input_hash}.npz"

    def get_logprobs(
        self, model_name: str, benchmark: str, inputs: list[str]
    ) -> np.ndarray | None:
        h = _hash_inputs(inputs)
        path = self._key_path(model_name, benchmark, "logprobs", h)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=True) as data:
            return data["arr"]

    def put_logprobs(
        self,
        model_name: str,
        benchmark: str,
        inputs: list[str],
        logprobs: np.ndarray,
    ) -> None:
        h = _hash_inputs(inputs)
        path = self._key_path(model_name, benchmark, "logprobs", h)
        np.savez_compressed(path, arr=logprobs)

    def get_generate(
        self, model_name: str, benchmark: str, inputs: list[str]
    ) -> list[str] | None:
        h = _hash_inputs(inputs)
        path = self._key_path(model_name, benchmark, "generate", h).with_suffix(".jsonl")
        if not path.exists():
            return None
        return [json.loads(line)["text"] for line in path.read_text().splitlines()]

    def put_generate(
        self,
        model_name: str,
        benchmark: str,
        inputs: list[str],
        generations: list[str],
    ) -> None:
        h = _hash_inputs(inputs)
        path = self._key_path(model_name, benchmark, "generate", h).with_suffix(".jsonl")
        path.write_text(
            "\n".join(json.dumps({"text": g}, ensure_ascii=False) for g in generations)
        )


def _hash_inputs(inputs: list[str]) -> str:
    h = hashlib.sha256()
    for s in inputs:
        h.update(s.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _safe(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")
