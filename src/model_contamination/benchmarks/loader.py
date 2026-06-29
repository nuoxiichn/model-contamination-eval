"""Benchmark 题目加载层。

把 BenchmarkSpec（yaml 元信息）+ 数据集 column → 归一化 BenchmarkQuestion。

每个 benchmark 的 column 命名各异（mmlu 用 'choices'/'answer'，gsm8k 用 'answer'
拼 chain-of-thought，math 用 'problem'/'solution' 等）。loader 不做通用映射，按
benchmark 名 dispatch 到具体 normalizer；未注册的 normalizer 显式 raise，让
新 benchmark 接入路径透明可控。

第一版 normalizer 覆盖 sanity-check 路径需要的 5 个：
    gsm8k / math / math-500 / mmlu-pro / mmlu-cf
其余 benchmark 报 NotImplementedError，附说明。
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable
from typing import Any

from model_contamination.types import BenchmarkQuestion, BenchmarkSpec

Normalizer = Callable[[dict[str, Any], int, BenchmarkSpec], BenchmarkQuestion]


def load_questions(
    spec: BenchmarkSpec,
    *,
    split: str = "test",
    limit: int | None = None,
    subset: str | None = None,
    indices: list[int] | None = None,
) -> list[BenchmarkQuestion]:
    """从 spec.data_id 加载题目并归一化。

    参数:
        spec:    benchmark 元信息
        split:   HF dataset split，默认 'test'；部分数据集要传 'validation'
        limit:   截断前 N 条，None 全量；与 indices 互斥
        subset:  HF dataset 子集名（如 mmlu 的学科），None 时按 normalizer 默认
        indices: 指定 HF 行号子集（与 SFT 注入 manifest 配套使用），按给定顺序取；
                 与 limit 互斥。out-of-range 的下标直接报错，不静默跳过。

    返回:
        list[BenchmarkQuestion]
    """
    if limit is not None and indices is not None:
        raise ValueError("limit 与 indices 互斥，同时给出会产生歧义")
    if spec.data_id == "TBD":
        raise ValueError(
            f"benchmark {spec.name} 的 data_id 未确定（yaml 中为 'TBD'），"
            "需先在 configs/benchmarks.yaml 补全 HF 路径"
        )

    if spec.data_source == "local":
        return _load_local(spec, limit=limit)
    if spec.data_source == "livebench-api":
        raise NotImplementedError(
            f"{spec.name}: livebench-api 后端待实装（Phase 2+）"
        )
    if spec.data_source == "modelscope":
        raise NotImplementedError(
            f"{spec.name}: modelscope 后端待实装；目前先用 HF 镜像"
        )
    if spec.data_source != "hf":
        raise ValueError(f"未知 data_source: {spec.data_source}")

    normalizer = _NORMALIZERS.get(spec.name)
    if normalizer is None:
        raise NotImplementedError(
            f"benchmark {spec.name} 尚未注册 normalizer。"
            f"在 loader.py 的 _NORMALIZERS 添加 '{spec.name}' → 归一化函数。"
            f"参考已实现：{sorted(_NORMALIZERS)}"
        )

    # 懒加载 datasets，让仅做 metadata 操作时不必依赖 hf extra
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "需要 `uv sync --extra hf` 才能加载 HF 数据集"
        ) from e

    load_kwargs: dict[str, Any] = {}
    if subset is not None:
        load_kwargs["name"] = subset
    elif spec.data_subset is not None:
        load_kwargs["name"] = spec.data_subset

    ds = load_dataset(spec.data_id, split=split, **load_kwargs)

    out: list[BenchmarkQuestion] = []
    if indices is not None:
        n = len(ds)
        for idx in indices:
            if not 0 <= idx < n:
                raise IndexError(
                    f"{spec.name}: manifest idx {idx} 超出 dataset 范围 [0, {n})"
                )
            out.append(normalizer(ds[idx], idx, spec))
        return out

    for idx, row in enumerate(_take(ds, limit)):
        out.append(normalizer(row, idx, spec))
    return out


def _take(iterable: Iterable[dict[str, Any]], limit: int | None) -> Iterable[dict[str, Any]]:
    if limit is None:
        yield from iterable
        return
    for i, x in enumerate(iterable):
        if i >= limit:
            return
        yield x


# ----------------------------- normalizers ----------------------------- #
# 每个函数签名: (row, idx, spec) -> BenchmarkQuestion
# row 是 datasets 单行 dict；idx 用于生成稳定 id；spec 用于 benchmark / format 字段。

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _normalize_gsm8k(row: dict[str, Any], idx: int, spec: BenchmarkSpec) -> BenchmarkQuestion:
    """gsm8k: question + answer（含 '#### {数值}' 形式的最终答案）"""
    raw_answer = row["answer"]
    m = re.search(r"####\s*([\-0-9./,]+)", raw_answer)
    final = m.group(1).replace(",", "").strip() if m else raw_answer.strip()
    return BenchmarkQuestion(
        id=f"gsm8k-{idx}",
        benchmark=spec.name,
        format="math_cot",
        prompt=row["question"],
        answer=final,
        raw={"full_answer": raw_answer},
    )


def _normalize_math(row: dict[str, Any], idx: int, spec: BenchmarkSpec) -> BenchmarkQuestion:
    """hendrycks/competition_math: problem + solution，答案在 \\boxed{...}"""
    solution = row.get("solution", "")
    final = _extract_boxed(solution) or solution
    return BenchmarkQuestion(
        id=f"math-{idx}",
        benchmark=spec.name,
        format="math_cot",
        prompt=row["problem"],
        answer=final,
        raw={"level": row.get("level"), "type": row.get("type"), "solution": solution},
    )


def _normalize_math_500(row: dict[str, Any], idx: int, spec: BenchmarkSpec) -> BenchmarkQuestion:
    """HuggingFaceH4/MATH-500: problem + answer（已抽取） + solution"""
    return BenchmarkQuestion(
        id=f"math500-{idx}",
        benchmark=spec.name,
        format="math_cot",
        prompt=row["problem"],
        answer=str(row.get("answer", "")).strip(),
        raw={"level": row.get("level"), "subject": row.get("subject"),
             "solution": row.get("solution", "")},
    )


def _normalize_mmlu_pro(row: dict[str, Any], idx: int, spec: BenchmarkSpec) -> BenchmarkQuestion:
    """TIGER-Lab/MMLU-Pro: question + options (list) + answer (letter) + answer_index"""
    options = list(row["options"])
    ans_idx = int(row["answer_index"])
    return BenchmarkQuestion(
        id=f"mmlupro-{idx}",
        benchmark=spec.name,
        format="multiple_choice",
        prompt=row["question"],
        choices=options,
        answer=_LETTERS[ans_idx],
        answer_index=ans_idx,
        raw={"category": row.get("category"), "src": row.get("src")},
    )


def _normalize_mmlu_cf(row: dict[str, Any], idx: int, spec: BenchmarkSpec) -> BenchmarkQuestion:
    """microsoft/MMLU-CF: Question + A/B/C/D + Answer（字母）"""
    choices = [row["A"], row["B"], row["C"], row["D"]]
    letter = str(row["Answer"]).strip().upper()
    ans_idx = _LETTERS.index(letter) if letter in _LETTERS else 0
    return BenchmarkQuestion(
        id=f"mmlucf-{idx}",
        benchmark=spec.name,
        format="multiple_choice",
        prompt=row["Question"],
        choices=choices,
        answer=letter,
        answer_index=ans_idx,
        raw={"subject": row.get("Subject"), "category": row.get("Category")},
    )


def _normalize_evalplus(row: dict[str, Any], idx: int, spec: BenchmarkSpec) -> BenchmarkQuestion:
    """evalplus/humanevalplus: task_id + prompt (docstring+签名) + canonical_solution。

    answer 字段放 canonical_solution（函数体）；prompt 是 docstring+签名。
    Min-K%++ 需要把 prompt+answer 当作一个完整文本来检测，所以这种切分
    与 prep_bench_to_sft.py 的注入形态一致。
    """
    return BenchmarkQuestion(
        id=f"evalplus-{idx}",
        benchmark=spec.name,
        format="code_completion",
        prompt=row["prompt"],
        answer=row["canonical_solution"],
        raw={"task_id": row.get("task_id"), "entry_point": row.get("entry_point")},
    )


_NORMALIZERS: dict[str, Normalizer] = {
    "gsm8k": _normalize_gsm8k,
    "math": _normalize_math,
    "math-500": _normalize_math_500,
    "mmlu-pro": _normalize_mmlu_pro,
    "mmlu-cf": _normalize_mmlu_cf,
    "evalplus": _normalize_evalplus,
}


def _extract_boxed(s: str) -> str | None:
    """从 latex \\boxed{...} 提取最外层内容。"""
    i = s.find("\\boxed{")
    if i < 0:
        return None
    depth = 0
    start = i + len("\\boxed{")
    for j in range(start, len(s)):
        c = s[j]
        if c == "{":
            depth += 1
        elif c == "}":
            if depth == 0:
                return s[start:j].strip()
            depth -= 1
    return None


def _load_local(spec: BenchmarkSpec, limit: int | None) -> list[BenchmarkQuestion]:
    """本地 jsonl 加载（gsm1k 等）。"""
    import json
    from pathlib import Path

    path = Path(spec.data_id)
    if not path.is_absolute():
        path = Path(os.getcwd()) / path
    if not path.exists():
        raise FileNotFoundError(
            f"{spec.name}: 本地数据文件不存在 {path}。"
            "若该 benchmark 不在仓库里，需手动放置到 data/ 目录下"
        )

    out: list[BenchmarkQuestion] = []
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            row = json.loads(line)
            # gsm1k 与 gsm8k 同 schema
            if spec.name == "gsm1k":
                out.append(_normalize_gsm8k(row, idx, spec))
            else:
                raise NotImplementedError(
                    f"本地 benchmark {spec.name} 尚未实装 normalizer"
                )
    return out
