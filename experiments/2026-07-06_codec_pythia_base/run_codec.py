#!/usr/bin/env python3
"""CoDeC 小规模复现：Pythia-2.8b 上 seen(Pile) vs unseen 的 CoDeC 分离度。

论文 arXiv:2510.27055。核心结论：seen 数据 CoDeC≈100%，unseen<60%，清晰分离。

跑法（开发机，走 hf-mirror）：
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HOME=/mnt/public/code/chennuoxi/hf_cache \
    PYTHONPATH=src python3 experiments/2026-07-06_codec_pythia_base/run_codec.py

结果写到本目录 outputs/（不进 git）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

# 让脚本能 import model_contamination（假定用 PYTHONPATH=src，否则补上 src）
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from model_contamination.benchmarks.loader import load_questions  # noqa: E402
from model_contamination.benchmarks.registry import load_registry  # noqa: E402
from model_contamination.models.hf_local import HFLocalModel  # noqa: E402
from model_contamination.shared.codec import codec_detect  # noqa: E402
from model_contamination.types import (  # noqa: E402
    BenchmarkQuestion,
    BenchmarkSpec,
    Verdict,
)

_HERE = Path(__file__).resolve().parent
_CHUNK_CHARS = 600  # 论文：连续文本切 600 字符 chunk


def _text_spec(name: str) -> BenchmarkSpec:
    """给 seen 纯文本数据集造一个最小 spec（不进主 yaml）。"""
    return BenchmarkSpec(
        name=name, family="pile", format="open_generation", language="en",
        variants=[], applicable_methods=["codec"],
        trustworthiness_default=Verdict.DIRTY,  # seen 数据 ground-truth 已知污染
        data_source="hf", data_id="NeelNanda/pile-10k",
    )


def _load_pile_chunks(
    pile_set_name: str, n_samples: int
) -> list[BenchmarkQuestion]:
    """从 NeelNanda/pile-10k 取指定子集（meta.pile_set_name），切 600 字符 chunk。

    pile-10k 是 Pile 的公开 10k 采样（Pythia 训练语料），非 gated。
    """
    from datasets import load_dataset

    ds = load_dataset("NeelNanda/pile-10k", split="train")

    out: list[BenchmarkQuestion] = []
    for row in ds:
        meta = row.get("meta") or {}
        if meta.get("pile_set_name") != pile_set_name:
            continue
        text = row.get("text") or ""
        # 切 600 字符 chunk（论文做法）
        for start in range(0, len(text), _CHUNK_CHARS):
            chunk = text[start : start + _CHUNK_CHARS]
            if len(chunk) < _CHUNK_CHARS // 2:  # 丢弃过短尾块
                continue
            out.append(
                BenchmarkQuestion(
                    id=f"{pile_set_name}-{len(out)}",
                    benchmark=f"pile-{pile_set_name}",
                    format="open_generation", prompt=chunk, answer="",
                )
            )
            if len(out) >= n_samples:
                return out
    return out


def main() -> None:
    cfg = yaml.safe_load((_HERE / "run.yaml").read_text(encoding="utf-8"))
    codec_cfg = cfg["codec"]
    n_samples = codec_cfg["n_samples"]
    kwargs = dict(
        n_context=codec_cfg["n_context"],
        n_seeds=codec_cfg["n_seeds"],
        skip_first_tokens=codec_cfg["skip_first_tokens"],
    )

    print(f"[load] model {cfg['model']['path']}")
    model = HFLocalModel(
        model_path=cfg["model"]["path"],
        stage_tag=cfg["model"]["stage_tag"],
        name=cfg["model"]["name"],
    )

    reg = load_registry()
    results: list[dict] = []

    # ---- unseen ---- #
    for item in cfg["unseen"]:
        try:
            spec = reg.get(item["loader"])
            print(f"[unseen] loading {spec.name} (limit={n_samples})")
            qs = load_questions(spec, limit=n_samples)
            r = codec_detect(model, spec, qs, **kwargs)
            results.append(_row("unseen", spec.name, r))
            print(f"  -> signal={r.signal}  verdict={r.verdict_hint.value}  n={r.evidence.get('n_samples')}")
        except Exception as e:  # 单数据集失败不阻断其余
            print(f"  !! {item['key']} FAILED: {type(e).__name__}: {str(e)[:150]}")

    # ---- seen ---- #
    for item in cfg["seen"]:
        try:
            spec = _text_spec(item["key"])
            print(f"[seen] loading pile-10k/{item['pile_set_name']}")
            qs = _load_pile_chunks(item["pile_set_name"], n_samples)
            r = codec_detect(model, spec, qs, **kwargs)
            results.append(_row("seen", spec.name, r))
            print(f"  -> signal={r.signal}  verdict={r.verdict_hint.value}  n={r.evidence.get('n_samples')}")
        except Exception as e:
            print(f"  !! {item['key']} FAILED: {type(e).__name__}: {str(e)[:150]}")

    # ---- 分离度 ---- #
    seen_scores = [x["signal"] for x in results if x["group"] == "seen" and x["signal"] is not None]
    unseen_scores = [x["signal"] for x in results if x["group"] == "unseen" and x["signal"] is not None]
    summary = {
        "seen_mean": _mean(seen_scores),
        "unseen_mean": _mean(unseen_scores),
        "seen_min": min(seen_scores) if seen_scores else None,
        "unseen_max": max(unseen_scores) if unseen_scores else None,
        # 完全分离 = 所有 seen > 所有 unseen
        "cleanly_separated": (
            bool(seen_scores and unseen_scores and min(seen_scores) > max(unseen_scores))
        ),
    }

    out_dir = _HERE / "outputs"
    out_dir.mkdir(exist_ok=True)
    payload = {"config": cfg, "per_dataset": results, "summary": summary}
    (out_dir / "codec_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\n[done] wrote {out_dir / 'codec_results.json'}")


def _row(group: str, name: str, r) -> dict:
    return {
        "group": group,
        "dataset": name,
        "signal": r.signal,
        "verdict": r.verdict_hint.value,
        "prerequisites_met": r.prerequisites_met,
        "evidence": r.evidence,
        "error": r.error,
    }


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


if __name__ == "__main__":
    main()
