#!/usr/bin/env python3
"""把 sweep_results.jsonl 汇总成跨模型 ranked list + 每方法对照表。

无 positive control → 只出相对排名，不出绝对红黄绿（对齐 CLAUDE.md 红线）。

跑法：
    PYTHONPATH=src /opt/conda/bin/python \
      experiments/2026-07-08_market_model_sweep/summarize.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE / "outputs" / "sweep_results.jsonl"


def _load() -> list[dict]:
    if not _RESULTS.exists():
        raise SystemExit(f"没有结果文件：{_RESULTS}（先跑 run_sweep.py）")
    return [
        json.loads(l) for l in _RESULTS.read_text(encoding="utf-8").splitlines() if l.strip()
    ]


def _fmt(v) -> str:
    return f"{v:+.3f}" if isinstance(v, (int, float)) else "  -  "


def main() -> None:
    rows = _load()
    methods = ["codec", "mink_plus_plus", "paraphrase", "perm_option", "spv_mia"]
    # benchmark 列按固定语义顺序排（污染侧 → 干净侧 → MC-only），只显示实际出现过的
    _order = [
        "gsm8k", "math-500", "math", "gsm-plus", "mgsm",
        "mmlu-pro", "mmlu", "mmlu-cf", "gpqa", "gpqa-diamond", "mmmlu",
        "evalplus",
    ]
    present = {r["benchmark"] for r in rows}
    benches = [b for b in _order if b in present] + sorted(present - set(_order))
    models = sorted({r["model"] for r in rows})

    # signal[method][model][bench]
    sig: dict = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        sig[r["method"]][r["model"]][r["benchmark"]] = r["signal"]

    for method in methods:
        if method not in sig:
            continue
        print(f"\n=== {method} ===")
        header = f"{'model':22s} " + " ".join(f"{b:>10s}" for b in benches)
        print(header)
        print("-" * len(header))
        for m in models:
            cells = " ".join(f"{_fmt(sig[method][m].get(b)):>10s}" for b in benches)
            print(f"{m:22s} {cells}")

    # 跨模型 ranked list：以 codec 均值当主可疑度（自带对照、免 GT）
    print("\n=== 跨模型 ranked list（按 codec 均值降序，越高越可疑）===")
    rank = []
    for m in models:
        vals = [v for v in sig["codec"][m].values() if isinstance(v, (int, float))]
        if vals:
            rank.append((sum(vals) / len(vals), m))
    for score, m in sorted(rank, reverse=True):
        print(f"  {score:+.3f}  {m}")

    print("\n注：无 positive control，以上为相对排名 + 原始信号，非绝对裁决。")


if __name__ == "__main__":
    main()
