"""从 outputs/results/<model>__<bench>.json 汇总成统一 scenario_b.csv。

数据并行分片各写各的 CSV，per-cell JSON 才是权威来源。本脚本 glob 所有 JSON，
重建一张完整表（含 1.5B / 7B / 72B 全部 model×bench）。幂等，随时可重跑。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "outputs" / "results"

FIELDS = [
    "model", "benchmark", "signal_leak_fraction", "verdict",
    "n_questions", "mean_perms_per_q", "primary_threshold",
    "leak_fraction_by_threshold", "prerequisites_met", "error",
]


def main() -> None:
    rows: list[dict] = []
    for p in sorted(RESULTS.glob("*__*.json")):
        r = json.loads(p.read_text())
        ev = r.get("evidence") or {}
        verdict = r.get("verdict_hint")
        if isinstance(verdict, dict):
            verdict = verdict.get("value")
        rows.append({
            "model": p.stem.split("__")[0],
            "benchmark": r.get("benchmark"),
            "signal_leak_fraction": r.get("signal"),
            "verdict": verdict,
            "n_questions": ev.get("n_questions"),
            "mean_perms_per_q": ev.get("mean_perms_per_q"),
            "primary_threshold": ev.get("primary_threshold"),
            "leak_fraction_by_threshold": json.dumps(
                ev.get("leak_fraction_by_threshold"), ensure_ascii=False
            ),
            "prerequisites_met": r.get("prerequisites_met"),
            "error": r.get("error"),
        })

    # 排序：model 世代 → benchmark
    order = {"qwen2.5-1.5b": 0, "qwen2.5-7b": 1, "qwen2.5-72b": 2}
    rows.sort(key=lambda x: (order.get(x["model"], 9), x["benchmark"] or ""))

    out = RESULTS / "scenario_b.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[✓] merged {len(rows)} cells → {out}")


if __name__ == "__main__":
    main()
