"""perm_option FP —— 从 per-cell JSON 汇总统一 CSV（数据并行分片后合并）。

各 shard 写自己的 <model>__<bench>.json；本脚本扫描 outputs/results/*.json
汇总成 fp.csv，避免多进程并发写同一 CSV 的 race。

运行（8 卡机跑完后）：
    PYTHONPATH=src python3 experiments/2026-07-10_perm_option_fp/collect_fp.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "outputs" / "results"


def main() -> None:
    rows: list[dict] = []
    for jf in sorted(RESULTS.glob("*.json")):
        r = json.loads(jf.read_text(encoding="utf-8"))
        ev = r.get("evidence") or {}
        rows.append({
            "model": jf.stem.split("__")[0],
            "benchmark": r.get("benchmark"),
            "fpr_leak_fraction": r.get("signal"),
            "verdict": r.get("verdict_hint"),
            "n_questions": ev.get("n_questions"),
            "mean_perms_per_q": ev.get("mean_perms_per_q"),
            "primary_threshold": ev.get("primary_threshold"),
            "leak_fraction_by_threshold": json.dumps(
                ev.get("leak_fraction_by_threshold"), ensure_ascii=False
            ),
            "prerequisites_met": r.get("prerequisites_met"),
            "error": r.get("error"),
        })
    rows.sort(key=lambda d: (str(d["model"]), str(d["benchmark"])))
    out = RESULTS / "fp.csv"
    if rows:
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"[✓] wrote {out}  ({len(rows)} cells)")
    # 打印一张速览表
    print("\n=== FP (leak_fraction = false-positive rate) ===")
    print(f"{'model':<16}{'benchmark':<24}{'FPR':>8}")
    for r in rows:
        v = r["fpr_leak_fraction"]
        print(f"{r['model']:<16}{r['benchmark']:<24}{('n/a' if v is None else f'{float(v):.4f}'):>8}")


if __name__ == "__main__":
    main()
