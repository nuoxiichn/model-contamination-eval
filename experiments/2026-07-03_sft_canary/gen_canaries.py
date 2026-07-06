"""生成 canary_v1 + 分桶（注入重复次数网格）。

调 shared.canary.generate_sft_canaries 造 100 条唯一 canary，按 bucket 分配注入重复次数，
产出：
  data/canary/canary_v1.jsonl          —— 100 条 canary（明文，不进 git）
  data/canary/canary_v1.jsonl.meta.json—— seed / sha256 存档
  <此目录>/bucket_manifest.jsonl        —— canary_id → bucket → rep 映射（recall 分桶用）

网格（实验计划见 notes.md）：
  B1 rep=1  B2 rep=3  B3 rep=10  B4 rep=30  B5 rep=100，每桶 20 条 → 100 条

执行（开发机 CPU 即可）：
    PYTHONPATH=src python3 experiments/2026-07-03_sft_canary/gen_canaries.py
"""

from __future__ import annotations

import json
from pathlib import Path

from model_contamination.shared.canary import generate_sft_canaries

HERE = Path(__file__).parent
REPO_ROOT = Path(__file__).resolve().parents[2]

SEED = 42
N_PER_BUCKET = 20
BUCKETS = [
    {"bucket": "B1", "rep": 1},
    {"bucket": "B2", "rep": 3},
    {"bucket": "B3", "rep": 10},
    {"bucket": "B4", "rep": 30},
    {"bucket": "B5", "rep": 100},
]
CANARY_PATH = REPO_ROOT / "data" / "canary" / "canary_v1.jsonl"


def main() -> None:
    n_total = N_PER_BUCKET * len(BUCKETS)
    canaries = generate_sft_canaries(n=n_total, seed=SEED, output_path=CANARY_PATH)
    assert len(canaries) == n_total, f"期望 {n_total}，得到 {len(canaries)}"

    # 按顺序切片分桶：前 20 → B1，次 20 → B2 …
    manifest = []
    for bi, bucket_cfg in enumerate(BUCKETS):
        lo = bi * N_PER_BUCKET
        hi = lo + N_PER_BUCKET
        for c in canaries[lo:hi]:
            manifest.append({
                "canary_id": c["canary_id"],
                "bucket": bucket_cfg["bucket"],
                "rep": bucket_cfg["rep"],
            })

    manifest_path = HERE / "bucket_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[gen] {n_total} canaries → {CANARY_PATH}")
    print(f"[gen] bucket manifest → {manifest_path}")
    for bc in BUCKETS:
        print(f"  {bc['bucket']}: rep={bc['rep']} (有效曝光 ×3 epoch = {bc['rep'] * 3}) n={N_PER_BUCKET}")
    total_lines = sum(bc["rep"] * N_PER_BUCKET for bc in BUCKETS)
    print(f"[gen] 注入总行数 = {total_lines}")


if __name__ == "__main__":
    main()
