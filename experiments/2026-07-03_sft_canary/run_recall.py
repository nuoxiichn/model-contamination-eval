"""Canary recall runner：对 canary_v1_merged / clean_merged / base 跑召回。

产出：
  outputs/2026-07-03_sft_canary/<ts>/result.json  —— 每 ckpt 的 DetectionResult + per-canary
  outputs/2026-07-03_sft_canary/<ts>/recall_curve.csv —— ckpt × bucket × recall（画曲线用）

成功判据：
  clean / base recall_contains = 0/100（FPR=0 实证）
  canary ckpt 出现随 rep 单调爬升的召回曲线 + 可辨识饱和点

执行（8 卡机，GPU；训练 + merge 完成后）：
    HF_ENDPOINT=https://hf-mirror.com \\
    PYTHONPATH=src python3 experiments/2026-07-03_sft_canary/run_recall.py \\
        --ckpt canary_v1,clean,base --device cuda
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from model_contamination.models.hf_local import HFLocalModel
from model_contamination.shared.canary import measure_canary_recall

HERE = Path(__file__).parent
REPO_ROOT = Path(__file__).resolve().parents[2]

CANARY_PATH = REPO_ROOT / "data" / "canary" / "canary_v1.jsonl"
MANIFEST_PATH = HERE / "bucket_manifest.jsonl"
OUT_ROOT = REPO_ROOT / "outputs" / "2026-07-03_sft_canary"

# ckpt id → (path, stage_tag)
CKPTS: dict[str, dict] = {
    "base": {"path": "/mnt/public/model/huggingface/Qwen3-1.7B-Base", "stage_tag": "base"},
    "clean": {"path": "/mnt/public/code/chennuoxi/LlamaFactory/saves/contam/clean_merged",
              "stage_tag": "sft"},
    "canary_v1": {"path": "/mnt/public/code/chennuoxi/LlamaFactory/saves/contam/canary_v1_merged",
                  "stage_tag": "sft"},
}


def _json_default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "value"):
        return o.value
    return str(o)


def _free_model(model) -> None:
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _bucket_map() -> dict[str, dict]:
    """canary_id → {bucket, rep}。"""
    return {
        json.loads(ln)["canary_id"]: json.loads(ln)
        for ln in MANIFEST_PATH.read_text().splitlines() if ln.strip()
    }


def _per_bucket_recall(per_canary: list[dict], bmap: dict[str, dict]) -> dict:
    """把 per-canary 命中按 bucket 聚合 → bucket → {rep, n, contains, exact, recall_*}。"""
    agg: dict[str, dict] = defaultdict(
        lambda: {"rep": None, "n": 0, "n_contains": 0, "n_exact": 0})
    for pc in per_canary:
        b = bmap.get(pc["canary_id"])
        if not b:
            continue
        cell = agg[b["bucket"]]
        cell["rep"] = b["rep"]
        cell["n"] += 1
        cell["n_contains"] += int(pc["contains"])
        cell["n_exact"] += int(pc["exact"])
    for cell in agg.values():
        cell["recall_contains"] = cell["n_contains"] / cell["n"] if cell["n"] else 0.0
        cell["recall_exact"] = cell["n_exact"] / cell["n"] if cell["n"] else 0.0
    return dict(agg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Canary recall runner")
    parser.add_argument("--ckpt", default="canary_v1,clean,base",
                        help=f"逗号分隔 ckpt id；可选 {list(CKPTS)}")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    ckpt_ids = [c.strip() for c in args.ckpt.split(",") if c.strip()]
    out_dir = Path(args.out_dir) if args.out_dir else OUT_ROOT / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    bmap = _bucket_map()

    all_results: dict[str, dict] = {}
    for cid in ckpt_ids:
        if cid not in CKPTS:
            print(f"[skip] 未知 ckpt id: {cid}")
            continue
        cfg = CKPTS[cid]
        path = Path(cfg["path"])
        if not path.exists():
            print(f"[skip] {cid}: path 不存在 {path}（训练/merge 未完成？）")
            continue
        print(f"\n[i] loading {cid} stage={cfg['stage_tag']} path={path}")
        model = HFLocalModel(
            model_path=str(path), stage_tag=cfg["stage_tag"],
            device=args.device, dtype=args.dtype, name=cid, trust_remote_code=True,
        )
        t0 = time.time()
        r = measure_canary_recall(model, CANARY_PATH, canary_type="sft")
        dt = time.time() - t0
        ev = r.evidence or {}
        per_bucket = _per_bucket_recall(ev.get("per_canary", []), bmap)
        print(f"  [{cid}] recall_contains={r.signal} recall_exact={ev.get('recall_exact')} "
              f"verdict={r.verdict_hint!r} elapsed={dt:.1f}s")
        for b in sorted(per_bucket):
            c = per_bucket[b]
            print(f"    {b} rep={c['rep']}: contains={c['recall_contains']:.2f} exact={c['recall_exact']:.2f}")
        all_results[cid] = {
            "stage_tag": cfg["stage_tag"],
            "result": asdict(r),
            "per_bucket": per_bucket,
            "elapsed_seconds": dt,
        }
        _free_model(model)
        (out_dir / "result.json").write_text(
            json.dumps({"ckpts": ckpt_ids, "results": all_results}, indent=2,
                       ensure_ascii=False, default=_json_default))

    # recall 曲线 CSV
    rows = []
    for cid, blob in all_results.items():
        for bucket, c in sorted(blob["per_bucket"].items()):
            rows.append({
                "ckpt": cid, "stage": blob["stage_tag"], "bucket": bucket,
                "rep": c["rep"], "effective_exposure": (c["rep"] or 0) * 3,
                "n": c["n"], "recall_contains": c["recall_contains"],
                "recall_exact": c["recall_exact"],
            })
    if rows:
        with (out_dir / "recall_curve.csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\n[done] wrote {out_dir / 'result.json'}")
    print(f"[done] wrote {out_dir / 'recall_curve.csv'}")


if __name__ == "__main__":
    main()
