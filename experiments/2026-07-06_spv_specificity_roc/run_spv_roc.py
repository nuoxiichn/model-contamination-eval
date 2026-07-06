"""SPV-MIA per-sample AUC / ROC 评测：特异性 + 跨benchmark + 阈值标定。

协议：member=注入题(label 1)，non-member=同 bench held-out(label 0)，score=−Δpv。
member 与 non-member 同分布 → Finding 8 的跨任务整体退化两边同时出现、抵消。
spv_mia() 的 AUC 模式（control_questions 给了就走）已返回 per-sample target/control Δpv，
ROC 在此后处理，不改方法代码。

矩阵 / 阈值 / 集合都在 run.yaml。产出（outputs/2026-07-06_spv_specificity_roc/{ts}/）：
  result.json  每行 spv_mia DetectionResult + roc block
  roc.csv      每行 AUC / TPR@FPR / operating threshold / verdict
  verdict.json benchmark 级裁决 + 聚合 FPR/FNR（对照 role/expect 真值列）

跑法：
  HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_CACHE=/mnt/public/code/chennuoxi/hf_cache \\
  PYTHONPATH=src python3 experiments/2026-07-06_spv_specificity_roc/run_spv_roc.py
"""

from __future__ import annotations

import csv
import gc
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from model_contamination.benchmarks import load_registry
from model_contamination.models.hf_local import HFLocalModel
from model_contamination.stage_sft.spv_mia import spv_mia
from model_contamination.types import BenchmarkQuestion

HERE = Path(__file__).parent
CFG = yaml.safe_load((HERE / "run.yaml").read_text())
OUT_DIR = Path(__file__).resolve().parents[2] / CFG["outputs"]["path"] / time.strftime("%Y%m%d-%H%M%S")


def _json_default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "value"):
        return o.value
    return str(o)


def _load_set(rel_path: str) -> list[BenchmarkQuestion]:
    """从 jsonl 缓存读一组 BenchmarkQuestion。相对 run.yaml 所在目录解析。"""
    p = (HERE / rel_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"题集缓存缺失: {p}")
    out: list[BenchmarkQuestion] = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(BenchmarkQuestion(**json.loads(line)))
    return out


def _slice(qs: list[BenchmarkQuestion], sl) -> list[BenchmarkQuestion]:
    return qs[sl[0]:sl[1]] if sl else qs


def _roc(member_dpv: list[float], nonmember_dpv: list[float], target_fprs: list[float]) -> dict:
    """从 member/non-member 的 Δpv 数组算 ROC。score = −Δpv（越大越像 member）。

    AUC = P(member_score > nonmember_score)（Mann–Whitney，0.5 处理 tie），与
    spv_mia._auc_target_vs_control 同口径。TPR@FPR：在 FPR<=目标的阈值里取最大 TPR，
    对应的阈值即该 FPR 下的 operating threshold（可部署判定点）。
    """
    m = -np.asarray(member_dpv, dtype=float)
    n = -np.asarray(nonmember_dpv, dtype=float)
    m = m[np.isfinite(m)]
    n = n[np.isfinite(n)]
    n_m, n_n = len(m), len(n)
    if n_m < 2 or n_n < 2:
        return {"auc": None, "n_member": int(n_m), "n_nonmember": int(n_n),
                "degraded": "too_few_finite", "operating_points": {}}

    # Mann–Whitney AUC（tie 用平均秩）
    combined = np.concatenate([m, n])
    _, inv, counts = np.unique(combined, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg_rank = (start + csum + 1) / 2.0
    ranks = avg_rank[inv]
    auc = (ranks[:n_m].sum() - n_m * (n_m + 1) / 2.0) / (n_m * n_n)

    # ROC 曲线：阈值扫所有唯一分数（降序）
    thresholds = np.unique(combined)[::-1]
    tpr_list, fpr_list = [], []
    for thr in thresholds:
        tpr_list.append(float((m >= thr).sum()) / n_m)
        fpr_list.append(float((n >= thr).sum()) / n_n)
    tpr_arr = np.asarray(tpr_list)
    fpr_arr = np.asarray(fpr_list)

    ops = {}
    for tf in target_fprs:
        mask = fpr_arr <= tf
        if mask.any():
            i = int(np.argmax(np.where(mask, tpr_arr, -1)))
            ops[f"fpr<={tf}"] = {
                "tpr": float(tpr_arr[i]),
                "fpr": float(fpr_arr[i]),
                "threshold_neg_dpv": float(thresholds[i]),
            }
        else:
            ops[f"fpr<={tf}"] = {"tpr": 0.0, "fpr": 0.0, "threshold_neg_dpv": float(thresholds[0])}

    return {
        "auc": float(auc),
        "n_member": int(n_m),
        "n_nonmember": int(n_n),
        "operating_points": ops,
        "roc_curve": {"fpr": fpr_arr.tolist(), "tpr": tpr_arr.tolist(),
                      "threshold_neg_dpv": thresholds.tolist()},
    }


def _load_model(name: str, path_str: str, stage_tag: str, tokenizer_path: str | None = None) -> HFLocalModel | None:
    path = Path(path_str)
    if not path.exists():
        if CFG.get("skip_missing_checkpoints", True):
            print(f"[skip] {name}: path 不存在 {path}")
            return None
        raise FileNotFoundError(path)
    print(f"\n[i] loading {name} stage={stage_tag} path={path}")
    return HFLocalModel(
        model_path=str(path), stage_tag=stage_tag,
        device=CFG["params"]["device"], dtype=CFG["params"]["dtype"], name=name,
        tokenizer_path=tokenizer_path,
    )


def _free(model) -> None:
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _verdict(auc: float | None, vc: dict) -> str:
    if auc is None:
        return "inconclusive"
    if auc >= vc["auc_dirty"]:
        return "dirty"
    if auc >= vc["auc_suspect"]:
        return "suspect"
    return "clean"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    registry = load_registry()
    sp = CFG["params"]["spv_mia"]
    target_fprs = CFG["params"]["roc"]["target_fprs"]
    vc = CFG["params"]["verdict"]

    # 预载所有题集
    sets = {k: _load_set(v) for k, v in CFG["sets"].items()}
    for k, v in sets.items():
        print(f"[set] {k}: n={len(v)}")

    # 预载 reference（base），整个 run 常驻显存
    ref_cfg = CFG["reference"]
    reference_model = _load_model(f"{ref_cfg['name']}_ref", ref_cfg["path"], ref_cfg["stage_tag"])
    if reference_model is None:
        raise RuntimeError(f"reference base 缺失: {ref_cfg['path']}")

    # 按 target ckpt 分组（每个 ckpt 只 load 一次）
    matrix = CFG["matrix"]
    row_filter = os.environ.get("SPV_ROC_ROWS")  # 例 "3,4" 只跑指定 id（冒烟/续跑）
    if row_filter:
        wanted = {int(x) for x in row_filter.split(",") if x.strip()}
        matrix = [r for r in matrix if r["id"] in wanted]
        print(f"[filter] SPV_ROC_ROWS={row_filter} → 只跑 {len(matrix)} 行")
    by_target: dict[str, list[dict]] = {}
    for row in matrix:
        by_target.setdefault(row["target"], []).append(row)

    results: list[dict] = []
    for target_name, rows in by_target.items():
        path_str = CFG["checkpoints"].get(target_name)
        # merged ckpt 复用 base tokenizer（LlamaFactory export 的 extra_special_tokens 坑）
        model = _load_model(target_name, path_str, "sft", tokenizer_path=ref_cfg["path"]) if path_str else None
        if model is None:
            for row in rows:
                results.append({**row, "skipped": True, "reason": "checkpoint missing"})
            continue

        for row in rows:
            spec = registry.get(row["spec"])
            member = _slice(sets[row["member"]], row.get("member_slice"))
            nonmember = _slice(sets[row["nonmember"]], row.get("nonmember_slice"))
            print(f"\n  [row {row['id']}] {target_name} × {row['spec']} "
                  f"({row['role']}): member={len(member)} nonmember={len(nonmember)}")
            t0 = time.time()
            r = spv_mia(
                model, reference_model, spec, member,
                control_questions=nonmember,
                n_neighbors=sp["n_neighbors"], mask_ratio=sp["mask_ratio"],
                min_samples=sp["min_samples"], paraphraser=sp["paraphraser"], seed=sp["seed"],
            )
            dt = time.time() - t0
            ev = r.evidence or {}
            roc = _roc(ev.get("target_delta_pv", []), ev.get("control_delta_pv", []), target_fprs)
            verdict = _verdict(roc["auc"], vc)
            print(f"    spv signal(auc)={r.signal!r} roc_auc={roc['auc']!r} "
                  f"verdict={verdict} expect={row['expect']} elapsed={dt:.1f}s"
                  + (f" degraded={ev.get('degraded_reason')}" if ev.get("degraded_reason") else ""))
            results.append({
                **row, "skipped": False,
                "spv_result": asdict(r), "roc": roc,
                "verdict": verdict, "elapsed_seconds": dt,
            })
            # 增量落盘
            (OUT_DIR / "result.json").write_text(
                json.dumps({"config": CFG, "results": results}, indent=2,
                           ensure_ascii=False, default=_json_default))

        _free(model)

    _free(reference_model)
    _write_outputs(results, vc)
    print(f"\n[i] wrote {OUT_DIR}/result.json, roc.csv, verdict.json")


def _write_outputs(results: list[dict], vc: dict) -> None:
    # roc.csv
    rows = []
    for r in results:
        if r.get("skipped"):
            rows.append({"id": r["id"], "target": r["target"], "spec": r["spec"],
                         "role": r["role"], "skipped": True})
            continue
        roc = r["roc"]
        ops = roc.get("operating_points", {})
        row = {
            "id": r["id"], "target": r["target"], "spec": r["spec"], "role": r["role"],
            "expect": r["expect"], "auc": roc.get("auc"), "verdict": r["verdict"],
            "n_member": roc.get("n_member"), "n_nonmember": roc.get("n_nonmember"),
        }
        for k, op in ops.items():
            row[f"tpr@{k}"] = op["tpr"]
            row[f"thr@{k}"] = op["threshold_neg_dpv"]
        rows.append(row)
    if rows:
        keys = sorted({k for row in rows for k in row})
        with (OUT_DIR / "roc.csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    # benchmark 级聚合 FPR/FNR：expect 为 chance 的格是真负(应 clean)，其余是真正(应 dirty/suspect)
    tp = fp = tn = fn = 0
    for r in results:
        if r.get("skipped") or r["roc"].get("auc") is None:
            continue
        is_dirty_pred = r["verdict"] in ("dirty", "suspect")
        is_positive_truth = r["expect"] != "chance"
        if is_positive_truth and is_dirty_pred:
            tp += 1
        elif is_positive_truth and not is_dirty_pred:
            fn += 1
        elif not is_positive_truth and is_dirty_pred:
            fp += 1
        else:
            tn += 1
    n_pos, n_neg = tp + fn, fp + tn
    verdict_summary = {
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "fpr": (fp / n_neg) if n_neg else None,   # 误报率：真负里被判脏的比例
        "fnr": (fn / n_pos) if n_pos else None,   # 漏报率：真正里被判净的比例
        "note": ("benchmark 级裁决用 AUC 阈值 dirty>=%.2f/suspect>=%.2f；"
                 "expect==chance 视为真负(negctrl/cross-bench/FP)，其余真正(dose)。"
                 "positive control=污染 ckpt，本表即硬前置#5 的量化闭环。"
                 % (vc["auc_dirty"], vc["auc_suspect"])),
    }
    (OUT_DIR / "verdict.json").write_text(
        json.dumps(verdict_summary, indent=2, ensure_ascii=False))
    print(f"[verdict] confusion={verdict_summary['confusion']} "
          f"FPR={verdict_summary['fpr']} FNR={verdict_summary['fnr']}")


if __name__ == "__main__":
    main()
