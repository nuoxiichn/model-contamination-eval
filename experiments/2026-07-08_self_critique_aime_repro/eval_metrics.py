#!/usr/bin/env python3
"""Self-Critique AIME 复现评估：从 shard jsonl 出 F1 + AUC，对比论文。

复刻论文 evaluate_all_methods.py::evaluate_performance（sklearn）：
- roc_auc = auc(roc_curve)
- best_f1_score = max over precision_recall_curve（用户主看这个 F1）
- Youden's J 阈值下的 f1 / accuracy
- tpr_at_fpr_5

self_critique direction=1（score 高 = member），故 y_scores = score * 1。
overall（全 60 题）+ by data_source（aime / aime25 各 30）+ mean_auc（各 source 平均）。

跑法：
    PYTHONPATH=src python3 experiments/2026-07-08_self_critique_aime_repro/eval_metrics.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    precision_recall_curve,
    roc_curve,
)

_HERE = Path(__file__).resolve().parent
_DIRECTION = 1  # SelfCritiqueDetector.get_direction()


def evaluate_performance(y_true: np.ndarray, y_scores: np.ndarray) -> dict:
    """论文同款指标集（roc_auc / best_f1 / youden / tpr@fpr5）。"""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=np.float64)
    finite = np.isfinite(y_scores)
    y_true, y_scores = y_true[finite], y_scores[finite]

    if len(np.unique(y_true)) < 2:
        return {"roc_auc": None, "best_f1_score": None, "n": int(len(y_true)),
                "error": "Only one class present."}

    fpr, tpr, roc_thr = roc_curve(y_true, y_scores)
    roc_auc = float(auc(fpr, tpr))

    precision, recall, pr_thr = precision_recall_curve(y_true, y_scores)
    fscore = (2 * precision * recall) / (precision + recall + 1e-6)
    best_idx = int(np.argmax(fscore[:-1])) if len(fscore) > 1 else 0
    best_f1 = float(fscore[best_idx])
    thr_f1 = float(pr_thr[best_idx])
    acc_at_best_f1 = float(accuracy_score(y_true, (y_scores >= thr_f1).astype(int)))

    youden = tpr - fpr
    j_idx = int(np.argmax(youden))
    thr_j = float(roc_thr[j_idx])
    pred_j = (y_scores >= thr_j).astype(int)
    tp = int(np.sum((y_true == 1) & (pred_j == 1)))
    fp = int(np.sum((y_true == 0) & (pred_j == 1)))
    fn = int(np.sum((y_true == 1) & (pred_j == 0)))
    prec_j = tp / (tp + fp + 1e-6)
    rec_j = tp / (tp + fn + 1e-6)
    f1_at_j = float((2 * prec_j * rec_j) / (prec_j + rec_j + 1e-6))
    acc_at_j = float(accuracy_score(y_true, pred_j))

    idx_above = np.where(fpr >= 0.05)[0]
    if len(idx_above) > 0:
        ti = idx_above[0] - 1 if idx_above[0] > 0 else 0
        tpr_at_fpr5 = float(tpr[ti])
    else:
        tpr_at_fpr5 = float(tpr[-1]) if len(tpr) else None

    return {
        "n": int(len(y_true)),
        "roc_auc": roc_auc,
        "best_f1_score": best_f1,
        "accuracy_at_best_f1": acc_at_best_f1,
        "optimal_threshold_f1": thr_f1,
        "youden_j_score": float(youden[j_idx]),
        "f1_at_youden_threshold": f1_at_j,
        "accuracy_at_youden_threshold": acc_at_j,
        "tpr_at_fpr_5": tpr_at_fpr5,
    }


def _load_scores(subdir: str = "") -> list[dict]:
    out_dir = _HERE / "outputs" / subdir if subdir else _HERE / "outputs"
    recs: list[dict] = []
    seen: set[str] = set()
    for p in sorted(out_dir.glob("shard_*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            recs.append(r)
    return recs


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdir", type=str, default="",
                    help="读 outputs/<subdir>/ 的 shard 结果（区分干净 vs 污染模型）")
    args = ap.parse_args()

    recs = _load_scores(args.subdir)
    if not recs:
        print("[eval] no shard outputs found. Run run_self_critique_aime.py first.")
        return

    labels = np.array([r["ground_truth_label"] for r in recs])
    scores = np.array([r["self_critique_score"] for r in recs], dtype=np.float64) * _DIRECTION
    sources = [r["data_source"] for r in recs]

    n_nan = int(np.sum(~np.isfinite(scores)))
    overall = evaluate_performance(labels, scores)

    breakdown = {}
    aucs = []
    for src in sorted(set(sources)):
        mask = np.array([s == src for s in sources])
        perf = evaluate_performance(labels[mask], scores[mask])
        breakdown[src] = perf
        if perf.get("roc_auc") is not None:
            aucs.append(perf["roc_auc"])
    mean_auc = float(np.mean(aucs)) if aucs else None

    summary = {
        "method": "self_critique",
        "n_total": len(recs),
        "n_nan_scores": n_nan,
        "overall_performance": overall,
        "mean_auc_by_source": mean_auc,
        "breakdown_by_source": breakdown,
    }
    out_path = (_HERE / "outputs" / args.subdir if args.subdir else _HERE / "outputs") / "evaluation_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 打印对照表
    print("\n=== Self-Critique AIME24/25 复现结果 ===")
    print(f"样本 {len(recs)}（NaN score {n_nan}）")
    print(f"\n[overall]  AUC={overall.get('roc_auc')}  best_F1={overall.get('best_f1_score')}  "
          f"acc@F1={overall.get('accuracy_at_best_f1')}  TPR@FPR5={overall.get('tpr_at_fpr_5')}")
    print(f"[mean_auc_by_source] {mean_auc}")
    for src, perf in breakdown.items():
        print(f"  [{src:7s}] n={perf.get('n')}  AUC={perf.get('roc_auc')}  "
              f"best_F1={perf.get('best_f1_score')}  f1@youden={perf.get('f1_at_youden_threshold')}")
    print("\n对照论文（Qwen2.5-7B-Instruct, self_critique）：AIME AUC≈论文 Table 数值。")
    print("⚠️ 若用原始 Instruct（未训污染 ckpt），AUC 应≈0.5（null baseline，符合预期）。")
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
