#!/usr/bin/env python3
"""CoDeC 阈值校准（纯后处理，无需 GPU）。

消费：
    ../2026-07-06_codec_dose_response/outputs/dose_curves.json   （正样本 + p=0 负样本）
    ../2026-07-06_codec_specificity/outputs/specificity_results.json（阴性面板 + 溢出邻居）

方法：CoDeC 的裁决单位是「数据集」，signal = frac_negative（Δ<0 比例）。
    每个数据集实例用其 per-sample Δ 做 bootstrap，重采样出 n_boot 个合成 signal，
    丰富 ROC。正/负池各自汇总 → ROC/AUC、固定 FPR=5% 阈值、Youden 阈值。
    再算剂量分层 TPR：给定阈值，各剂量的检出率 → 最小可靠检出剂量。

正样本：dose_response 里 dose>0 的每个 (target,dose,seed) 终点 checkpoint。
负样本：specificity 阴性面板 + dose_response 的 p=0 终点 + specificity 溢出未训练邻居。
异质拼盘(2c) 作为「最坏干净点」单列，检查阈值是否落在其之上。

跑法：
    PYTHONPATH=src python3 experiments/2026-07-06_codec_calibration/run_calib.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent


def _load_json(p: Path):
    if not p.exists():
        raise SystemExit(
            f"[missing] {p}\n先跑完 dose_response / specificity 生成 outputs 再校准。"
        )
    return json.loads(p.read_text(encoding="utf-8"))


def _boot_signals(deltas, n_boot: int, rng) -> list[float]:
    """从 per-sample Δ bootstrap 出 n_boot 个数据集级 signal(=frac_negative)。"""
    if not deltas:
        return []
    neg = np.asarray(deltas) < 0.0
    n = len(neg)
    return [float(neg[rng.integers(0, n, n)].mean()) for _ in range(n_boot)]


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC = P(signal_pos > signal_neg)，Mann-Whitney U / (n_pos*n_neg)，含并列 0.5。"""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    # 处理并列：同值取平均秩
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    avg_rank = np.zeros(len(counts))
    cum = 0
    for i, c in enumerate(counts):
        avg_rank[i] = cum + (c + 1) / 2.0
        cum += c
    ranks = avg_rank[inv]
    r_pos = ranks[: len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def _roc(pos: np.ndarray, neg: np.ndarray, n_pts: int = 200) -> list[dict]:
    lo = float(min(pos.min(), neg.min()))
    hi = float(max(pos.max(), neg.max()))
    out = []
    for thr in np.linspace(lo, hi, n_pts):
        tpr = float((pos >= thr).mean())
        fpr = float((neg >= thr).mean())
        out.append({"thr": float(thr), "tpr": tpr, "fpr": fpr})
    return out


def main() -> None:
    cfg = yaml.safe_load((_HERE / "run.yaml").read_text(encoding="utf-8"))
    boot = cfg["bootstrap"]
    n_boot = int(boot["n_boot"])
    rng = np.random.default_rng(int(boot["seed"]))
    target_fpr = float(cfg["target_fpr"])

    dose_data = _load_json(_HERE / cfg["inputs"]["dose_curves"])
    spec_data = _load_json(_HERE / cfg["inputs"]["specificity"])["results"]

    pos_signals: list[float] = []
    neg_signals: list[float] = []
    # 剂量分层：dose -> 该剂量所有 run 终点的实测 signal（不 bootstrap，看原始检出）
    by_dose: dict[str, list[float]] = {}

    # ---- 正/负样本 from dose_response ---- #
    for _key, run in dose_data.items():
        curve = run.get("curve") or []
        if not curve:
            continue
        final = curve[-1]
        deltas = final.get("deltas")
        dose = run["dose"]
        boots = _boot_signals(deltas, n_boot, rng)
        if dose == 0.0:
            neg_signals.extend(boots)          # p=0 控制 = 负样本
        else:
            pos_signals.extend(boots)          # 注入污染 = 正样本
            by_dose.setdefault(str(dose), []).append(final.get("signal"))

    # ---- 负样本 from specificity 阴性面板 ---- #
    for _name, rec in spec_data.get("negative_panel", {}).items():
        neg_signals.extend(_boot_signals(rec.get("deltas"), n_boot, rng))

    # ---- 负样本 from specificity 溢出未训练邻居 ---- #
    sp = spec_data.get("spillover", {})
    trained = sp.get("train_target")
    for name, rec in sp.get("eval", {}).items():
        if name == trained:
            continue  # 训练目标是正样本方向，不进负池
        neg_signals.extend(_boot_signals(rec.get("deltas"), n_boot, rng))

    pos = np.asarray(pos_signals)
    neg = np.asarray(neg_signals)
    if pos.size == 0 or neg.size == 0:
        raise SystemExit(f"pos={pos.size} neg={neg.size}; 检查 outputs 是否含 deltas。")

    auc = _auc(pos, neg)
    roc = _roc(pos, neg)

    # 固定 FPR=target_fpr 反解阈值 = 负样本的 (1-fpr) 分位
    thr_fpr = float(np.percentile(neg, 100 * (1 - target_fpr)))
    tpr_at = float((pos >= thr_fpr).mean())
    # Youden J 阈值
    j = max(roc, key=lambda r: r["tpr"] - r["fpr"])

    # 剂量分层 TPR（在 thr_fpr 下）
    dose_tpr = {}
    for d, sigs in sorted(by_dose.items(), key=lambda kv: float(kv[0])):
        vals = [s for s in sigs if s is not None]
        dose_tpr[d] = {
            "mean_signal": float(np.mean(vals)) if vals else None,
            "tpr_at_thr": float(np.mean([s >= thr_fpr for s in vals])) if vals else None,
            "n_runs": len(vals),
        }

    # 异质拼盘 = 最坏干净点
    het = spec_data.get("heterogeneous", {})
    het_signal = het.get("signal")

    summary = {
        "auc": auc,
        "n_pos_boot": int(pos.size),
        "n_neg_boot": int(neg.size),
        "neg_signal_p95": float(np.percentile(neg, 95)),
        "pos_signal_mean": float(pos.mean()),
        "threshold_fixed_fpr": {
            "target_fpr": target_fpr, "threshold": thr_fpr, "tpr": tpr_at,
        },
        "threshold_youden": {
            "threshold": j["thr"], "tpr": j["tpr"], "fpr": j["fpr"],
        },
        "dose_stratified_tpr": dose_tpr,
        "heterogeneous_worst_clean": {
            "signal": het_signal,
            "below_fpr_threshold": (
                bool(het_signal is not None and het_signal < thr_fpr)
            ),
        },
        "caveats": [
            "单模型(Pythia-2.8b) + 合成注入正样本；阈值 model-specific，不跨模型族迁移。",
            "Pythia-calibrated operating characteristic，非可对外发布的绝对红/黄/绿裁决。",
            "真实预训练级污染的分布可能与受控 finetune 注入不同。",
        ],
    }

    out_dir = _HERE / "outputs"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "calibration.json").write_text(
        json.dumps({"summary": summary, "roc": roc}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("=== CALIBRATION SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[done] wrote {out_dir / 'calibration.json'}")


if __name__ == "__main__":
    main()
