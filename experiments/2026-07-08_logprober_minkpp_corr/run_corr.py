#!/usr/bin/env python3
"""LogProber vs Min-K%++ 信号相关性：WikiMIA + Pythia-6.9B。

判定 LogProber（arXiv:2408.14352）相对 Min-K%++ 是冗余还是互补。两信号同源
（单次前向的 per-token log-prob），在一次前向里同时算出。

每样本前向一次 → chosen(raw) / μ / σ → 派生三个信号：
1. minkpp    = bottom-K%( (chosen-μ)/σ )            现有方法（归一化 + bottom-k）
2. mink_loss = bottom-K%( chosen )                  桥接项（raw + bottom-k）
3. logprober = 累积 surprisal 曲线 S_x=A(1-e^{-Bx}) 拟合出的 B（acceleration）
              论文原版（raw + 曲线水平度）；B 越大=曲线越早触底=越像被记忆

输出 outputs/corr.json：
- 三对信号的 Spearman/Pearson（pooled / member-only / nonmember-only）
- 三个信号各自的 member-vs-nonmember AUROC（复用 _auc_target_vs_control）
- 每样本原始分数（画 scatter 用；不进 git）

结果写本目录 outputs/（不进 git）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from model_contamination.models.base import Capability  # noqa: E402
from model_contamination.models.hf_local import HFLocalModel  # noqa: E402
from model_contamination.stage_base.min_k_plus_plus import _auc_target_vs_control  # noqa: E402

_HERE = Path(__file__).resolve().parent


def _per_token_arrays(model, prefix: str, text: str) -> tuple[np.ndarray, np.ndarray]:
    """单样本前向一次 → (chosen[finite], normalized[finite])。

    chosen 保序（token 顺序），LogProber 累积曲线需要顺序；MinK++ 只取集合，顺序无关。
    """
    stats = model.token_logprob_stats(prefix, text)
    chosen = stats["chosen_logp"]
    mu = stats["mu"]
    sigma = stats["sigma"]
    if chosen.size == 0:
        return np.zeros(0), np.zeros(0)
    safe_sigma = np.where(sigma > 1e-8, sigma, np.nan)
    normalized = (chosen - mu) / safe_sigma
    return chosen[np.isfinite(chosen)], normalized[np.isfinite(normalized)]


def _bottom_k_mean(arr: np.ndarray, k_ratio: float) -> float:
    """bottom-K% 的均值（Min-K% 系 statistic）。空数组 → nan。"""
    if arr.size == 0:
        return float("nan")
    k = max(1, int(np.ceil(k_ratio * arr.size)))
    return float(np.mean(np.partition(arr, k - 1)[:k]))


def _saturating(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * (1.0 - np.exp(-b * x))


def _logprober_fit(chosen: np.ndarray, min_tokens: int) -> tuple[float, float]:
    """LogProber 曲线水平度：拟合累积 surprisal S_x = A(1-e^{-Bx})。

    surprisal_i = -log p_i ≥ 0，S_x = Σ_{i<=x} surprisal_i 单调递增。
    - A = asymptote（累积 surprisal 上界，越小=整段越可预测）
    - B = acceleration（越大=越快触底=水平度越高=越像被记忆）
    返回 (A, B)；token 数不足或拟合不收敛 → (nan, nan)（上层过滤）。
    """
    if chosen.size < min_tokens:
        return float("nan"), float("nan")
    surprisal = -chosen
    s = np.cumsum(surprisal)
    n = s.size
    x = np.arange(1, n + 1, dtype=np.float64)
    a0 = float(s[-1]) if s[-1] > 0 else 1.0
    try:
        popt, _ = curve_fit(
            _saturating, x, s, p0=[a0, 0.1],
            bounds=([0.0, 0.0], [np.inf, np.inf]), maxfev=5000,
        )
        return float(popt[0]), float(popt[1])
    except (RuntimeError, ValueError):
        return float("nan"), float("nan")


def _finite_pair(a: list[float], b: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """成对取 finite（相关性要求两列同时有效）。"""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _corr(a: list[float], b: list[float]) -> dict:
    x, y = _finite_pair(a, b)
    if x.size < 3:
        return {"spearman": None, "pearson": None, "n": int(x.size)}
    rho, _ = spearmanr(x, y)
    r, _ = pearsonr(x, y)
    return {"spearman": float(rho), "pearson": float(r), "n": int(x.size)}


def _auroc(member: list[float], nonmember: list[float]) -> float:
    """member 应得分更高 → AUROC = P(member > nonmember)。方向未对齐时 <0.5。"""
    m = np.asarray([v for v in member if np.isfinite(v)])
    n = np.asarray([v for v in nonmember if np.isfinite(v)])
    if m.size == 0 or n.size == 0:
        return float("nan")
    return _auc_target_vs_control(m, n)


def main() -> None:
    cfg = yaml.safe_load((_HERE / "run.yaml").read_text(encoding="utf-8"))
    model_cfg = cfg["model"]
    wm_cfg = cfg["wikimia"]
    report_k = float(cfg["report_k"])
    min_tokens_fit = int(cfg["min_tokens_fit"])
    prefix = cfg["prefix"]
    limit = wm_cfg["limit"]

    from datasets import load_dataset

    print(f"[model] loading {model_cfg['path']}")
    model = HFLocalModel(
        model_path=model_cfg["path"], stage_tag=model_cfg["stage_tag"],
        dtype=model_cfg.get("dtype"), name=model_cfg["name"],
    )
    assert model.supports(Capability.TOKEN_DIST_STATS), "model must expose TOKEN_DIST_STATS"

    out_dir = _HERE / "outputs"
    out_dir.mkdir(exist_ok=True)
    results: dict = {"config": cfg, "splits": {}}

    for split in wm_cfg["splits"]:
        ds = load_dataset(wm_cfg["dataset"], split=split)
        if limit:
            ds = ds.select(range(min(int(limit), len(ds))))
        print(f"\n=== {split}  (n={len(ds)}) ===")

        rows: list[dict] = []  # 每样本一条：label + 三信号
        for i, row in enumerate(ds):
            chosen, normalized = _per_token_arrays(model, prefix, row["input"])
            _a, b = _logprober_fit(chosen, min_tokens_fit)
            rows.append({
                "label": int(row["label"]),
                "minkpp": _bottom_k_mean(normalized, report_k),
                "mink_loss": _bottom_k_mean(chosen, report_k),
                "logprober_b": b,
                "logprober_a": _a,
            })
            if (i + 1) % 100 == 0:
                print(f"  scored {i + 1}/{len(ds)}")

        split_rec = _analyze(rows)
        split_rec["rows"] = rows
        results["splits"][split] = split_rec
        _print_split(split, split_rec)

    (out_dir / "corr.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[done] wrote {out_dir / 'corr.json'}")


def _analyze(rows: list[dict]) -> dict:
    """算三对相关性（pooled/member/nonmember）+ 三信号 AUROC。"""
    def col(rs: list[dict], key: str) -> list[float]:
        return [r[key] for r in rs]

    member = [r for r in rows if r["label"] == 1]
    nonmember = [r for r in rows if r["label"] == 0]

    pairs = [
        ("logprober_b__minkpp", "logprober_b", "minkpp"),
        ("logprober_b__mink_loss", "logprober_b", "mink_loss"),
        ("minkpp__mink_loss", "minkpp", "mink_loss"),
    ]
    corr = {}
    for name, ka, kb in pairs:
        corr[name] = {
            "pooled": _corr(col(rows, ka), col(rows, kb)),
            "member": _corr(col(member, ka), col(member, kb)),
            "nonmember": _corr(col(nonmember, ka), col(nonmember, kb)),
        }

    auroc = {
        sig: _auroc(col(member, sig), col(nonmember, sig))
        for sig in ("minkpp", "mink_loss", "logprober_b", "logprober_a")
    }
    return {
        "n": len(rows),
        "n_member": len(member),
        "n_nonmember": len(nonmember),
        "correlation": corr,
        "auroc": auroc,
    }


def _print_split(split: str, rec: dict) -> None:
    c = rec["correlation"]["logprober_b__minkpp"]["pooled"]
    cb = rec["correlation"]["logprober_b__mink_loss"]["pooled"]
    a = rec["auroc"]
    print(
        f"  ρ(logprober_B, minkpp)={_fmt(c['spearman'])}  "
        f"ρ(logprober_B, mink_loss)={_fmt(cb['spearman'])}"
    )
    print(
        f"  AUROC  minkpp={_fmt(a['minkpp'])}  mink_loss={_fmt(a['mink_loss'])}  "
        f"logprober_B={_fmt(a['logprober_b'])}  logprober_A={_fmt(a['logprober_a'])}"
    )


def _fmt(x: float | None) -> str:
    return "  nan" if x is None or not np.isfinite(x) else f"{x:+.3f}"


if __name__ == "__main__":
    main()
