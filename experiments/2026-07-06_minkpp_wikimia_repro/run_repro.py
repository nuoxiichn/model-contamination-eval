#!/usr/bin/env python3
"""Min-K%++ 论文复现：WikiMIA + Pythia-6.9B → member/non-member AUROC。

验证 `_mink_pp_sample_score` 实现无误。流程：
1. 加载 WikiMIA（input + label；label=1 member/见过，label=0 non-member/未见过）；
2. 每样本对原始文本（绕开 QA 封装）算 per-token normalized log p 与 chosen log p，前向只跑一次；
3. k 扫描复用缓存数组，算 Min-K%++（normalized）与 Min-K%（loss）两法的样本级分数；
4. AUROC = P(member_score > nonmember_score)，复用 _auc_target_vs_control（Mann–Whitney U）。

预期：Min-K%++ AUROC 对齐论文 Pythia-6.9B ≈ 0.70，且显著高于 Min-K% loss 基线。

结果写本目录 outputs/（不进 git）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from model_contamination.models.base import Capability  # noqa: E402
from model_contamination.models.hf_local import HFLocalModel  # noqa: E402
from model_contamination.stage_base.min_k_plus_plus import _auc_target_vs_control  # noqa: E402

_HERE = Path(__file__).resolve().parent


def _per_token_arrays(model, prefix: str, text: str) -> tuple[np.ndarray, np.ndarray]:
    """单样本前向一次 → (chosen_logp[finite], normalized[finite])。

    chosen 用于 Min-K%（loss），normalized=(chosen-μ)/σ 用于 Min-K%++。
    σ≈0 位置在 normalized 里剔除；chosen 保留全部有限值。
    """
    stats = model.token_logprob_stats(prefix, text)
    chosen = stats["chosen_logp"]
    mu = stats["mu"]
    sigma = stats["sigma"]
    if chosen.size == 0:
        return np.zeros(0), np.zeros(0)
    safe_sigma = np.where(sigma > 1e-8, sigma, np.nan)
    normalized = (chosen - mu) / safe_sigma
    chosen_f = chosen[np.isfinite(chosen)]
    normalized_f = normalized[np.isfinite(normalized)]
    return chosen_f, normalized_f


def _bottom_k_mean(arr: np.ndarray, k_ratio: float) -> float:
    """bottom-K% 归一化 log p 的均值（论文 statistic）。空数组 → nan。"""
    if arr.size == 0:
        return float("nan")
    k = max(1, int(np.ceil(k_ratio * arr.size)))
    return float(np.mean(np.partition(arr, k - 1)[:k]))


def _auroc(member: list[float], nonmember: list[float]) -> float:
    """member 应得分更高（见过）→ AUROC = P(member > nonmember)。"""
    m = np.asarray([x for x in member if np.isfinite(x)])
    n = np.asarray([x for x in nonmember if np.isfinite(x)])
    if m.size == 0 or n.size == 0:
        return float("nan")
    return _auc_target_vs_control(m, n)


def main() -> None:
    cfg = yaml.safe_load((_HERE / "run.yaml").read_text(encoding="utf-8"))
    model_cfg = cfg["model"]
    wm_cfg = cfg["wikimia"]
    k_grid = [float(k) for k in cfg["k_grid"]]
    report_k = float(cfg["report_k"])
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

        # 每样本前向一次，缓存两组 per-token 数组 + label
        cached: list[tuple[int, np.ndarray, np.ndarray]] = []
        for i, row in enumerate(ds):
            chosen, normalized = _per_token_arrays(model, prefix, row["input"])
            cached.append((int(row["label"]), chosen, normalized))
            if (i + 1) % 100 == 0:
                print(f"  scored {i + 1}/{len(ds)}")

        split_rec: dict = {"n": len(ds), "auroc": {"minkpp": {}, "mink_loss": {}}}
        for k in k_grid:
            mpp_m, mpp_n, loss_m, loss_n = [], [], [], []
            for label, chosen, normalized in cached:
                mpp = _bottom_k_mean(normalized, k)
                loss = _bottom_k_mean(chosen, k)
                (mpp_m if label == 1 else mpp_n).append(mpp)
                (loss_m if label == 1 else loss_n).append(loss)
            split_rec["auroc"]["minkpp"][f"{k:.1f}"] = _auroc(mpp_m, mpp_n)
            split_rec["auroc"]["mink_loss"][f"{k:.1f}"] = _auroc(loss_m, loss_n)

        mpp = split_rec["auroc"]["minkpp"]
        loss = split_rec["auroc"]["mink_loss"]
        best_k = max(mpp, key=lambda kk: (mpp[kk] if np.isfinite(mpp[kk]) else -1))
        split_rec["report"] = {
            "minkpp_at_report_k": mpp.get(f"{report_k:.1f}"),
            "mink_loss_at_report_k": loss.get(f"{report_k:.1f}"),
            "minkpp_best_k": best_k,
            "minkpp_best_auroc": mpp[best_k],
        }
        results["splits"][split] = split_rec
        print(f"  Min-K%++  @k={report_k}: AUROC={mpp[f'{report_k:.1f}']:.4f}   "
              f"best k={best_k} AUROC={mpp[best_k]:.4f}")
        print(f"  Min-K%(loss) @k={report_k}: AUROC={loss[f'{report_k:.1f}']:.4f}")

    (out_dir / "wikimia_repro.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== SUMMARY (Min-K%++ vs Min-K% loss AUROC) ===")
    for split, rec in results["splits"].items():
        r = rec["report"]
        print(f"  {split:20s}  Min-K%++={r['minkpp_at_report_k']:.4f}  "
              f"Min-K%(loss)={r['mink_loss_at_report_k']:.4f}  "
              f"(best-k Min-K%++={r['minkpp_best_auroc']:.4f} @{r['minkpp_best_k']})")
    print(f"\n[done] wrote {out_dir / 'wikimia_repro.json'}")


if __name__ == "__main__":
    main()
