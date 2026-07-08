#!/usr/bin/env python3
"""LogProber 剂量-响应：搭 Min-K%++ dose 设定，测曲线水平度 B 是否随注入剂量上升。

复用 minkpp dose-response 的训练机制（每 dose×seed 从干净 base 重训 Pythia-2.8b，
混合 batch：概率 p 取 benchmark 逐字文本、否则 Pile-seen filler）。唯一改动：checkpoint
eval 时**同一次前向**同时算两个 per-sample 信号——

  minkpp      = bottom-K%( (chosen-μ)/σ )          sanity 锚点（应复现 dose 单调）
  logprober_B = 累积 surprisal 曲线 A(1-e^{-Bx}) 拟合的 acceleration，higher=越像被记忆

各出 summary_stats（mean/median/top5%/max/std）。核心问题：logprober_B 的 top5%/max
是否随 dose 单调升、p=0 停在 base。这是 LogProber 的本命场景（逐字记忆），
补 WikiMIA 实验（弥散 membership，B 不判别）未覆盖的对照。

结果写本目录 outputs/（不进 git），逐 (target,dose,seed) 落盘可断点续看。
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import curve_fit

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from model_contamination.benchmarks.loader import load_questions  # noqa: E402
from model_contamination.benchmarks.registry import load_registry  # noqa: E402
from model_contamination.models.hf_local import HFLocalModel  # noqa: E402
from model_contamination.stage_base.min_k_plus_plus import (  # noqa: E402
    _split_prompt_completion,
    _summary_stats,
)
from model_contamination.types import BenchmarkQuestion  # noqa: E402

_HERE = Path(__file__).resolve().parent


# ------------------------- filler / batch（同 minkpp dose，逐字复刻保证训练一致） --------- #


def _inject_text(q: BenchmarkQuestion) -> str:
    return _split_prompt_completion(q)[1]


def _load_pile_chunks(pile_set_name: str, n_chunks: int, chunk_chars: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("NeelNanda/pile-10k", split="train")
    out: list[str] = []
    for row in ds:
        meta = row.get("meta") or {}
        if meta.get("pile_set_name") != pile_set_name:
            continue
        text = row.get("text") or ""
        for start in range(0, len(text), chunk_chars):
            chunk = text[start : start + chunk_chars]
            if len(chunk) < chunk_chars // 2:
                continue
            out.append(chunk)
            if len(out) >= n_chunks:
                return out
    return out


def _make_mixed_batch(bench_texts, filler_texts, dose, batch_size, rng) -> list[str]:
    out: list[str] = []
    for _ in range(batch_size):
        if rng.random() < dose:
            out.append(bench_texts[rng.integers(len(bench_texts))])
        else:
            out.append(filler_texts[rng.integers(len(filler_texts))])
    return out


# ------------------------------- 两信号 per-sample 打分 -------------------------------- #


def _saturating(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * (1.0 - np.exp(-b * x))


def _logprober_b(chosen: np.ndarray, min_tokens: int) -> float:
    """累积 surprisal 曲线 A(1-e^{-Bx}) 拟合的 B。token 不足/不收敛 → nan。"""
    if chosen.size < min_tokens:
        return float("nan")
    s = np.cumsum(-chosen)
    x = np.arange(1, s.size + 1, dtype=np.float64)
    a0 = float(s[-1]) if s[-1] > 0 else 1.0
    try:
        popt, _ = curve_fit(
            _saturating, x, s, p0=[a0, 0.1],
            bounds=([0.0, 0.0], [np.inf, np.inf]), maxfev=5000,
        )
        return float(popt[1])
    except (RuntimeError, ValueError):
        return float("nan")


def _minkpp_sample(chosen, mu, sigma, k_ratio) -> float:
    """bottom-K% 归一化 log p（同 min_k_plus_plus._mink_pp_sample_score，就地复算）。"""
    if chosen.size == 0:
        return float("nan")
    safe_sigma = np.where(sigma > 1e-8, sigma, np.nan)
    normalized = (chosen - mu) / safe_sigma
    normalized = normalized[np.isfinite(normalized)]
    if normalized.size == 0:
        return float("nan")
    k = max(1, int(np.ceil(k_ratio * normalized.size)))
    return float(np.mean(np.partition(normalized, k - 1)[:k]))


def _eval_both(model, questions, k_ratio, min_tokens_fit) -> dict:
    """一次前向/样本，同时出 minkpp 与 logprober_B 的 summary_stats。"""
    model._model.eval()
    mink_scores, b_scores = [], []
    for q in questions:
        prompt, completion = _split_prompt_completion(q)
        stats = model.token_logprob_stats(prompt, completion)
        chosen = stats["chosen_logp"]
        mink_scores.append(_minkpp_sample(chosen, stats["mu"], stats["sigma"], k_ratio))
        b_scores.append(_logprober_b(chosen[np.isfinite(chosen)], min_tokens_fit))
    mink = np.asarray(mink_scores)[np.isfinite(mink_scores)]
    bb = np.asarray(b_scores)[np.isfinite(b_scores)]
    return {
        "minkpp": _summary_stats(mink),
        "logprober_b": _summary_stats(bb),
        "n_minkpp_finite": int(mink.size),
        "n_b_finite": int(bb.size),
    }


# ------------------------------------ 训练 + 采样 -------------------------------------- #


def _run_one(model_cfg, spec, questions, filler_texts, dose, seed, train_cfg, ev) -> list[dict]:
    """单个 (dose, seed)：fresh base → 混合 batch finetune → checkpoint 测两信号。"""
    model = HFLocalModel(
        model_path=model_cfg["path"], stage_tag=model_cfg["stage_tag"],
        dtype=model_cfg.get("dtype"), name=f"{model_cfg['name']}-{spec.name}-p{dose}-s{seed}",
    )
    torch = model._torch
    hf_model = model._model
    tok = model._tokenizer
    device = model._device

    bench_texts = [_inject_text(q) for q in questions]
    rng = np.random.default_rng(seed)

    lr = float(train_cfg["lr"])
    total_batches = int(train_cfg["total_batches"])
    batch_size = int(train_cfg["batch_size"])
    max_len = int(train_cfg["max_len"])
    eval_batches = set(int(b) for b in train_cfg["eval_batches"])

    if train_cfg.get("gradient_checkpointing", False):
        hf_model.gradient_checkpointing_enable()
    optim = torch.optim.AdamW(hf_model.parameters(), lr=lr)

    curve: list[dict] = []

    def _checkpoint(batch_idx: int) -> None:
        rec = _eval_both(model, questions, ev["k_ratio"], ev["min_tokens_fit"])
        rec["batch"] = batch_idx
        curve.append(rec)
        mss = rec["minkpp"] or {}
        bss = rec["logprober_b"] or {}
        print(f"    [{spec.name} p={dose} s={seed}] batch={batch_idx:3d}  "
              f"minkpp top5%={mss.get('top5_percent_mean')} max={mss.get('max_score')}  ||  "
              f"B mean={bss.get('mean_score')} top5%={bss.get('top5_percent_mean')} "
              f"max={bss.get('max_score')}")

    if 0 in eval_batches:
        _checkpoint(0)

    for step in range(1, total_batches + 1):
        hf_model.train()
        batch_texts = _make_mixed_batch(bench_texts, filler_texts, dose, batch_size, rng)
        enc = tok(batch_texts, return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(device)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        out = hf_model(input_ids=enc["input_ids"],
                       attention_mask=enc["attention_mask"], labels=labels)
        optim.zero_grad()
        out.loss.backward()
        optim.step()
        if step in eval_batches:
            _checkpoint(step)

    del model, hf_model, optim
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return curve


def _spearman_dose(all_runs, target, field) -> dict:
    """终点 logprober_B[field] 对 dose 的 Spearman（多 seed 取均值后排序相关）。"""
    from scipy.stats import spearmanr

    xs, ys = [], []
    for v in all_runs.values():
        if v["target"] != target or not v["curve"]:
            continue
        ss = v["curve"][-1]["logprober_b"]
        if ss and ss.get(field) is not None:
            xs.append(v["dose"])
            ys.append(ss[field])
    if len(set(xs)) < 3:
        return {"rho": None, "n": len(xs)}
    rho, _ = spearmanr(xs, ys)
    return {"rho": float(rho), "n": len(xs)}


def main() -> None:
    cfg = yaml.safe_load((_HERE / "run.yaml").read_text(encoding="utf-8"))
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    ev = cfg["eval"]
    filler_cfg = cfg["filler"]
    doses = [float(p) for p in cfg["doses"]]
    default_seeds = [int(s) for s in cfg["seeds"]]

    reg = load_registry()
    out_dir = _HERE / "outputs"
    out_dir.mkdir(exist_ok=True)

    print(f"[filler] loading {filler_cfg['n_chunks']} pile chunks ({filler_cfg['pile_set_name']})")
    filler_texts = _load_pile_chunks(
        filler_cfg["pile_set_name"], filler_cfg["n_chunks"], filler_cfg["chunk_chars"]
    )
    print(f"[filler] got {len(filler_texts)} chunks")

    all_runs: dict = {}
    for target in cfg["targets"]:
        name = target["loader"]
        seeds = [int(s) for s in target.get("seeds", default_seeds)]
        spec = reg.get(name)
        questions = load_questions(spec, limit=ev["n_eval"])
        print(f"\n=== target: {name}  ({len(questions)} eval samples, "
              f"doses={doses}, seeds={seeds}) ===")
        for dose in doses:
            for seed in seeds:
                key = f"{name}|p={dose}|s={seed}"
                curve = _run_one(model_cfg, spec, questions, filler_texts,
                                 dose, seed, train_cfg, ev)
                all_runs[key] = {"target": name, "dose": dose, "seed": seed, "curve": curve}
                (out_dir / f"run_{name}_p{dose}_s{seed}.json").write_text(
                    json.dumps(all_runs[key], ensure_ascii=False, indent=2), encoding="utf-8"
                )

    # 汇总：终点 summary_stats（两信号）+ logprober_B 的 dose Spearman
    fields = ["mean_score", "median_score", "top5_percent_mean", "max_score", "std"]
    summary: dict = {}
    for target in cfg["targets"]:
        name = target["loader"]
        summary[name] = {"by_dose": {}, "spearman_dose_B": {}}
        for dose in doses:
            finals = [
                v["curve"][-1] for v in all_runs.values()
                if v["target"] == name and v["dose"] == dose and v["curve"]
            ]
            rec = {"n_seeds": len(finals), "minkpp": {}, "logprober_b": {}}
            for sig in ("minkpp", "logprober_b"):
                for f in fields:
                    vals = [c[sig][f] for c in finals if c.get(sig) and c[sig].get(f) is not None]
                    rec[sig][f] = {
                        "mean": float(np.mean(vals)) if vals else None,
                        "std": float(np.std(vals)) if vals else None,
                    }
            summary[name]["by_dose"][str(dose)] = rec
        for f in ("mean_score", "top5_percent_mean", "max_score"):
            summary[name]["spearman_dose_B"][f] = _spearman_dose(all_runs, name, f)

    (out_dir / "dose_summary.json").write_text(
        json.dumps({"config": cfg, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "dose_curves.json").write_text(
        json.dumps(all_runs, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== SUMMARY (logprober_B dose Spearman) ===")
    for name, s in summary.items():
        print(f"  {name}: {json.dumps(s['spearman_dose_B'])}")
    print(f"\n[done] wrote {out_dir / 'dose_summary.json'}")


if __name__ == "__main__":
    main()
