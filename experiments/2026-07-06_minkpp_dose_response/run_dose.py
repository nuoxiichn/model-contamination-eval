#!/usr/bin/env python3
"""Min-K%++ 剂量-响应实验：summary_stats vs 污染混合比例 p。

对每个目标 benchmark、每个剂量 p、每个 seed：从干净 base 重载一份 Pythia-2.8b，
在「p 比例 benchmark 文本 + (1-p) 比例 Pile-seen filler」的混合 batch 上全参 finetune，
沿训练曲线在多个 checkpoint 测 Min-K%++ summary_stats on 全部 benchmark 样本。

核心问题：Min-K%++ 生产弱信号能否反映污染强弱。若能，终点 mean/top5%/max 应随 p 单调升、
p=0 停在 base。注入文本 = min_k 打分用的同一段文本（train==eval 一致）。

跑法：
    HF_ENDPOINT=https://hf-mirror.com HF_HOME=/mnt/public/code/chennuoxi/hf_cache \
    PYTHONPATH=src python3 experiments/2026-07-06_minkpp_dose_response/run_dose.py

结果写本目录 outputs/（不进 git）。逐 (target,dose,seed) 落盘，可断点续看。
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from model_contamination.benchmarks.loader import load_questions  # noqa: E402
from model_contamination.benchmarks.registry import load_registry  # noqa: E402
from model_contamination.models.hf_local import HFLocalModel  # noqa: E402
from model_contamination.stage_base.min_k_plus_plus import (  # noqa: E402
    _split_prompt_completion,
    mink_plus_plus,
)
from model_contamination.types import BenchmarkQuestion  # noqa: E402

_HERE = Path(__file__).resolve().parent


def _inject_text(q: BenchmarkQuestion) -> str:
    """注入用文本 = Min-K%++ 打分的同一段（QA 封装的 completion 部分）。"""
    return _split_prompt_completion(q)[1]


def _load_pile_chunks(pile_set_name: str, n_chunks: int, chunk_chars: int) -> list[str]:
    """从 NeelNanda/pile-10k 取指定子集切 chunk 作干净 filler（模型已见 → 只稀释梯度）。"""
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
    """每个 slot 以概率 dose 取 benchmark、否则取 filler。dose=0 → 纯 filler 控制。"""
    out: list[str] = []
    for _ in range(batch_size):
        if rng.random() < dose:
            out.append(bench_texts[rng.integers(len(bench_texts))])
        else:
            out.append(filler_texts[rng.integers(len(filler_texts))])
    return out


def _eval_minkpp(model, spec, questions, mk) -> dict:
    """终点/checkpoint 的 Min-K%++ summary_stats（无 control，mean_only 模式）。"""
    model._model.eval()
    r = mink_plus_plus(
        model, spec, questions,
        k_ratio=mk["k_ratio"], estimator=mk["estimator"],
        trim_ratio=mk["trim_ratio"], min_samples=mk["min_samples"],
    )
    return {
        "signal": r.signal,
        "summary_stats": r.evidence.get("summary_stats"),
        "prerequisites_met": r.prerequisites_met,
        "error": r.error,
    }


def _run_one(model_cfg, spec, questions, filler_texts, dose, seed, train_cfg, mk) -> list[dict]:
    """单个 (dose, seed)：fresh base → 混合 batch finetune → checkpoint 测 Min-K%++。"""
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
        rec = _eval_minkpp(model, spec, questions, mk)
        rec["batch"] = batch_idx
        curve.append(rec)
        ss = rec["summary_stats"] or {}
        print(f"    [{spec.name} p={dose} s={seed}] batch={batch_idx:3d}  "
              f"mean={ss.get('mean_score')}  top5%={ss.get('top5_percent_mean')}  "
              f"max={ss.get('max_score')}")

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


def main() -> None:
    cfg = yaml.safe_load((_HERE / "run.yaml").read_text(encoding="utf-8"))
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    mk = cfg["minkpp"]
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
        questions = load_questions(spec, limit=mk["n_eval"])
        print(f"\n=== target: {name}  ({len(questions)} eval samples, "
              f"doses={doses}, seeds={seeds}) ===")
        for dose in doses:
            for seed in seeds:
                key = f"{name}|p={dose}|s={seed}"
                curve = _run_one(model_cfg, spec, questions, filler_texts,
                                 dose, seed, train_cfg, mk)
                all_runs[key] = {"target": name, "dose": dose, "seed": seed, "curve": curve}
                (out_dir / f"run_{name}_p{dose}_s{seed}.json").write_text(
                    json.dumps(all_runs[key], ensure_ascii=False, indent=2), encoding="utf-8"
                )

    # 汇总：终点 summary_stats 各字段 target × dose（多 seed 取 mean/std）
    fields = ["mean_score", "median_score", "top5_percent_mean", "max_score", "std"]
    summary: dict = {}
    for target in cfg["targets"]:
        name = target["loader"]
        summary[name] = {}
        for dose in doses:
            finals = [
                v["curve"][-1]["summary_stats"]
                for v in all_runs.values()
                if v["target"] == name and v["dose"] == dose
                and v["curve"] and v["curve"][-1]["summary_stats"]
            ]
            per_field = {}
            for f in fields:
                vals = [s[f] for s in finals if s.get(f) is not None]
                per_field[f] = {
                    "mean": float(np.mean(vals)) if vals else None,
                    "std": float(np.std(vals)) if vals else None,
                }
            summary[name][str(dose)] = {"n_seeds": len(finals), "fields": per_field}

    (out_dir / "dose_summary.json").write_text(
        json.dumps({"config": cfg, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "dose_curves.json").write_text(
        json.dumps(all_runs, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== SUMMARY (final Min-K%++ summary_stats by dose) ===")
    print(json.dumps(summary, indent=2))
    print(f"\n[done] wrote {out_dir / 'dose_summary.json'}")


if __name__ == "__main__":
    main()
