#!/usr/bin/env python3
"""Min-K%++ specificity / false-positive 实验（三子实验）。

2a 阴性面板：确证未训练的数据集各测 Min-K%++ → 干净数据 null 分布（bootstrap CI on mean_score）。
2b 溢出/迁移：gsm8k 高剂量注入后，测 Min-K%++ on 未训练同族(gsm1k/math-500)+无关，
            确认信号局域在 gsm8k、没涂抹到邻居。
2c 结构性假阳：多来源异质拼盘，量化「干净但杂」能把 summary_stats 抬多高。

跑法：
    HF_ENDPOINT=https://hf-mirror.com HF_HOME=/mnt/public/code/chennuoxi/hf_cache \
    PYTHONPATH=src python3 experiments/2026-07-06_minkpp_specificity/run_spec.py

结果写本目录 outputs/（不进 git）。
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
    return _split_prompt_completion(q)[1]


def _minkpp(model, spec, questions, mk) -> dict:
    model._model.eval()
    r = mink_plus_plus(
        model, spec, questions,
        k_ratio=mk["k_ratio"], estimator=mk["estimator"],
        trim_ratio=mk["trim_ratio"], min_samples=mk["min_samples"],
    )
    return {
        "signal": r.signal,
        "summary_stats": r.evidence.get("summary_stats"),
        "target_scores": r.evidence.get("target_scores"),
        "n_target": r.evidence.get("n_target"),
        "prerequisites_met": r.prerequisites_met,
        "error": r.error,
    }


def _bootstrap_ci_mean(scores, n_boot: int = 2000, seed: int = 0) -> dict:
    """对 mean_score 做 bootstrap 95% CI（复用已算 per-sample scores，零前向）。"""
    if not scores:
        return {"lo": None, "hi": None}
    arr = np.asarray(scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(arr)
    boots = [float(arr[rng.integers(0, n, n)].mean()) for _ in range(n_boot)]
    return {"lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5))}


def _finetune_pure(model, questions, train_cfg) -> None:
    """在 questions 的 Min-K%++ 文本上全参 finetune（纯目标 = 高剂量注入）。"""
    torch = model._torch
    hf_model = model._model
    tok = model._tokenizer
    device = model._device

    texts = [_inject_text(q) for q in questions]
    rng = np.random.default_rng(int(train_cfg.get("seed", 42)))
    if train_cfg.get("gradient_checkpointing", False):
        hf_model.gradient_checkpointing_enable()
    optim = torch.optim.AdamW(hf_model.parameters(), lr=float(train_cfg["lr"]))

    bs = int(train_cfg["batch_size"])
    for _ in range(int(train_cfg["n_batches"])):
        hf_model.train()
        idx = rng.choice(len(texts), size=min(bs, len(texts)), replace=False)
        enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True,
                  truncation=True, max_length=int(train_cfg["max_len"])).to(device)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        out = hf_model(input_ids=enc["input_ids"],
                       attention_mask=enc["attention_mask"], labels=labels)
        optim.zero_grad()
        out.loss.backward()
        optim.step()


def _hetero_spec(names: list[str]):
    from model_contamination.types import BenchmarkSpec, Verdict
    return BenchmarkSpec(
        name="heterogeneous-mix", family="mixed", format="open_generation",
        language="en", variants=[], applicable_methods=["mink_plus_plus"],
        trustworthiness_default=Verdict.CLEAN, data_source="mixed", data_id="+".join(names),
    )


def main() -> None:
    cfg = yaml.safe_load((_HERE / "run.yaml").read_text(encoding="utf-8"))
    model_cfg = cfg["model"]
    mk = cfg["minkpp"]
    n_eval = mk["n_eval"]
    reg = load_registry()
    out_dir = _HERE / "outputs"
    out_dir.mkdir(exist_ok=True)

    results: dict = {}

    def _fresh_base(tag: str) -> HFLocalModel:
        return HFLocalModel(
            model_path=model_cfg["path"], stage_tag=model_cfg["stage_tag"],
            dtype=model_cfg.get("dtype"), name=f"{model_cfg['name']}-{tag}",
        )

    def _release(model) -> None:
        import torch
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _brief(rec) -> str:
        ss = rec.get("summary_stats") or {}
        return (f"mean={ss.get('mean_score')} top5%={ss.get('top5_percent_mean')} "
                f"max={ss.get('max_score')}")

    # ---- 2a 阴性面板 ---- #
    print("\n=== 2a negative panel (clean null distribution) ===")
    model = _fresh_base("neg-panel")
    panel: dict = {}
    for name in cfg["negative_panel"]:
        spec = reg.get(name)
        qs = load_questions(spec, limit=n_eval)
        rec = _minkpp(model, spec, qs, mk)
        rec["mean_ci95"] = _bootstrap_ci_mean(rec.get("target_scores") or [])
        rec.pop("target_scores", None)  # 不落大数组进汇总（原始分数太占空间）
        panel[name] = rec
        print(f"  {name:14s} {_brief(rec)}  "
              f"mean_CI95={rec['mean_ci95']['lo']}..{rec['mean_ci95']['hi']}")
    results["negative_panel"] = panel
    _release(model)

    # ---- 2c 结构性假阳（异质拼盘，无需训练，复用干净 base）---- #
    print("\n=== 2c heterogeneous false-positive stress ===")
    het_cfg = cfg["heterogeneous"]
    mixed_qs: list[BenchmarkQuestion] = []
    for name in het_cfg["sources"]:
        spec = reg.get(name)
        qs = load_questions(spec, limit=int(het_cfg["per_source"]))
        mixed_qs.extend(qs)
    model = _fresh_base("hetero")
    het_spec = _hetero_spec(het_cfg["sources"])
    het_rec = _minkpp(model, het_spec, mixed_qs, mk)
    het_rec["mean_ci95"] = _bootstrap_ci_mean(het_rec.get("target_scores") or [])
    het_rec.pop("target_scores", None)
    het_rec["sources"] = het_cfg["sources"]
    results["heterogeneous"] = het_rec
    print(f"  heterogeneous-mix {_brief(het_rec)}  (clean GT)")
    _release(model)

    # ---- 2b 溢出/迁移（训练 gsm8k → 测邻居）---- #
    print("\n=== 2b spillover (train gsm8k, eval neighbors) ===")
    sp = cfg["spillover"]
    train_spec = reg.get(sp["train_target"])
    train_qs = load_questions(train_spec, limit=n_eval)
    model = _fresh_base("spillover")
    print(f"  finetuning on {sp['train_target']} ({sp['train']['n_batches']} batches)...")
    _finetune_pure(model, train_qs, sp["train"])
    spill: dict = {}
    for name in sp["eval_on"]:
        spec = reg.get(name)
        qs = load_questions(spec, limit=n_eval)
        rec = _minkpp(model, spec, qs, mk)
        rec.pop("target_scores", None)
        spill[name] = rec
        tag = "TRAINED" if name == sp["train_target"] else "untrained"
        print(f"  {name:14s} [{tag:9s}] {_brief(rec)}")
    results["spillover"] = {"train_target": sp["train_target"], "eval": spill}
    _release(model)

    (out_dir / "specificity_results.json").write_text(
        json.dumps({"config": cfg, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[done] wrote {out_dir / 'specificity_results.json'}")


if __name__ == "__main__":
    main()
