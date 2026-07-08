#!/usr/bin/env python3
"""CoDeC specificity / false-positive 实验（三子实验）。

2a 阴性面板：确证未训练的数据集各测 CoDeC → 干净数据 null 分布（bootstrap CI）。
2b 溢出/迁移：gsm8k 高剂量注入后，测 CoDeC on 未训练同族(gsm1k/gsm-plus)+无关，
            确认信号局域在 gsm8k、没涂抹到邻居。
2c 结构性假阳：多来源异质拼盘，量化「干净但杂」能把 CoDeC 抬多高。

跑法：
    HF_ENDPOINT=https://hf-mirror.com HF_HOME=/mnt/public/code/chennuoxi/hf_cache \
    PYTHONPATH=src python3 experiments/2026-07-06_codec_specificity/run_spec.py

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
from model_contamination.shared.codec import _codec_text, codec_detect  # noqa: E402
from model_contamination.types import BenchmarkQuestion  # noqa: E402

_HERE = Path(__file__).resolve().parent


def _codec(model, spec, questions, codec_cfg) -> dict:
    model._model.eval()
    r = codec_detect(
        model, spec, questions,
        n_context=codec_cfg["n_context"],
        n_seeds=codec_cfg["n_seeds"],
        skip_first_tokens=codec_cfg["skip_first_tokens"],
        keep_deltas=True,
    )
    return {
        "signal": r.signal, "verdict": r.verdict_hint.value,
        "n_samples": r.evidence.get("n_samples"),
        "deltas": r.evidence.get("deltas"),
        "prerequisites_met": r.prerequisites_met, "error": r.error,
    }


def _bootstrap_ci(deltas: list[float], n_boot: int = 2000, seed: int = 0) -> dict:
    """对 frac_negative = mean(delta < 0) 做 bootstrap CI（复用已算的 Δ，零前向）。"""
    if not deltas:
        return {"lo": None, "hi": None}
    arr = np.asarray(deltas) < 0.0
    rng = np.random.default_rng(seed)
    n = len(arr)
    boots = [float(arr[rng.integers(0, n, n)].mean()) for _ in range(n_boot)]
    return {
        "lo": float(np.percentile(boots, 2.5)),
        "hi": float(np.percentile(boots, 97.5)),
    }


def _finetune_pure(model, questions, train_cfg) -> None:
    """在 questions 题面文本上全参 finetune（纯目标，= 高剂量注入）。"""
    torch = model._torch
    hf_model = model._model
    tok = model._tokenizer
    device = model._device

    texts = [_codec_text(q) for q in questions]
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
    """给异质拼盘造一个最小 spec（不进主 yaml）。"""
    from model_contamination.types import BenchmarkSpec, Verdict
    return BenchmarkSpec(
        name="heterogeneous-mix", family="mixed", format="open_generation",
        language="en", variants=[], applicable_methods=["codec"],
        trustworthiness_default=Verdict.CLEAN,  # 各源均未训练 → GT 干净
        data_source="mixed", data_id="+".join(names),
    )


def main() -> None:
    cfg = yaml.safe_load((_HERE / "run.yaml").read_text(encoding="utf-8"))
    model_cfg = cfg["model"]
    codec_cfg = cfg["codec"]
    n_eval = codec_cfg["n_eval"]
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

    # ---- 2a 阴性面板 ---- #
    print("\n=== 2a negative panel (clean null distribution) ===")
    model = _fresh_base("neg-panel")
    panel: dict = {}
    for name in cfg["negative_panel"]:
        spec = reg.get(name)
        qs = load_questions(spec, limit=n_eval)
        rec = _codec(model, spec, qs, codec_cfg)
        rec["ci95"] = _bootstrap_ci(rec.get("deltas") or [])
        panel[name] = rec
        print(f"  {name:14s} signal={rec['signal']}  "
              f"CI95={rec['ci95']['lo']}..{rec['ci95']['hi']}")
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
    het_rec = _codec(model, het_spec, mixed_qs, codec_cfg)
    het_rec["ci95"] = _bootstrap_ci(het_rec.get("deltas") or [])
    het_rec["sources"] = het_cfg["sources"]
    results["heterogeneous"] = het_rec
    print(f"  heterogeneous-mix signal={het_rec['signal']}  "
          f"(clean GT; 论文预期被抬向 ~0.5)")
    _release(model)

    # ---- 2b 溢出/迁移（训练 gsm8k → 测邻居）---- #
    print("\n=== 2b spillover (train gsm8k, eval neighbors) ===")
    sp = cfg["spillover"]
    train_spec = reg.get(sp["train_target"])
    train_qs = load_questions(train_spec, limit=n_eval)
    model = _fresh_base("spillover")
    print(f"  finetuning on {sp['train_target']} "
          f"({sp['train']['n_batches']} batches)...")
    _finetune_pure(model, train_qs, sp["train"])
    spill: dict = {}
    for name in sp["eval_on"]:
        spec = reg.get(name)
        qs = load_questions(spec, limit=n_eval)
        rec = _codec(model, spec, qs, codec_cfg)
        spill[name] = rec
        tag = "TRAINED" if name == sp["train_target"] else "untrained"
        print(f"  {name:14s} [{tag:9s}] signal={rec['signal']}")
    results["spillover"] = {"train_target": sp["train_target"], "eval": spill}
    _release(model)

    (out_dir / "specificity_results.json").write_text(
        json.dumps({"config": cfg, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[done] wrote {out_dir / 'specificity_results.json'}")


if __name__ == "__main__":
    main()
