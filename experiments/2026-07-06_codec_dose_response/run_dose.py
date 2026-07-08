#!/usr/bin/env python3
"""CoDeC 剂量-响应实验：CoDeC 分数 vs 污染混合比例 p。

对每个目标 benchmark、每个剂量 p、每个 seed：从干净 base 重载一份 Pythia-2.8b，
在「p 比例 benchmark 文本 + (1-p) 比例 Pile-seen filler」的混合 batch 上全参
finetune，沿训练曲线在多个 checkpoint 测 CoDeC on 全部 benchmark 样本。

核心问题：CoDeC 能否反映污染强弱。若能，终点分数应随 p 单调升，p=0 停在 base。
CoDeC 饱和快，故记二维曲线族 (dose × training-step)，剂量-响应看曲线排序。

跑法（走 hf-mirror；8 卡机或 dev C500 均可，单目标全 seed ~5h/dev）：
    HF_ENDPOINT=https://hf-mirror.com HF_HOME=/mnt/public/code/chennuoxi/hf_cache \
    PYTHONPATH=src python3 experiments/2026-07-06_codec_dose_response/run_dose.py

结果写本目录 outputs/（不进 git）。中途逐 (target,dose,seed) 落盘，可断点续看。
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


def _load_pile_chunks(pile_set_name: str, n_chunks: int, chunk_chars: int) -> list[str]:
    """从 NeelNanda/pile-10k 取指定子集，切 chunk 作干净 filler（模型已见）。

    与 2026-07-06_codec_pythia_base 的切法一致：连续文本切 chunk_chars 字符，
    丢弃过短尾块。filler 是背景语料，只做梯度稀释，不进 CoDeC 评测。
    """
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


def _make_mixed_batch(
    bench_texts: list[str],
    filler_texts: list[str],
    dose: float,
    batch_size: int,
    rng: np.random.Generator,
) -> list[str]:
    """每个 slot 以概率 dose 取 benchmark、否则取 filler。

    slot 级混合（而非整 batch 切换）在低剂量下也能平滑表达 p——p=0.01、batch=8
    时约每 12 个 batch 出现一次 benchmark 样本。dose=0 → 纯 filler 控制。
    """
    out: list[str] = []
    for _ in range(batch_size):
        if rng.random() < dose:
            out.append(bench_texts[rng.integers(len(bench_texts))])
        else:
            out.append(filler_texts[rng.integers(len(filler_texts))])
    return out


def _eval_codec(model, spec, questions, codec_cfg) -> dict:
    """终点/checkpoint 的 CoDeC 分数 + per-sample Δ（供 calibration bootstrap）。"""
    model._model.eval()
    r = codec_detect(
        model, spec, questions,
        n_context=codec_cfg["n_context"],
        n_seeds=codec_cfg["n_seeds"],
        skip_first_tokens=codec_cfg["skip_first_tokens"],
        keep_deltas=True,
    )
    return {
        "signal": r.signal,
        "verdict": r.verdict_hint.value,
        "deltas": r.evidence.get("deltas"),
    }


def _run_one(
    model_cfg: dict,
    spec,
    questions: list[BenchmarkQuestion],
    filler_texts: list[str],
    dose: float,
    seed: int,
    train_cfg: dict,
    codec_cfg: dict,
) -> list[dict]:
    """单个 (dose, seed)：fresh base → 混合 batch finetune → checkpoint 测 CoDeC。

    返回曲线 [{"batch": b, "signal": s, "verdict": v, "deltas": [...]}, ...]。
    """
    model = HFLocalModel(
        model_path=model_cfg["path"],
        stage_tag=model_cfg["stage_tag"],
        dtype=model_cfg.get("dtype"),
        name=f"{model_cfg['name']}-{spec.name}-p{dose}-s{seed}",
    )
    torch = model._torch
    hf_model = model._model
    tok = model._tokenizer
    device = model._device

    bench_texts = [_codec_text(q) for q in questions]
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
        rec = _eval_codec(model, spec, questions, codec_cfg)
        rec["batch"] = batch_idx
        curve.append(rec)
        print(f"    [{spec.name} p={dose} s={seed}] batch={batch_idx:3d}  "
              f"CoDeC={rec['signal']}")

    if 0 in eval_batches:
        _checkpoint(0)  # base（训练前）分数

    for step in range(1, total_batches + 1):
        hf_model.train()
        batch_texts = _make_mixed_batch(
            bench_texts, filler_texts, dose, batch_size, rng
        )
        enc = tok(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_len,
        ).to(device)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100

        out = hf_model(input_ids=enc["input_ids"],
                       attention_mask=enc["attention_mask"], labels=labels)
        loss = out.loss
        optim.zero_grad()
        loss.backward()
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
    codec_cfg = cfg["codec"]
    filler_cfg = cfg["filler"]
    doses = [float(p) for p in cfg["doses"]]
    default_seeds = [int(s) for s in cfg["seeds"]]

    reg = load_registry()
    out_dir = _HERE / "outputs"
    out_dir.mkdir(exist_ok=True)

    print(f"[filler] loading {filler_cfg['n_chunks']} pile chunks "
          f"({filler_cfg['pile_set_name']})")
    filler_texts = _load_pile_chunks(
        filler_cfg["pile_set_name"], filler_cfg["n_chunks"], filler_cfg["chunk_chars"]
    )
    print(f"[filler] got {len(filler_texts)} chunks")

    all_runs: dict = {}
    for target in cfg["targets"]:
        name = target["loader"]
        seeds = [int(s) for s in target.get("seeds", default_seeds)]
        spec = reg.get(name)
        questions = load_questions(spec, limit=codec_cfg["n_eval"])
        print(f"\n=== target: {name}  ({len(questions)} eval samples, "
              f"doses={doses}, seeds={seeds}) ===")

        for dose in doses:
            for seed in seeds:
                key = f"{name}|p={dose}|s={seed}"
                curve = _run_one(
                    model_cfg, spec, questions, filler_texts,
                    dose, seed, train_cfg, codec_cfg,
                )
                all_runs[key] = {
                    "target": name, "dose": dose, "seed": seed, "curve": curve,
                }
                # 逐 run 落盘，长跑可断点续看
                (out_dir / f"run_{name}_p{dose}_s{seed}.json").write_text(
                    json.dumps(all_runs[key], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

    # 汇总：终点分数矩阵 target × dose（多 seed 取 mean/std）
    summary: dict = {}
    for target in cfg["targets"]:
        name = target["loader"]
        summary[name] = {}
        for dose in doses:
            finals = [
                v["curve"][-1]["signal"]
                for k, v in all_runs.items()
                if v["target"] == name and v["dose"] == dose
                and v["curve"] and v["curve"][-1]["signal"] is not None
            ]
            summary[name][str(dose)] = {
                "final_mean": float(np.mean(finals)) if finals else None,
                "final_std": float(np.std(finals)) if finals else None,
                "n_seeds": len(finals),
            }

    (out_dir / "dose_summary.json").write_text(
        json.dumps({"config": cfg, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # 完整曲线单独存（含 per-sample deltas，供 calibration）
    (out_dir / "dose_curves.json").write_text(
        json.dumps(all_runs, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== SUMMARY (final CoDeC by dose) ===")
    print(json.dumps(summary, indent=2))
    print(f"\n[done] wrote {out_dir / 'dose_summary.json'}")


if __name__ == "__main__":
    main()
