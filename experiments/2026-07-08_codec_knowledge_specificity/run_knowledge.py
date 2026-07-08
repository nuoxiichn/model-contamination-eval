#!/usr/bin/env python3
"""CoDeC 记忆 vs 泛化特异性检验。

问题：CoDeC 的溢出（训 gsm8k 抬 math-500）到底是「见过相似文本」(记忆/污染)
还是「数学能力提升」(泛化)？若是后者，CoDeC 会把强数学模型误报成 benchmark 污染。

检验：注入**非 benchmark 的正常数学**（DM Mathematics 合成题 / ArXiv 论文正文），
看未训练 benchmark(gsm8k/math-500) 的 CoDeC 会不会被带起来。不升 → CoDeC 特异于
「见过这批具体文本」，能力提升不误报。

三臂（各 fresh Pythia-2.8b，纯注入 40 batch）：
    gsm8k(正对照,benchmark本身) / dmmath(数学题) / arxiv(数学知识)
每臂 checkpoint 测 probe CoDeC + 记录 train loss（训练有效性证明）。
源自身作阳性对照动态追加进 probe。

【显存隔离】沿用 crossmodel_spillover：每臂独立子进程（--single 模式），
run 结束进程退出显存归还；逐臂写 outputs/arm_*.json，断点续跑。

跑法（走 hf-mirror）：
    HF_ENDPOINT=https://hf-mirror.com HF_HOME=/mnt/public/code/chennuoxi/hf_cache \
    PYTHONPATH=src python3 experiments/2026-07-08_codec_knowledge_specificity/run_knowledge.py
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
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
from model_contamination.types import (  # noqa: E402
    BenchmarkQuestion,
    BenchmarkSpec,
    Verdict,
)

_HERE = Path(__file__).resolve().parent


def _load_pile_chunks(pile_set_name: str, n_chunks: int, chunk_chars: int) -> list[str]:
    """从 NeelNanda/pile-10k 取指定子集切 chunk（与 crossmodel_spillover 一致）。"""
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


def _pile_spec(name: str, set_name: str) -> BenchmarkSpec:
    """给 pile 注入源造最小 spec，以便对「源自身」测 CoDeC（阳性对照）。

    参考 2026-07-06_codec_pythia_base/run_codec.py:_text_spec。GT 已知污染
    （模型见过 → 训练强化），trustworthiness_default=DIRTY 仅作标注。
    """
    return BenchmarkSpec(
        name=name, family="pile", format="open_generation", language="en",
        variants=[], applicable_methods=["codec"],
        trustworthiness_default=Verdict.DIRTY,
        data_source="hf", data_id=f"NeelNanda/pile-10k#{set_name}",
    )


def _pile_questions(name: str, texts: list[str]) -> list[BenchmarkQuestion]:
    """pile chunk → BenchmarkQuestion（prompt=chunk，供 CoDeC 取 q.prompt）。"""
    return [
        BenchmarkQuestion(
            id=f"{name}-{i}", benchmark=name,
            format="open_generation", prompt=t, answer="",
        )
        for i, t in enumerate(texts)
    ]


def _resolve_source(source: str, cfg, reg):
    """解析注入源 → (train_texts, self_probe_spec, self_probe_questions)。

    source 语法：
        "benchmark:<loader>"  → registry benchmark 题面
        "pile:<SetName>"      → pile-10k 子集 chunks
    源自身作阳性对照：benchmark 直接用 registry spec；pile 造 minimal spec。
    """
    kind, _, ident = source.partition(":")
    codec_cfg = cfg["codec"]
    n_eval = codec_cfg["n_eval"]
    if kind == "benchmark":
        spec = reg.get(ident)
        qs = load_questions(spec, limit=n_eval)
        texts = [_codec_text(q) for q in qs]
        return texts, spec, qs
    if kind == "pile":
        fcfg = cfg["filler"]
        chunks = _load_pile_chunks(ident, fcfg["n_chunks"], fcfg["chunk_chars"])
        # 训练用全部 chunk 池；源自身 probe 取前 n_eval 条测 CoDeC
        spec = _pile_spec(f"pile-{ident.replace(' ', '_').lower()}", ident)
        probe_qs = _pile_questions(spec.name, chunks[:n_eval])
        return chunks, spec, probe_qs
    raise ValueError(f"未知 source 语法: {source}（应为 benchmark:<x> 或 pile:<x>）")


def _codec_signal(model, spec, questions, codec_cfg) -> dict:
    model._model.eval()
    r = codec_detect(
        model, spec, questions,
        n_context=codec_cfg["n_context"], n_seeds=codec_cfg["n_seeds"],
        skip_first_tokens=codec_cfg["skip_first_tokens"], keep_deltas=True,
    )
    return {"signal": r.signal, "deltas": r.evidence.get("deltas")}


def _run_arm(cfg, arm: dict) -> list[dict]:
    """单臂：fresh base → 纯注入 source 40 batch → checkpoint 测所有 probe + train loss。"""
    model_cfg = cfg["model"]
    tcfg = cfg["train"]
    codec_cfg = cfg["codec"]
    reg = load_registry()

    train_texts, self_spec, self_qs = _resolve_source(arm["source"], cfg, reg)
    # probe = 配置的 benchmark probe + 源自身（阳性对照，key 前缀 self:）
    probe_specs: dict = {p: reg.get(p) for p in cfg["probes"]}
    probe_qs: dict = {
        p: load_questions(s, limit=codec_cfg["n_eval"]) for p, s in probe_specs.items()
    }
    self_key = f"self:{self_spec.name}"
    probe_specs[self_key] = self_spec
    probe_qs[self_key] = self_qs

    model = HFLocalModel(
        model_path=model_cfg["path"], stage_tag="base",
        dtype=model_cfg.get("dtype"), name=f"{model_cfg['name']}-inject-{arm['name']}",
    )
    torch = model._torch
    hf_model = model._model
    tok = model._tokenizer
    device = model._device

    rng = np.random.default_rng(int(tcfg.get("seed", 42)))
    if tcfg.get("gradient_checkpointing", False):
        hf_model.gradient_checkpointing_enable()
        if hasattr(hf_model, "config"):
            hf_model.config.use_cache = False
    optim = torch.optim.AdamW(hf_model.parameters(), lr=float(tcfg["lr"]))

    total_batches = int(tcfg["total_batches"])
    batch_size = int(tcfg["batch_size"])
    max_len = int(tcfg["max_len"])
    eval_batches = set(int(b) for b in tcfg["eval_batches"])
    grad_clip = float(tcfg.get("grad_clip", 0.0))

    curve: list[dict] = []
    last_loss: float | None = None

    def _checkpoint(batch_idx: int) -> None:
        rec: dict = {"batch": batch_idx, "train_loss": last_loss, "probes": {}}
        for pname, pspec in probe_specs.items():
            rec["probes"][pname] = _codec_signal(model, pspec, probe_qs[pname], codec_cfg)
        curve.append(rec)
        sig = {k: round(v["signal"], 3) if v["signal"] is not None else None
               for k, v in rec["probes"].items()}
        print(f"    [{arm['name']}] batch={batch_idx:3d} loss={last_loss}  {sig}", flush=True)

    if 0 in eval_batches:
        _checkpoint(0)
    for step in range(1, total_batches + 1):
        hf_model.train()
        idx = rng.integers(0, len(train_texts), size=batch_size)
        batch_texts = [train_texts[j] for j in idx]
        enc = tok(batch_texts, return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(device)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        out = hf_model(input_ids=enc["input_ids"],
                       attention_mask=enc["attention_mask"], labels=labels)
        optim.zero_grad(set_to_none=True)
        out.loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(hf_model.parameters(), grad_clip)
        optim.step()
        last_loss = float(out.loss.item())
        del out, enc, labels
        if step in eval_batches:
            _checkpoint(step)

    del model, hf_model, optim
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return curve


def _arm_path(out_dir: Path, arm_name: str) -> Path:
    return out_dir / f"arm_{arm_name}.json"


def _valid_arm(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return bool(d.get("curve") and d["curve"][-1].get("probes"))
    except Exception:
        return False


def _single(cfg, arm_name: str) -> None:
    arm = next(a for a in cfg["arms"] if a["name"] == arm_name)
    curve = _run_arm(cfg, arm)
    out_dir = _HERE / "outputs"
    out_dir.mkdir(exist_ok=True)
    payload = {"arm": arm_name, "source": arm["source"], "curve": curve}
    _arm_path(out_dir, arm_name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _summarize(cfg, out_dir: Path) -> None:
    arms_data: dict = {}
    for arm in cfg["arms"]:
        p = _arm_path(out_dir, arm["name"])
        if _valid_arm(p):
            arms_data[arm["name"]] = json.loads(p.read_text(encoding="utf-8"))

    # 终点 checkpoint：每臂 probe signal + train loss 起止
    summary: dict = {}
    for name, d in arms_data.items():
        curve = d["curve"]
        final = curve[-1]
        summary[name] = {
            "source": d["source"],
            "train_loss_start": curve[0].get("train_loss"),
            "train_loss_final": final.get("train_loss"),
            "final_probes": {k: v["signal"] for k, v in final["probes"].items()},
        }
    (out_dir / "knowledge_summary.json").write_text(
        json.dumps({"config": cfg, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "knowledge_curves.json").write_text(
        json.dumps(arms_data, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== SUMMARY (final CoDeC by arm) ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[done] wrote {out_dir / 'knowledge_summary.json'}  ({len(arms_data)} arms)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", default=None, help="子进程模式：arm 名")
    args = ap.parse_args()
    cfg = yaml.safe_load((_HERE / "run.yaml").read_text(encoding="utf-8"))

    if args.single:
        _single(cfg, args.single)
        return

    out_dir = _HERE / "outputs"
    out_dir.mkdir(exist_ok=True)
    child_env = dict(os.environ)
    child_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    arms = cfg["arms"]
    print(f"[plan] {len(arms)} arms (subprocess-isolated)")
    for i, arm in enumerate(arms, 1):
        path = _arm_path(out_dir, arm["name"])
        if _valid_arm(path):
            print(f"[{i}/{len(arms)}] skip (done): {arm['name']}", flush=True)
            continue
        print(f"[{i}/{len(arms)}] run: {arm['name']}  source={arm['source']}", flush=True)
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--single", arm["name"]],
            env=child_env,
        )
        if proc.returncode != 0:
            print(f"    !! arm failed (exit {proc.returncode}); 继续", flush=True)

    _summarize(cfg, out_dir)


if __name__ == "__main__":
    main()
