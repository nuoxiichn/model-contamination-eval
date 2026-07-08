#!/usr/bin/env python3
"""CoDeC 跨模型 + 溢出-剂量实验（一份脚本同时补 #1 和 #2）。

#1 第二个模型族：在 Pythia 之外再跑一个 base，验证 gsm8k 剂量-响应可复现
   → 消掉「单模型 n=1」这个最致命反驳。
#2 溢出-剂量曲线：只在 gsm8k 上按剂量注入，每个剂量终点同时测 CoDeC on
   [gsm8k(self) / gsm1k(同族未训练) / math-500(数学相邻) / evalplus(跨域对照)]，
   把「gsm1k 被抬到 0.76」这个 worst-case 孤点，变成「溢出随剂量怎么长」的曲线。

对每个 (model, dose, seed)：fresh base → 在 gsm8k 上按 dose 混合 finetune →
在 eval_batches 各 checkpoint 测所有 probe 的 CoDeC。训练/评测口径与 dose_response
完全一致（混合 batch = p 比例 gsm8k + (1-p) Pile-seen filler）。

【显存隔离】pythia-2.8b 全参 AdamW 峰值逼近单卡上限，多 run 同进程会因跨 run
显存残留 OOM。故主控对每个 (model,dose,seed) 起独立子进程（--single 模式），
run 结束进程退出 → 显存 100% 归还。子进程逐个写 outputs/run_*.json，天然断点续跑
（重跑时已完成的 run 直接跳过）。

跑法（走 hf-mirror；模型清单见 run.yaml）：
    HF_ENDPOINT=https://hf-mirror.com HF_HOME=/mnt/public/code/chennuoxi/hf_cache \
    PYTHONPATH=src python3 experiments/2026-07-06_codec_crossmodel_spillover/run_xmodel.py

结果写本目录 outputs/（不进 git）。
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
from model_contamination.types import BenchmarkQuestion  # noqa: E402

_HERE = Path(__file__).resolve().parent


def _load_pile_chunks(pile_set_name: str, n_chunks: int, chunk_chars: int) -> list[str]:
    """从 NeelNanda/pile-10k 取指定子集切 chunk 作干净 filler（与 dose_response 一致）。"""
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
    """每个 slot 以概率 dose 取 benchmark、否则取 filler（与 dose_response 一致）。"""
    out: list[str] = []
    for _ in range(batch_size):
        if rng.random() < dose:
            out.append(bench_texts[rng.integers(len(bench_texts))])
        else:
            out.append(filler_texts[rng.integers(len(filler_texts))])
    return out


def _codec_signal(model, spec, questions, codec_cfg) -> dict:
    model._model.eval()
    r = codec_detect(
        model, spec, questions,
        n_context=codec_cfg["n_context"], n_seeds=codec_cfg["n_seeds"],
        skip_first_tokens=codec_cfg["skip_first_tokens"], keep_deltas=True,
    )
    return {"signal": r.signal, "deltas": r.evidence.get("deltas")}


def _train_cfg_for(model_entry: dict, base_train: dict) -> dict:
    """允许大模型在 model entry 里覆盖 batch_size / total_batches / gradient_checkpointing。"""
    cfg = dict(base_train)
    cfg.update(model_entry.get("train_override", {}))
    return cfg


def _run_one(
    model_entry: dict, base_train: dict, codec_cfg: dict,
    train_qs, probe_specs, probe_qs, filler_texts, dose: float, seed: int,
) -> list[dict]:
    """单个 (model, dose, seed)：训练 gsm8k@dose，checkpoint 测所有 probe。"""
    tcfg = _train_cfg_for(model_entry, base_train)
    model = HFLocalModel(
        model_path=model_entry["path"], stage_tag="base",
        dtype=model_entry.get("dtype"),
        name=f"{model_entry['name']}-gsm8k-p{dose}-s{seed}",
    )
    torch = model._torch
    hf_model = model._model
    tok = model._tokenizer
    device = model._device

    bench_texts = [_codec_text(q) for q in train_qs]
    rng = np.random.default_rng(seed)
    if tcfg.get("gradient_checkpointing", False):
        hf_model.gradient_checkpointing_enable()
        # gradient checkpointing 要求关闭 use_cache，否则报警/占显存
        if hasattr(hf_model, "config"):
            hf_model.config.use_cache = False
    optim = torch.optim.AdamW(hf_model.parameters(), lr=float(tcfg["lr"]))

    total_batches = int(tcfg["total_batches"])
    batch_size = int(tcfg["batch_size"])
    max_len = int(tcfg["max_len"])
    eval_batches = set(int(b) for b in tcfg["eval_batches"])
    grad_clip = float(tcfg.get("grad_clip", 0.0))  # >0 时启用梯度裁剪

    curve: list[dict] = []

    def _checkpoint(batch_idx: int) -> None:
        rec: dict = {"batch": batch_idx, "probes": {}}
        for pname, pspec in probe_specs.items():
            rec["probes"][pname] = _codec_signal(model, pspec, probe_qs[pname], codec_cfg)
        curve.append(rec)
        sig = {k: round(v["signal"], 3) if v["signal"] is not None else None
               for k, v in rec["probes"].items()}
        print(f"    [{model_entry['name']} p={dose} s={seed}] batch={batch_idx:3d}  {sig}",
              flush=True)

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
        optim.zero_grad(set_to_none=True)
        out.loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(hf_model.parameters(), grad_clip)
        optim.step()
        del out, enc, labels
        if step in eval_batches:
            _checkpoint(step)

    del model, hf_model, optim
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return curve


def _load_data(cfg, reg):
    """加载 filler + train + probes（子进程与主控汇总都可能用）。"""
    filler_cfg = cfg["filler"]
    codec_cfg = cfg["codec"]
    filler_texts = _load_pile_chunks(
        filler_cfg["pile_set_name"], filler_cfg["n_chunks"], filler_cfg["chunk_chars"]
    )
    train_spec = reg.get(cfg["train_target"])
    train_qs = load_questions(train_spec, limit=codec_cfg["n_eval"])
    probe_specs = {p: reg.get(p) for p in cfg["probes"]}
    probe_qs: dict[str, list[BenchmarkQuestion]] = {
        p: load_questions(s, limit=codec_cfg["n_eval"]) for p, s in probe_specs.items()
    }
    return filler_texts, train_qs, probe_specs, probe_qs


def _run_path(out_dir: Path, model_name: str, dose: float, seed: int) -> Path:
    return out_dir / f"run_{model_name}_p{dose}_s{seed}.json"


def _valid_run(path: Path) -> bool:
    """已完成且格式正确的 run json → 跳过重跑。"""
    if not path.exists():
        return False
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return bool(d.get("curve") and d["curve"][-1].get("probes"))
    except Exception:
        return False


def _single(cfg, spec_key: str) -> None:
    """子进程模式：跑一个 (model,dose,seed) 并落盘。spec_key = 'model|dose|seed'。"""
    model_name, dose_s, seed_s = spec_key.split("|")
    dose, seed = float(dose_s), int(seed_s)
    model_entry = next(m for m in cfg["models"] if m["name"] == model_name)

    reg = load_registry()
    filler_texts, train_qs, probe_specs, probe_qs = _load_data(cfg, reg)
    curve = _run_one(
        model_entry, cfg["train"], cfg["codec"],
        train_qs, probe_specs, probe_qs, filler_texts, dose, seed,
    )
    out_dir = _HERE / "outputs"
    out_dir.mkdir(exist_ok=True)
    payload = {"model": model_name, "dose": dose, "seed": seed, "curve": curve}
    _run_path(out_dir, model_name, dose, seed).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _summarize(cfg, out_dir: Path) -> None:
    doses = [float(p) for p in cfg["doses"]]
    seeds = [int(s) for s in cfg["seeds"]]
    all_runs: dict = {}
    for m in cfg["models"]:
        for dose in doses:
            for seed in seeds:
                p = _run_path(out_dir, m["name"], dose, seed)
                if _valid_run(p):
                    all_runs[f"{m['name']}|p={dose}|s={seed}"] = json.loads(
                        p.read_text(encoding="utf-8")
                    )

    summary: dict = {}
    for m in cfg["models"]:
        name = m["name"]
        summary[name] = {}
        for dose in doses:
            runs = [v for v in all_runs.values()
                    if v["model"] == name and v["dose"] == dose and v["curve"]]
            probe_finals: dict = {}
            for probe in cfg["probes"]:
                vals = [r["curve"][-1]["probes"][probe]["signal"] for r in runs
                        if r["curve"][-1]["probes"].get(probe, {}).get("signal") is not None]
                probe_finals[probe] = float(np.mean(vals)) if vals else None
            summary[name][str(dose)] = probe_finals

    (out_dir / "xmodel_summary.json").write_text(
        json.dumps({"config": cfg, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "xmodel_curves.json").write_text(
        json.dumps(all_runs, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== SUMMARY (final CoDeC: model -> dose -> probe) ===")
    print(json.dumps(summary, indent=2))
    print(f"[done] wrote {out_dir / 'xmodel_summary.json'}  ({len(all_runs)} runs)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", default=None, help="子进程模式：'model|dose|seed'")
    args = ap.parse_args()
    cfg = yaml.safe_load((_HERE / "run.yaml").read_text(encoding="utf-8"))

    if args.single:
        _single(cfg, args.single)
        return

    # ---- 主控：每个 run 起独立子进程（显存隔离 + 断点续跑）---- #
    out_dir = _HERE / "outputs"
    out_dir.mkdir(exist_ok=True)
    doses = [float(p) for p in cfg["doses"]]
    seeds = [int(s) for s in cfg["seeds"]]

    child_env = dict(os.environ)
    # 抗显存碎片：expandable_segments 让分配器复用碎片段，降 OOM 概率
    child_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    jobs = [(m["name"], dose, seed)
            for m in cfg["models"] for dose in doses for seed in seeds]
    print(f"[plan] {len(jobs)} runs (subprocess-isolated)")
    for i, (name, dose, seed) in enumerate(jobs, 1):
        path = _run_path(out_dir, name, dose, seed)
        if _valid_run(path):
            print(f"[{i}/{len(jobs)}] skip (done): {name} p={dose} s={seed}", flush=True)
            continue
        print(f"[{i}/{len(jobs)}] run: {name} p={dose} s={seed}", flush=True)
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "--single", f"{name}|{dose}|{seed}"],
            env=child_env,
        )
        if proc.returncode != 0:
            print(f"    !! run failed (exit {proc.returncode}); 继续下一个", flush=True)

    _summarize(cfg, out_dir)


if __name__ == "__main__":
    main()
