"""ΔMIA：Qwen3-1.7B-Base vs Qwen3-1.7B（官方同源 SFT/Instruct）。

同一 target (gsm8k) + control (math-500)，比较两个 checkpoint 的 Min-K%++ AUC。
预期：若 SFT 阶段引入 GSM8K 相关污染，AUC_sft > AUC_base（ΔAUC > 0）。

执行：
    HF_ENDPOINT=https://hf-mirror.com \\
    HF_DATASETS_CACHE=/mnt/public/code/chennuoxi/hf_cache \\
    PYTHONPATH=src python3 experiments/2026-06-22_qwen3-1.7b_delta_mia/run.py
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import yaml

from model_contamination.benchmarks import load_questions, load_registry
from model_contamination.models.hf_local import HFLocalModel
from model_contamination.stage_base.min_k_plus_plus import mink_plus_plus

HERE = Path(__file__).parent
CFG = yaml.safe_load((HERE / "run.yaml").read_text())
OUT_DIR = Path(__file__).resolve().parents[2] / CFG["outputs"]["path"] / time.strftime("%Y%m%d-%H%M%S")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _json_default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "value"):
        return o.value
    return str(o)


def _run_one(stage_key: str, qs_target, qs_control, spec_target):
    cfg = CFG["models"][stage_key]
    print(f"\n[i] loading {stage_key} model {cfg['path']}…")
    model = HFLocalModel(
        model_path=cfg["path"],
        stage_tag=cfg["stage_tag"],
        device=CFG["params"]["device"],
        dtype=CFG["params"]["dtype"],
    )
    print(f"[i] n_layers={model.n_layers} max_pos={model._max_position}")
    t0 = time.time()
    result = mink_plus_plus(
        model, spec_target, qs_target,
        k_ratio=CFG["params"]["k_ratio"],
        control_questions=qs_control,
        min_samples=CFG["params"]["min_samples"],
    )
    dt = time.time() - t0
    print(f"[i] {stage_key}: AUC={result.signal!r}, "
          f"target_mean={result.evidence.get('target_mean'):.4f}, "
          f"control_mean={result.evidence.get('control_mean'):.4f}, "
          f"elapsed={dt:.1f}s")
    # 释放显存：把 model 引用断开（不显式 .to('cpu')，靠 Python GC）
    del model
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, dt


def main() -> None:
    reg = load_registry()
    spec_target = reg.get(CFG["datasets"]["target"]["name"])
    spec_control = reg.get(CFG["datasets"]["control"]["name"])

    print(f"[i] target={spec_target.name} limit={CFG['params']['limit_target']}")
    qs_target = load_questions(spec_target, limit=CFG["params"]["limit_target"])
    print(f"[i] control={spec_control.name} limit={CFG['params']['limit_control']}")
    qs_control = load_questions(spec_control, limit=CFG["params"]["limit_control"])
    print(f"[i] loaded target={len(qs_target)} control={len(qs_control)}")

    results = {}
    for stage_key in ("base", "sft"):
        r, dt = _run_one(stage_key, qs_target, qs_control, spec_target)
        results[stage_key] = {"result": asdict(r), "elapsed_seconds": dt}

    auc_base = results["base"]["result"]["signal"]
    auc_sft = results["sft"]["result"]["signal"]
    delta_auc = (auc_sft - auc_base) if (auc_base is not None and auc_sft is not None) else None

    summary = {
        "config": CFG,
        "auc_base": auc_base,
        "auc_sft": auc_sft,
        "delta_auc": delta_auc,
        "results": results,
    }
    out_path = OUT_DIR / "result.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default))
    print(f"\n[i] wrote {out_path}")
    print(f"  AUC_base = {auc_base!r}")
    print(f"  AUC_sft  = {auc_sft!r}")
    print(f"  ΔAUC     = {delta_auc!r}")


if __name__ == "__main__":
    main()
