"""真模型 Min-K%++ smoke：Qwen3-1.7B-Base on GSM8K (target) vs MATH-500 (control)。

直接执行：
    HF_ENDPOINT=https://hf-mirror.com \\
    HF_DATASETS_CACHE=/mnt/public/code/chennuoxi/hf_cache \\
    PYTHONPATH=src python3 experiments/2026-06-22_qwen3-1.7b-base_minkpp/run.py
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
RUN_CFG = yaml.safe_load((HERE / "run.yaml").read_text())
OUT_DIR = Path(__file__).resolve().parents[2] / RUN_CFG["outputs"]["path"] / time.strftime("%Y%m%d-%H%M%S")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _json_default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "value"):
        return o.value
    return str(o)


def main() -> None:
    reg = load_registry()
    spec_target = reg.get(RUN_CFG["datasets"]["target"]["name"])
    spec_control = reg.get(RUN_CFG["datasets"]["control"]["name"])

    print(f"[i] loading target  {spec_target.name} (limit={RUN_CFG['params']['limit_target']})…")
    qs_target = load_questions(spec_target, limit=RUN_CFG["params"]["limit_target"])
    print(f"[i] loading control {spec_control.name} (limit={RUN_CFG['params']['limit_control']})…")
    qs_control = load_questions(spec_control, limit=RUN_CFG["params"]["limit_control"])
    print(f"[i] target={len(qs_target)} control={len(qs_control)}")

    print(f"[i] loading model {RUN_CFG['model']}…")
    model = HFLocalModel(
        model_path=RUN_CFG["model"],
        stage_tag=RUN_CFG["stage"],
        device=RUN_CFG["params"]["device"],
        dtype=RUN_CFG["params"]["dtype"],
    )
    print(f"[i] n_layers={model.n_layers} max_pos={model._max_position}")

    print("[i] running Min-K%++ with control…")
    t0 = time.time()
    result = mink_plus_plus(
        model, spec_target, qs_target,
        k_ratio=RUN_CFG["params"]["k_ratio"],
        control_questions=qs_control,
        min_samples=RUN_CFG["params"]["min_samples"],
    )
    dt = time.time() - t0
    print(f"[i] elapsed: {dt:.1f}s")

    payload = {
        "config": RUN_CFG,
        "elapsed_seconds": dt,
        "result": asdict(result),
    }
    out_path = OUT_DIR / "result.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))
    print(f"\n[i] wrote {out_path}")
    print(f"  signal (AUC): {result.signal!r}")
    print(f"  verdict_hint: {result.verdict_hint.value}")
    print(f"  prerequisites_met: {result.prerequisites_met}")
    if result.evidence:
        print(f"  delta_mean: {result.evidence.get('delta_mean')!r}")
        print(f"  target_mean: {result.evidence.get('target_mean')!r}")
        print(f"  control_mean: {result.evidence.get('control_mean')!r}")
        print(f"  n_target / n_control: {result.evidence.get('n_target')} / {result.evidence.get('n_control')}")


if __name__ == "__main__":
    main()
