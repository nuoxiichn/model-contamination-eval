"""SFT 污染 GT calibration 评测：7 ckpt × 3 bench × {mink_plus_plus, spv_mia}。

按 plan.md §5 落两份产出：
- outputs/.../{ts}/result.json    全量 DetectionResult
- outputs/.../{ts}/calibration.csv plan §5 表格

SPV-MIA reference 模型：base ckpt（LoRA 场景天然 reference）。
  目标加载循环外预加载 reference 一次，整个 run 保持在 GPU 显存里；
  target 是 base 自己时跳过 spv_mia（target==reference 无意义）。

执行：
    HF_ENDPOINT=https://hf-mirror.com \\
    HF_DATASETS_CACHE=/mnt/public/code/chennuoxi/hf_cache \\
    PYTHONPATH=src python3 experiments/2026-06-27_sft_contam_gt/run_calibration.py
"""

from __future__ import annotations

import csv
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path

import yaml

from model_contamination.benchmarks import load_registry
from model_contamination.models.hf_local import HFLocalModel
from model_contamination.stage_base.min_k_plus_plus import mink_plus_plus
from model_contamination.stage_sft.spv_mia import spv_mia
from model_contamination.types import BenchmarkQuestion

HERE = Path(__file__).parent
CFG = yaml.safe_load((HERE / "calibration.yaml").read_text())
OUT_DIR = Path(__file__).resolve().parents[2] / CFG["outputs"]["path"] / time.strftime("%Y%m%d-%H%M%S")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = HERE / "questions_cache"


def _json_default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "value"):
        return o.value
    return str(o)


def _fmt(x, spec: str = ".4f") -> str:
    """None / NaN-safe 浮点格式化。早 return / 退化路径下 evidence 字段会缺，
    打印不该因此 crash —— 直接显示 'n/a' 给运行日志做信号。"""
    if x is None:
        return "n/a"
    try:
        return format(float(x), spec)
    except (TypeError, ValueError):
        return "n/a"


def _load_manifest_indices(manifest_path: Path) -> list[int]:
    """从 manifest jsonl 抽 idx 列（HF dataset 行号）。

    评测必须用与训练相同的题；manifest 在 prep_bench_to_sft.py 写入时
    每行含 {"bench": ..., "idx": ..., "q_sha1": ..., ...}。
    """
    indices: list[int] = []
    with manifest_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            indices.append(int(row["idx"]))
    return indices


def _load_questions_for_bench(bench_cfg: dict, registry):
    spec = registry.get(bench_cfg["spec_name"])
    manifest_path = HERE / bench_cfg["manifest"]
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest 缺失: {manifest_path}")
    indices = _load_manifest_indices(manifest_path)

    # 优先读本地 cache（dump_questions.py 在开发机预生成），避开 8 卡机容器里
    # datasets 旧版与 HF 元数据不兼容的问题。
    cache_path = CACHE_DIR / f"{bench_cfg['name']}.jsonl"
    if cache_path.exists():
        questions: list[BenchmarkQuestion] = []
        with cache_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                questions.append(BenchmarkQuestion(**json.loads(line)))
        if len(questions) != len(indices):
            raise RuntimeError(
                f"{bench_cfg['name']}: cache 数量 {len(questions)} ≠ manifest {len(indices)}，"
                "重跑 dump_questions.py"
            )
        print(f"  [cache] {bench_cfg['name']}: loaded {len(questions)} from {cache_path.name}")
        return spec, questions

    # 兜底：直接调 load_questions（要求容器 datasets 工作）
    from model_contamination.benchmarks import load_questions
    questions = load_questions(spec, indices=indices)
    return spec, questions


def _run_methods(
    model: HFLocalModel,
    reference_model: HFLocalModel | None,
    spec,
    questions,
    params: dict,
    target_stage_tag: str,
) -> dict[str, dict]:
    """对单个 (ckpt, bench) 跑 mink_plus_plus + spv_mia；返回 method -> result dict。

    spv_mia 仅在 reference_model 存在且 target 不是 base 时跑。
    """
    out: dict[str, dict] = {}

    if "mink_plus_plus" in params["methods"]:
        mp = params["mink_plus_plus"]
        t0 = time.time()
        r = mink_plus_plus(
            model, spec, questions,
            k_ratio=mp["k_ratio"],
            control_questions=None,  # mean-only 模式；跨 ckpt 比 mean
            min_samples=mp["min_samples"],
            estimator=mp.get("estimator", "trim_mean"),
            trim_ratio=mp.get("trim_ratio", 0.1),
        )
        dt = time.time() - t0
        ev = r.evidence or {}
        print(
            f"    [mink++] mode={ev.get('mode')}, "
            f"estimator={ev.get('estimator')}, "
            f"target_mean={_fmt(ev.get('target_mean'))} (raw={_fmt(ev.get('target_mean_raw'))}), "
            f"verdict={r.verdict_hint!r}, elapsed={dt:.1f}s"
        )
        out["mink_plus_plus"] = {"result": asdict(r), "elapsed_seconds": dt}

    if "spv_mia" in params["methods"]:
        sp = params["spv_mia"]
        if reference_model is None:
            print("    [spv_mia] skip: reference model not loaded")
        elif sp.get("skip_when_target_is_base", True) and target_stage_tag == "base":
            print("    [spv_mia] skip: target is base (target==reference, no signal)")
        else:
            t0 = time.time()
            r = spv_mia(
                model, reference_model, spec, questions,
                n_neighbors=sp["n_neighbors"],
                mask_ratio=sp["mask_ratio"],
                control_questions=None,  # mean_only 模式；跨 ckpt 比 mean(Δpv)
                min_samples=sp["min_samples"],
                paraphraser=sp["paraphraser"],
                seed=sp["seed"],
            )
            dt = time.time() - t0
            ev = r.evidence or {}
            print(
                f"    [spv_mia] mode={ev.get('mode')}, "
                f"target_mean_dpv={_fmt(ev.get('target_mean_delta_pv'))}, "
                f"n_target={ev.get('n_target')}, "
                f"verdict={r.verdict_hint!r}, elapsed={dt:.1f}s"
                + (f", degraded={ev.get('degraded_reason')}" if ev.get("degraded_reason") else "")
                + (f", err={r.error}" if r.error else "")
            )
            out["spv_mia"] = {"result": asdict(r), "elapsed_seconds": dt}

    return out


def _try_load_model(ckpt_cfg: dict) -> HFLocalModel | None:
    path = Path(ckpt_cfg["path"])
    if not path.exists() and CFG.get("skip_missing_checkpoints", True):
        print(f"[skip] {ckpt_cfg['name']}: path 不存在 {path}")
        return None
    print(f"\n[i] loading checkpoint={ckpt_cfg['name']} stage={ckpt_cfg['stage_tag']} path={path}")
    return HFLocalModel(
        model_path=str(path),
        stage_tag=ckpt_cfg["stage_tag"],
        device=CFG["params"]["device"],
        dtype=CFG["params"]["dtype"],
        name=ckpt_cfg["name"],
    )


def _free_model(model) -> None:
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _write_csv(all_results: dict, csv_path: Path) -> None:
    """落 plan §5 落点表：method, benchmark, ckpt_name, signal, evidence 摘要。"""
    rows: list[dict] = []
    for ckpt_name, ckpt_blob in all_results.items():
        for bench_name, bench_blob in ckpt_blob["benches"].items():
            for method_name, method_blob in bench_blob["methods"].items():
                r = method_blob["result"]
                ev = r.get("evidence") or {}
                rows.append({
                    "method": method_name,
                    "benchmark": bench_name,
                    "checkpoint": ckpt_name,
                    "stage": r.get("stage"),
                    "signal": r.get("signal"),
                    "verdict_hint": r.get("verdict_hint"),
                    "prerequisites_met": r.get("prerequisites_met"),
                    # mink_plus_plus 字段
                    "target_mean": ev.get("target_mean"),
                    "target_mean_raw": ev.get("target_mean_raw"),
                    "target_std": ev.get("target_std"),
                    "estimator": ev.get("estimator"),
                    # spv_mia 字段
                    "target_mean_delta_pv": ev.get("target_mean_delta_pv"),
                    "target_median_delta_pv": ev.get("target_median_delta_pv"),
                    "target_std_delta_pv": ev.get("target_std_delta_pv"),
                    "spv_n_neighbors": ev.get("n_neighbors"),
                    "spv_mask_ratio": ev.get("mask_ratio"),
                    # 共用
                    "n_target": ev.get("n_target"),
                    "degraded_reason": ev.get("degraded_reason"),
                    "error": r.get("error"),
                })
    with csv_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    registry = load_registry()

    # 预加载所有 bench questions（按 manifest），避免每 ckpt 重新读
    print("[i] loading benchmarks via manifests…")
    bench_data: dict[str, tuple] = {}
    for bench_cfg in CFG["benchmarks"]:
        spec, questions = _load_questions_for_bench(bench_cfg, registry)
        bench_data[bench_cfg["name"]] = (spec, questions)
        print(f"  {bench_cfg['name']}: n={len(questions)} spec={spec.name}")

    method_params = {
        "methods": CFG["methods"],
        "mink_plus_plus": CFG["params"]["mink_plus_plus"],
        "spv_mia": CFG["params"].get("spv_mia", {}),
    }

    # 预加载 reference model（base ckpt）一次，整个 run 保持在显存里。
    # SPV-MIA 用它做差分。每个 SFT ckpt 评测期间显存峰值 ≈ 2 × ckpt_size。
    reference_model: HFLocalModel | None = None
    if "spv_mia" in CFG["methods"]:
        base_ckpts = [c for c in CFG["checkpoints"] if c["stage_tag"] == "base"]
        if len(base_ckpts) != 1:
            raise RuntimeError(
                f"spv_mia requires exactly one base ckpt as reference, found {len(base_ckpts)}: "
                f"{[c['name'] for c in base_ckpts]}"
            )
        ref_cfg = base_ckpts[0]
        ref_path = Path(ref_cfg["path"])
        if not ref_path.exists():
            print(f"[warn] reference base path missing: {ref_path}; spv_mia will be skipped")
        else:
            print(f"\n[i] loading reference (base) model={ref_cfg['name']} path={ref_path}")
            reference_model = HFLocalModel(
                model_path=str(ref_path),
                stage_tag=ref_cfg["stage_tag"],
                device=CFG["params"]["device"],
                dtype=CFG["params"]["dtype"],
                name=f"{ref_cfg['name']}_ref",
            )

    all_results: dict[str, dict] = {}
    for ckpt_cfg in CFG["checkpoints"]:
        model = _try_load_model(ckpt_cfg)
        if model is None:
            continue
        ckpt_results: dict[str, dict] = {}
        for bench_cfg in CFG["benchmarks"]:
            bench_name = bench_cfg["name"]
            spec, questions = bench_data[bench_name]
            print(f"  [{ckpt_cfg['name']} × {bench_name}] running methods…")
            method_results = _run_methods(
                model, reference_model, spec, questions, method_params,
                target_stage_tag=ckpt_cfg["stage_tag"],
            )
            ckpt_results[bench_name] = {"methods": method_results}
        all_results[ckpt_cfg["name"]] = {
            "stage_tag": ckpt_cfg["stage_tag"],
            "path": ckpt_cfg["path"],
            "benches": ckpt_results,
        }
        _free_model(model)

        # 增量落盘：每跑完一个 ckpt 就刷一次，避免中断丢全部
        summary = {"config": CFG, "results": all_results}
        (OUT_DIR / "result.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default)
        )

    if reference_model is not None:
        _free_model(reference_model)

    _write_csv(all_results, OUT_DIR / "calibration.csv")
    print(f"\n[i] wrote {OUT_DIR / 'result.json'}")
    print(f"[i] wrote {OUT_DIR / 'calibration.csv'}")


if __name__ == "__main__":
    main()
