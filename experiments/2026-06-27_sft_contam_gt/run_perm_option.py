"""perm_option calibration 评测：7 ckpt × mmlu-pro × perm_option。

补救 calibration v3 中 SPV-MIA 在 MMLU-Pro 上 7/7 ckpt 全部 degraded
（[[calibration-findings-2026-06-29]] Finding 9 / notes.md §U.5）。perm_option
仅适用 multiple_choice，所以只跑 mmlu-pro 一个 bench。

复用 run_calibration.py 的 questions cache / model loader helpers，
独立成脚本因为 method 维度只有 perm_option（无 reference model，bench 维度只有 1）。

执行（8 卡容器内 fresh 跑）：
    HF_ENDPOINT=https://hf-mirror.com \\
    HF_DATASETS_CACHE=/mnt/public/code/chennuoxi/hf_cache \\
    PYTHONPATH=src python3 experiments/2026-06-27_sft_contam_gt/run_perm_option.py

续跑（复用已有时间戳目录，跳过已完成 ckpt）：
    ... python3 run_perm_option.py --resume outputs/2026-06-27_sft_contam_gt_perm_option/20260629-202724
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path

import yaml

from model_contamination.benchmarks import load_registry
from model_contamination.models.hf_local import HFLocalModel
from model_contamination.stage_base.option_permutation import option_permutation_test
from model_contamination.types import BenchmarkQuestion

HERE = Path(__file__).parent
CFG = yaml.safe_load((HERE / "calibration_perm_option.yaml").read_text())
REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = HERE / "questions_cache"


def _json_default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "value"):
        return o.value
    return str(o)


def _fmt(x, spec: str = ".4f") -> str:
    if x is None:
        return "n/a"
    try:
        return format(float(x), spec)
    except (TypeError, ValueError):
        return "n/a"


def _load_manifest_indices(manifest_path: Path) -> list[int]:
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
                f"{bench_cfg['name']}: cache 数量 {len(questions)} ≠ manifest {len(indices)}"
            )
        print(
            f"  [cache] {bench_cfg['name']}: loaded {len(questions)} from {cache_path.name}"
        )
        return spec, questions

    from model_contamination.benchmarks import load_questions
    questions = load_questions(spec, indices=indices)
    return spec, questions


def _try_load_model(ckpt_cfg: dict) -> HFLocalModel | None:
    path = Path(ckpt_cfg["path"])
    if not path.exists() and CFG.get("skip_missing_checkpoints", True):
        print(f"[skip] {ckpt_cfg['name']}: path 不存在 {path}")
        return None
    print(
        f"\n[i] loading checkpoint={ckpt_cfg['name']} "
        f"stage={ckpt_cfg['stage_tag']} path={path}"
    )
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
    rows: list[dict] = []
    for ckpt_name, ckpt_blob in all_results.items():
        for bench_name, bench_blob in ckpt_blob["benches"].items():
            r = bench_blob["result"]
            ev = r.get("evidence") or {}
            rows.append({
                "method": r.get("method"),
                "benchmark": bench_name,
                "checkpoint": ckpt_name,
                "stage": r.get("stage"),
                "signal": r.get("signal"),
                "verdict_hint": r.get("verdict_hint"),
                "prerequisites_met": r.get("prerequisites_met"),
                "n_questions": ev.get("n_questions"),
                "max_permutations": ev.get("max_permutations"),
                "mean_perms_per_q": ev.get("mean_perms_per_q"),
                "primary_threshold": ev.get("primary_threshold"),
                "leak_fraction": ev.get("leak_fraction"),
                "leak_score": ev.get("leak_score"),
                "elapsed_seconds": bench_blob.get("elapsed_seconds"),
                "error": r.get("error"),
            })
    if not rows:
        csv_path.write_text("")
        return
    with csv_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "已有输出目录（含 result.json），跳过其中已完成 ckpt 并追加新结果到同目录；"
            "不传则用当前时间戳新建目录 fresh 跑全 7 ckpt。"
        ),
    )
    args = parser.parse_args()

    if args.resume:
        out_dir = Path(args.resume)
        if not out_dir.is_absolute():
            out_dir = (REPO_ROOT / out_dir).resolve()
        if not out_dir.exists():
            raise FileNotFoundError(f"--resume 指向的目录不存在: {out_dir}")
        result_path = out_dir / "result.json"
        prior_results: dict = {}
        if result_path.exists():
            prior_blob = json.loads(result_path.read_text())
            prior_results = prior_blob.get("results", {}) or {}
        completed_ckpts = set(prior_results.keys())
        print(f"[resume] out_dir={out_dir}")
        print(f"[resume] 已完成 {len(completed_ckpts)} 个 ckpt: {sorted(completed_ckpts)}")
    else:
        out_dir = REPO_ROOT / CFG["outputs"]["path"] / time.strftime("%Y%m%d-%H%M%S")
        prior_results = {}
        completed_ckpts = set()
        print(f"[fresh] out_dir={out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    registry = load_registry()

    print("[i] loading benchmarks via manifests…")
    bench_data: dict[str, tuple] = {}
    for bench_cfg in CFG["benchmarks"]:
        spec, questions = _load_questions_for_bench(bench_cfg, registry)
        bench_data[bench_cfg["name"]] = (spec, questions)
        print(f"  {bench_cfg['name']}: n={len(questions)} spec={spec.name}")

    perm_params = CFG["params"]["perm_option"]

    all_results: dict[str, dict] = dict(prior_results)
    for ckpt_cfg in CFG["checkpoints"]:
        if ckpt_cfg["name"] in completed_ckpts:
            print(f"[skip] {ckpt_cfg['name']}: 已在 --resume 目录中完成")
            continue
        model = _try_load_model(ckpt_cfg)
        if model is None:
            continue
        ckpt_results: dict[str, dict] = {}
        for bench_cfg in CFG["benchmarks"]:
            bench_name = bench_cfg["name"]
            spec, questions = bench_data[bench_name]
            print(f"  [{ckpt_cfg['name']} × {bench_name}] running perm_option…")
            t0 = time.time()
            r = option_permutation_test(
                model, spec, questions,
                max_permutations=perm_params["max_permutations"],
                min_samples=perm_params["min_samples"],
                seed=perm_params["seed"],
            )
            dt = time.time() - t0
            ev = r.evidence or {}
            print(
                f"    [perm_option] signal={_fmt(r.signal)}, "
                f"leak_frac={_fmt(ev.get('leak_fraction'))}, "
                f"mean_perms={_fmt(ev.get('mean_perms_per_q'))}, "
                f"n_q={ev.get('n_questions')}, "
                f"verdict={r.verdict_hint!r}, elapsed={dt:.1f}s"
                + (f", err={r.error}" if r.error else "")
            )
            ckpt_results[bench_name] = {
                "result": asdict(r),
                "elapsed_seconds": dt,
            }
        all_results[ckpt_cfg["name"]] = {
            "stage_tag": ckpt_cfg["stage_tag"],
            "path": ckpt_cfg["path"],
            "benches": ckpt_results,
        }
        _free_model(model)

        # 增量落盘
        summary = {"config": CFG, "results": all_results}
        (out_dir / "result.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default)
        )

    _write_csv(all_results, out_dir / "perm_option.csv")
    print(f"\n[i] wrote {out_dir / 'result.json'}")
    print(f"[i] wrote {out_dir / 'perm_option.csv'}")


if __name__ == "__main__":
    main()
