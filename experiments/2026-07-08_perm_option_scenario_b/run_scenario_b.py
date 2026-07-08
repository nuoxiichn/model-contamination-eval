"""复现 Ni et al. AAAI 2025 Scenario b（Algorithm 2）在 Qwen 上 —— 主 runner。

对每个 (model × benchmark) 跑 option_permutation_test（我们对齐论文 Algorithm 2 的
实现），产出 leak_fraction。矩阵 = {Qwen2.5-1.5B, Qwen2.5-7B} × {c-eval, cmmlu,
cmb, mmlu-cf}。

前置：先在联网开发机跑 prepare_data.py 把 outputs/data/<bench>.jsonl 准备好；
本 runner 在 8 卡机（有 GPU）跑推理，只读本地 JSONL，不联网。

运行（8 卡机）：
    HF_HOME=/mnt/public/code/chennuoxi/hf_cache \
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH=src python3 experiments/2026-07-08_perm_option_scenario_b/run_scenario_b.py

    # 冒烟：单模型 + 单 bench + 截断
    ... run_scenario_b.py --models qwen2.5-1.5b --benchmarks c-eval --limit 20

结果：outputs/results/scenario_b.csv（一行一个 model×bench）+ 每格 full JSON。
resume：已完成的 (model,bench) 若在 CSV 里则跳过（--fresh 强制重跑）。
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

from model_contamination.models.hf_local import HFLocalModel
from model_contamination.stage_base.option_permutation import option_permutation_test
from model_contamination.types import BenchmarkQuestion, BenchmarkSpec, Verdict

HERE = Path(__file__).resolve().parent
CFG = yaml.safe_load((HERE / "config.yaml").read_text())


def _fmt(x, spec: str = ".4f") -> str:
    if x is None:
        return "n/a"
    try:
        return format(float(x), spec)
    except (TypeError, ValueError):
        return "n/a"


def _json_default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "value"):
        return o.value
    return str(o)


def _load_questions(bench: str, data_dir: Path, limit: int | None) -> list[BenchmarkQuestion]:
    """从 prepare_data.py 导出的 JSONL 读题，转 BenchmarkQuestion。"""
    path = data_dir / f"{bench}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{bench}: 缺 {path}；先跑 prepare_data.py 导出 outputs/data/{bench}.jsonl"
        )
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    qs: list[BenchmarkQuestion] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ai = int(r["answer_index"])
            qs.append(
                BenchmarkQuestion(
                    id=str(r["id"]),
                    benchmark=bench,
                    format="multiple_choice",
                    prompt=r["question"],
                    choices=list(r["choices"]),
                    answer=letters[ai] if 0 <= ai < len(letters) else "A",
                    answer_index=ai,
                    raw={},
                )
            )
            if limit is not None and len(qs) >= limit:
                break
    return qs


def _spec_for(bench: str) -> BenchmarkSpec:
    """构造 multiple_choice 的最小 spec（option_permutation 只看 format/name）。"""
    return BenchmarkSpec(
        name=bench, family="chinese-mc", format="multiple_choice", language="zh",
        variants=[], applicable_methods=["perm_option"],
        trustworthiness_default=Verdict.SUSPECT,
        data_source="local", data_id=f"outputs/data/{bench}.jsonl",
    )


def _load_prior(csv_path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if csv_path.exists():
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                done.add((row["model"], row["benchmark"]))
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[m["name"] for m in CFG["models"]])
    ap.add_argument("--benchmarks", nargs="+", default=CFG["benchmarks"])
    ap.add_argument("--limit", type=int, default=None, help="每 bench 题数上限（冒烟用）")
    ap.add_argument("--fresh", action="store_true", help="忽略已有 CSV，全部重跑")
    ap.add_argument(
        "--csv-name", type=str, default="scenario_b.csv",
        help="CSV 文件名。数据并行分片时每片用不同名避免写冲突（per-cell JSON 本就分开）",
    )
    args = ap.parse_args()

    data_dir = HERE / CFG["data_dir"]
    out_dir = HERE / CFG["outputs"]["path"]
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / args.csv_name

    params = CFG["params"]
    prior = set() if args.fresh else _load_prior(csv_path)
    model_cfgs = [m for m in CFG["models"] if m["name"] in args.models]

    # 预加载题目（一次），避免每模型重读
    bench_qs: dict[str, list[BenchmarkQuestion]] = {}
    for bench in args.benchmarks:
        qs = _load_questions(bench, data_dir, args.limit)
        bench_qs[bench] = qs
        print(f"[data] {bench}: n={len(qs)}")

    rows: list[dict] = []
    for mcfg in model_cfgs:
        mname = mcfg["name"]
        pending = [b for b in args.benchmarks if (mname, b) not in prior]
        if not pending:
            print(f"[skip] {mname}: 所有 bench 已在 CSV 中")
            continue
        print(f"[model] loading {mname} ← {mcfg['model_path']} …")
        t_load = time.time()
        model = HFLocalModel(
            model_path=mcfg["model_path"],
            stage_tag=mcfg["stage_tag"],
            dtype=mcfg.get("dtype"),
            name=mname,
            trust_remote_code=mcfg.get("trust_remote_code", False),
            device_map=mcfg.get("device_map"),
        )
        print(f"[model] {mname} loaded in {time.time()-t_load:.1f}s")

        for bench in pending:
            qs = bench_qs[bench]
            spec = _spec_for(bench)
            print(f"  [{mname} × {bench}] running Scenario b (n={len(qs)}) …")
            t0 = time.time()
            r = option_permutation_test(
                model, spec, qs,
                max_permutations=params["max_permutations"],
                min_samples=params["min_samples"],
                outlier_threshold=params["outlier_threshold"],
                seed=params["seed"],
            )
            dt = time.time() - t0
            ev = r.evidence or {}
            print(
                f"    signal(leak_frac)={_fmt(r.signal)}, "
                f"n_q={ev.get('n_questions')}, mean_perms={_fmt(ev.get('mean_perms_per_q'), '.1f')}, "
                f"verdict={r.verdict_hint.value if hasattr(r.verdict_hint,'value') else r.verdict_hint}, "
                f"elapsed={dt:.1f}s" + (f", err={r.error}" if r.error else "")
            )
            # 每格 full JSON
            (out_dir / f"{mname}__{bench}.json").write_text(
                json.dumps(asdict(r), ensure_ascii=False, indent=2, default=_json_default)
            )
            rows.append({
                "model": mname,
                "benchmark": bench,
                "signal_leak_fraction": r.signal,
                "verdict": r.verdict_hint.value if hasattr(r.verdict_hint, "value") else str(r.verdict_hint),
                "n_questions": ev.get("n_questions"),
                "mean_perms_per_q": ev.get("mean_perms_per_q"),
                "primary_threshold": ev.get("primary_threshold"),
                "leak_fraction_by_threshold": json.dumps(ev.get("leak_fraction_by_threshold"), ensure_ascii=False),
                "prerequisites_met": r.prerequisites_met,
                "elapsed_seconds": round(dt, 1),
                "error": r.error,
            })

        del model
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    _append_csv(csv_path, rows)
    print(f"[i] wrote {csv_path}  (+{len(rows)} rows)")


def _append_csv(csv_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    existing: list[dict] = []
    if csv_path.exists():
        with csv_path.open() as f:
            existing = list(csv.DictReader(f))
    all_rows = existing + rows
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k) for k in fieldnames})


if __name__ == "__main__":
    main()
