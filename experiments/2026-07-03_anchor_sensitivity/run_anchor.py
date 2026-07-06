"""Anchor 灵敏度 runner（E1.1.1）：单 anchor × N benchmark × M 方法。

用开源已知污染模型（anchor）验证每个检测方法的 TPR/FPR，产出灵敏度矩阵
（实验计划-v2.md §E1.1）。与自造 SFT 污染 ckpt 的三个 runner 不同：

- anchor 是**单一开源 base 模型**，不走 SFT 注入 manifest，直接从 benchmark
  全量取题（`load_questions(spec, limit=n)`），oren 需 n≥200
- AUC 类方法（mink_plus_plus）需 clean 对照 benchmark（BENCH_CONTROL 映射）
- SPV-MIA 默认不跑：anchor 无「同源未污染」reference（属 E1.2 专项）；仅当
  显式传 --reference-model 时才跑

执行（8 卡容器内，单 anchor 全套）：
    HF_ENDPOINT=https://hf-mirror.com \\
    HF_DATASETS_CACHE=/mnt/public/code/chennuoxi/hf_cache \\
    PYTHONPATH=src python3 experiments/2026-07-03_anchor_sensitivity/run_anchor.py \\
        --model A1 --bench gsm8k,mmlu-pro --n-samples 200

单点调用（E1.1.1 验收）：
    ... run_anchor.py --model /path/to/model --bench gsm8k --methods paraphrase

续跑（复用时间戳目录，跳过已完成 (bench, method) cell）：
    ... run_anchor.py --model A1 --bench gsm8k --resume outputs/.../A1/20260703-xxxxxx
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path

from model_contamination.benchmarks import load_questions, load_registry
from model_contamination.models.hf_local import HFLocalModel
from model_contamination.shared.oren_permutation import oren_permutation_test
from model_contamination.shared.paraphrase_stress import paraphrase_stress_test
from model_contamination.stage_base.min_k_plus_plus import mink_plus_plus
from model_contamination.stage_base.option_permutation import option_permutation_test
from model_contamination.stage_sft.spv_mia import spv_mia
from model_contamination.types import BenchmarkQuestion, DetectionResult, Stage, Verdict

HERE = Path(__file__).parent
REPO_ROOT = Path(__file__).resolve().parents[2]

# ── anchor 池（docs/positive_controls.md）。A1 本地已有；A2/A3/C1/C2 后台下载中。
ANCHORS: dict[str, dict] = {
    "A1": {"path": "/mnt/public/model/huggingface/Qwen3-1.7B-Base",
           "stage_tag": "base", "note": "Qwen3-1.7B-Base，本项目已复现 GSM8K 污染"},
    "A2": {"path": "/mnt/public/code/chennuoxi/hf_cache/models/A2_mistral-7b-v0.1",
           "stage_tag": "base", "note": "Mistral-7B-v0.1，ConStat/GSM1k 点名"},
    "A3": {"path": "/mnt/public/code/chennuoxi/hf_cache/models/A3_phi-3-mini-4k",
           "stage_tag": "base", "note": "Phi-3-mini-4k，需 trust_remote_code"},
    "C1": {"path": "/mnt/public/code/chennuoxi/hf_cache/models/C1_olmo-2-7b",
           "stage_tag": "base", "note": "OLMo-2-7B 阴性对照（Dolma 全公开）"},
    "C2": {"path": "/mnt/public/code/chennuoxi/hf_cache/models/C2_pythia-2.8b",
           "stage_tag": "base", "note": "Pythia-2.8B 阴性对照（Pile 早于 benchmark）"},
}

# ── bench → clean 对照 bench（AUC 类方法用）。控制集不可用时降级 mean_only。
BENCH_CONTROL: dict[str, str] = {
    "gsm8k": "gsm1k",
    "mmlu-pro": "mmlu-cf",
    "math": "gsm1k",
}

# ── 部分对照 benchmark 没有 test split（如 MMLU-CF 仅 val/dev）。覆盖默认 "test"。
BENCH_SPLIT: dict[str, str] = {
    "mmlu-cf": "val",
}

DEFAULT_METHODS = ["oren", "paraphrase", "mink_plus_plus", "perm_option"]
ALL_METHODS = DEFAULT_METHODS + ["spv_mia"]  # self_critique 仍是 Phase 3 stub

OUT_ROOT = REPO_ROOT / "outputs" / "2026-07-03_anchor_sensitivity"


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


def _free_model(model) -> None:
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _resolve_model(model_arg: str) -> tuple[str, dict]:
    """--model 既接受 anchor id（A1..）又接受任意 fs 路径。返回 (anchor_id, cfg)。"""
    if model_arg in ANCHORS:
        return model_arg, dict(ANCHORS[model_arg])
    path = Path(model_arg)
    anchor_id = path.name or "custom"
    return anchor_id, {"path": model_arg, "stage_tag": "base", "note": "custom path"}


def _load_questions_cached(
    spec, n_samples: int, qcache_dir: Path, split: str = "test",
) -> list[BenchmarkQuestion]:
    """从 benchmark 全量取前 n 题，缓存到 _qcache 复用。split 覆盖默认 test。"""
    qcache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = qcache_dir / f"{spec.name}_{split}_n{n_samples}.jsonl"
    if cache_path.exists():
        qs = [BenchmarkQuestion(**json.loads(ln))
              for ln in cache_path.read_text().splitlines() if ln.strip()]
        print(f"  [qcache] {spec.name}: {len(qs)} from {cache_path.name}")
        return qs
    qs = load_questions(spec, split=split, limit=n_samples)
    with cache_path.open("w") as f:
        for q in qs:
            f.write(json.dumps(asdict(q), ensure_ascii=False, default=_json_default) + "\n")
    print(f"  [load] {spec.name}: {len(qs)} → cached {cache_path.name}")
    return qs


def _load_control(bench_name: str, registry, n_samples: int, qcache_dir: Path):
    """取对照集；缺失（gsm1k 仅 50 条 / mmlu-cf 需 val split）时返回 None，AUC 降级 mean_only。"""
    ctrl_name = BENCH_CONTROL.get(bench_name)
    if not ctrl_name:
        return None
    try:
        ctrl_spec = registry.get(ctrl_name)
        ctrl_split = BENCH_SPLIT.get(ctrl_name, "test")
        ctrl_qs = _load_questions_cached(ctrl_spec, n_samples, qcache_dir, split=ctrl_split)
        if not ctrl_qs:
            raise RuntimeError("空对照集")
        print(f"  [control] {bench_name} ← {ctrl_name}(split={ctrl_split}): n={len(ctrl_qs)}")
        return ctrl_qs
    except Exception as e:  # noqa: BLE001 — 控制集缺失不该炸主流程
        print(f"  [control unavailable] {bench_name} ← {ctrl_name}: {e}；AUC 类方法降级 mean_only")
        return None


def _dispatch(
    method: str, model, spec, qs, control_qs, reference_model, seed: int, n_paraphrases: int,
    para_cache_dir: Path,
) -> DetectionResult:
    """跑单个方法，返回 DetectionResult。format/reference 不满足时返回 prereq 未满足，不抛。"""
    if method == "oren":
        return oren_permutation_test(model, spec, qs, seed=seed)
    if method == "paraphrase":
        return paraphrase_stress_test(
            model, spec, qs, n_paraphrases=n_paraphrases, paraphraser="rule",
            seed=seed, cache_dir=para_cache_dir,
        )
    if method == "mink_plus_plus":
        return mink_plus_plus(model, spec, qs, control_questions=control_qs)
    if method == "perm_option":
        if spec.format != "multiple_choice":
            return DetectionResult(
                method="perm_option", stage=Stage(model.stage_tag), benchmark=spec.name,
                signal=None, verdict_hint=Verdict.INCONCLUSIVE, prerequisites_met=False,
                error=f"perm_option 仅适用 multiple_choice，spec.format={spec.format}",
            )
        return option_permutation_test(model, spec, qs, seed=seed)
    if method == "spv_mia":
        if reference_model is None:
            return DetectionResult(
                method="spv_mia", stage=Stage(model.stage_tag), benchmark=spec.name,
                signal=None, verdict_hint=Verdict.INCONCLUSIVE, prerequisites_met=False,
                error="no reference model；anchor 无同源未污染对照，SPV-MIA 属 E1.2 专项",
            )
        return spv_mia(
            model, reference_model, spec, qs, control_questions=control_qs, seed=seed,
        )
    raise ValueError(f"未知方法: {method}")


def _write_csv(all_results: dict, csv_path: Path, anchor_id: str, stage_tag: str) -> None:
    rows: list[dict] = []
    for bench_name, methods_blob in all_results.items():
        for method_name, cell in methods_blob.items():
            r = cell["result"]
            ev = r.get("evidence") or {}
            rows.append({
                "anchor": anchor_id,
                "stage": r.get("stage", stage_tag),
                "benchmark": bench_name,
                "method": method_name,
                "signal": r.get("signal"),
                "verdict_hint": r.get("verdict_hint"),
                "prerequisites_met": r.get("prerequisites_met"),
                "mode": ev.get("mode"),
                "elapsed_seconds": cell.get("elapsed_seconds"),
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
    parser = argparse.ArgumentParser(description="Anchor 灵敏度 runner（E1.1.1）")
    parser.add_argument("--model", required=True,
                        help="anchor id（A1/A2/A3/C1/C2）或模型 fs 路径")
    parser.add_argument("--stage-tag", default=None,
                        help="覆盖 stage tag，默认取 anchor 池的 base")
    parser.add_argument("--bench", default="gsm8k,mmlu-pro",
                        help="逗号分隔 bench name（如 gsm8k,mmlu-pro）")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS),
                        help=f"逗号分隔方法名或 all；默认 {DEFAULT_METHODS}（all 含 spv_mia）")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="每 bench 取题数；oren 需 ≥200")
    parser.add_argument("--n-paraphrases", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Phi-3 等需要")
    parser.add_argument("--reference-model", default=None,
                        help="SPV-MIA 的 reference（base）模型路径；不传则 spv_mia 跳过")
    parser.add_argument("--resume", default=None,
                        help="已有输出目录（含 result.json），跳过已完成 (bench, method) cell")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    anchor_id, mcfg = _resolve_model(args.model)
    stage_tag = args.stage_tag or mcfg["stage_tag"]
    benches = [b.strip() for b in args.bench.split(",") if b.strip()]
    methods = ALL_METHODS if args.methods == "all" else \
        [m.strip() for m in args.methods.split(",") if m.strip()]

    # ── 输出目录 / resume
    if args.resume:
        out_dir = Path(args.resume)
        if not out_dir.is_absolute():
            out_dir = (REPO_ROOT / out_dir).resolve()
        if not out_dir.exists():
            raise FileNotFoundError(f"--resume 目录不存在: {out_dir}")
        prior_blob = {}
        rp = out_dir / "result.json"
        if rp.exists():
            prior_blob = json.loads(rp.read_text())
        all_results: dict[str, dict] = prior_blob.get("results", {}) or {}
        completed = {(b, m) for b, mb in all_results.items() for m in mb}
        print(f"[resume] out_dir={out_dir}，已完成 {len(completed)} cell")
    else:
        out_dir = Path(args.out_dir) if args.out_dir else \
            OUT_ROOT / anchor_id / time.strftime("%Y%m%d-%H%M%S")
        all_results = {}
        completed = set()
        print(f"[fresh] out_dir={out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    qcache_dir = OUT_ROOT / "_qcache"

    registry = load_registry()

    # ── 预取所有 bench 的题目 + 对照集
    print(f"[i] anchor={anchor_id} stage={stage_tag} benches={benches} methods={methods}")
    bench_qs: dict[str, tuple] = {}
    for b in benches:
        spec = registry.get(b)
        qs = _load_questions_cached(spec, args.n_samples, qcache_dir)
        need_control = any(m in ("mink_plus_plus", "spv_mia") for m in methods)
        ctrl = _load_control(b, registry, args.n_samples, qcache_dir) if need_control else None
        bench_qs[b] = (spec, qs, ctrl)

    # ── 载入模型（放在取题之后，取题失败不浪费显存）
    model_path = Path(mcfg["path"])
    if not model_path.exists():
        raise FileNotFoundError(
            f"模型路径不存在: {model_path}（anchor={anchor_id}；权重可能还在下载）"
        )
    print(f"[i] loading target={anchor_id} path={model_path}")
    model = HFLocalModel(
        model_path=str(model_path), stage_tag=stage_tag,
        device=args.device, dtype=args.dtype, name=anchor_id,
        trust_remote_code=args.trust_remote_code,
    )

    reference_model = None
    if args.reference_model and "spv_mia" in methods:
        ref_path = Path(args.reference_model)
        if not ref_path.exists():
            raise FileNotFoundError(f"--reference-model 路径不存在: {ref_path}")
        print(f"[i] loading reference path={ref_path}")
        reference_model = HFLocalModel(
            model_path=str(ref_path), stage_tag="base",
            device=args.device, dtype=args.dtype, name=f"{anchor_id}_ref",
            trust_remote_code=args.trust_remote_code,
        )

    para_cache_dir = out_dir / "paraphrase_cache"

    # ── 主循环：(bench, method) cell
    for b in benches:
        spec, qs, ctrl = bench_qs[b]
        methods_blob = all_results.setdefault(b, {})
        for method in methods:
            if (b, method) in completed:
                print(f"[skip] {b} × {method}: 已完成")
                continue
            print(f"  [{anchor_id} × {b} × {method}] running…")
            t0 = time.time()
            try:
                r = _dispatch(
                    method, model, spec, qs, ctrl, reference_model,
                    args.seed, args.n_paraphrases, para_cache_dir,
                )
                r_dict = asdict(r)
                ev = r.evidence or {}
                print(
                    f"    [{method}] signal={_fmt(r.signal)}, mode={ev.get('mode', '-')}, "
                    f"verdict={r.verdict_hint!r}, prereq={r.prerequisites_met}, "
                    f"elapsed={time.time() - t0:.1f}s"
                    + (f", err={r.error}" if r.error else "")
                )
            except Exception as e:  # noqa: BLE001 — 单方法炸不该拖垮整轮
                r_dict = {
                    "method": method, "stage": stage_tag, "benchmark": b,
                    "signal": None, "verdict_hint": "inconclusive",
                    "prerequisites_met": False, "evidence": {},
                    "error": f"{type(e).__name__}: {e}",
                }
                print(f"    [{method}] EXCEPTION: {type(e).__name__}: {e}")
            methods_blob[method] = {
                "result": r_dict, "elapsed_seconds": time.time() - t0,
            }
            # 每 cell 增量落盘
            summary = {
                "config": vars(args) | {"anchor_id": anchor_id, "stage_tag": stage_tag},
                "results": all_results,
            }
            (out_dir / "result.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default)
            )

    _free_model(model)
    if reference_model is not None:
        _free_model(reference_model)

    _write_csv(all_results, out_dir / "anchor_sensitivity.csv", anchor_id, stage_tag)
    print(f"\n[done] wrote {out_dir / 'result.json'}")
    print(f"[done] wrote {out_dir / 'anchor_sensitivity.csv'}")


if __name__ == "__main__":
    main()
