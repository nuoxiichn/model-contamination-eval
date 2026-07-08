"""Paraphrase Stress Test smoke run。

目的：在真实 GT ckpt 上验证 paraphrase_stress_test 信号方向：
  - clean ckpt：signal ≈ 0（干净模型识别语义，对表面改写鲁棒）
  - {bench}_heavy ckpt：signal > 0（被记忆的 prompt 遭改写破坏，acc 掉）

范围（可通过 --quick 收窄到最小）：
  ckpts: base / clean / gsm8k_heavy / mmlu_heavy
  benches: gsm8k / mmlu-pro   (前者 math_cot, 后者 mc — 覆盖两 dispatch 分支)
  n_samples: 10 (可 --n-samples 覆盖)
  n_paraphrases: 3
  seed: 42

执行（在 8 卡容器内；本机开发机 tokenizer 4.57 与 SFT ckpt tokenizer_config
中 extra_special_tokens=list 冲突，需在容器内跑）：
    HF_ENDPOINT=https://hf-mirror.com \\
    HF_DATASETS_CACHE=/mnt/public/code/chennuoxi/hf_cache \\
    HF_HUB_OFFLINE=1 \\
    PYTHONPATH=src python3 experiments/2026-06-27_sft_contam_gt/run_paraphrase.py \\
        --only-ckpt clean,gsm8k_heavy,mmlu_heavy --only-bench gsm8k,mmlu-pro \\
        --n-samples 10 --n-paraphrases 3

依赖 questions_cache/（已由 run_perm_option 生成）；缺失时会自动 load_questions。

期待信号：
    clean × gsm8k       signal ≈ 0    (干净模型语义鲁棒)
    gsm8k_heavy × gsm8k signal > 0.15 (改写破坏记忆的 prompt token 序列)
    mmlu_heavy × mmlu-pro signal > 0.15
    交叉污染（gsm8k_heavy × mmlu-pro / mmlu_heavy × gsm8k）应接近 0
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path

from model_contamination.benchmarks import load_registry
from model_contamination.models.hf_local import HFLocalModel
from model_contamination.shared.paraphrase_stress import paraphrase_stress_test
from model_contamination.types import BenchmarkQuestion

HERE = Path(__file__).parent
REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = HERE / "questions_cache"
MANIFESTS = HERE / "manifests"

# fixed smoke set —— 保持轻量
CKPTS = [
    {"name": "base",           "stage_tag": "base",
     "path": "/mnt/public/model/Qwen/Qwen3-1.7B-Base"},
    {"name": "clean",          "stage_tag": "sft",
     "path": "/mnt/public/code/chennuoxi/LlamaFactory/saves/contam/clean_merged"},
    {"name": "gsm8k_light",    "stage_tag": "sft",
     "path": "/mnt/public/code/chennuoxi/LlamaFactory/saves/contam/gsm8k_light_merged"},
    {"name": "gsm8k_medium",   "stage_tag": "sft",
     "path": "/mnt/public/code/chennuoxi/LlamaFactory/saves/contam/gsm8k_medium_merged"},
    {"name": "gsm8k_heavy",    "stage_tag": "sft",
     "path": "/mnt/public/code/chennuoxi/LlamaFactory/saves/contam/gsm8k_heavy_merged"},
    {"name": "mmlu_heavy",     "stage_tag": "sft",
     "path": "/mnt/public/code/chennuoxi/LlamaFactory/saves/contam/mmlu_heavy_merged"},
    {"name": "humaneval_heavy","stage_tag": "sft",
     "path": "/mnt/public/code/chennuoxi/LlamaFactory/saves/contam/humaneval_heavy_merged"},
]

BENCHES = [
    {"name": "gsm8k",    "spec_name": "gsm8k",    "manifest": "manifests/gsm8k.jsonl"},
    {"name": "mmlu-pro", "spec_name": "mmlu-pro", "manifest": "manifests/mmlu-pro.jsonl"},
]


def _json_default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "value"):
        return o.value
    return str(o)


def _load_manifest_indices(manifest_path: Path, limit: int) -> list[int]:
    indices: list[int] = []
    with manifest_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            indices.append(int(json.loads(line)["idx"]))
            if len(indices) >= limit:
                break
    return indices


def _load_questions_for_bench(bench_cfg: dict, registry, n_samples: int):
    spec = registry.get(bench_cfg["spec_name"])
    manifest_path = HERE / bench_cfg["manifest"]
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest 缺失: {manifest_path}")
    indices = _load_manifest_indices(manifest_path, limit=n_samples)

    cache_path = CACHE_DIR / f"{bench_cfg['name']}.jsonl"
    if cache_path.exists():
        questions: list[BenchmarkQuestion] = []
        with cache_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                questions.append(BenchmarkQuestion(**json.loads(line)))
        # 只取所需 indices
        wanted = set(indices)
        selected = [q for q in questions if _q_index(q) in wanted][:n_samples]
        if len(selected) < n_samples:
            # 如果 cache 不足，回退到全部 cache 前 n_samples 条
            selected = questions[:n_samples]
        print(f"  [cache] {bench_cfg['name']}: {len(selected)} qs from {cache_path.name}")
        return spec, selected

    from model_contamination.benchmarks import load_questions
    questions = load_questions(spec, indices=indices)
    return spec, questions[:n_samples]


def _q_index(q: BenchmarkQuestion) -> int:
    """从 BenchmarkQuestion.id 推 index：manifest 里 idx 是原 dataset idx。"""
    # id 形如 "gsm8k-1" / "mmlu-pro-9"；取尾巴的数字
    try:
        return int(q.id.rsplit("-", 1)[-1])
    except (ValueError, IndexError):
        return -1


def _load_model(ckpt_cfg: dict, device: str, dtype: str) -> HFLocalModel | None:
    path = Path(ckpt_cfg["path"])
    if not path.exists():
        print(f"[skip] {ckpt_cfg['name']}: path 不存在 {path}")
        return None
    print(f"\n[i] loading ckpt={ckpt_cfg['name']} stage={ckpt_cfg['stage_tag']} path={path}")
    return HFLocalModel(
        model_path=str(path),
        stage_tag=ckpt_cfg["stage_tag"],
        device=device,
        dtype=dtype,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--n-paraphrases", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--only-ckpt", type=str, default=None,
                        help="逗号分隔 ckpt name，默认全跑 4 个")
    parser.add_argument("--only-bench", type=str, default=None,
                        help="逗号分隔 bench name，默认全跑 gsm8k+mmlu-pro")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="默认 outputs/2026-06-27_sft_contam_gt_paraphrase/<ts>/")
    args = parser.parse_args()

    ckpts = CKPTS
    if args.only_ckpt:
        wanted = set(args.only_ckpt.split(","))
        ckpts = [c for c in ckpts if c["name"] in wanted]
    benches = BENCHES
    if args.only_bench:
        wanted = set(args.only_bench.split(","))
        benches = [b for b in benches if b["name"] in wanted]

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = (
            REPO_ROOT / "outputs" / "2026-06-27_sft_contam_gt_paraphrase"
            / time.strftime("%Y%m%d-%H%M%S")
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[out] {out_dir}")

    cache_para_dir = out_dir / "paraphrase_text_cache"

    registry = load_registry()

    # 先把 questions 全部加载好
    print("\n[i] loading benchmarks...")
    bench_qs: dict[str, tuple] = {}
    for bench_cfg in benches:
        spec, qs = _load_questions_for_bench(bench_cfg, registry, args.n_samples)
        bench_qs[bench_cfg["name"]] = (spec, qs)

    all_results: dict = {}
    result_path = out_dir / "result.json"

    def _flush_results() -> None:
        """每个 ckpt 跑完就落盘，避免整轮被 kill 丢全部进度（Jul3 教训）。"""
        result_path.write_text(json.dumps({
            "config": vars(args),
            "ckpts": [c["name"] for c in ckpts],
            "benches": [b["name"] for b in benches],
            "results": all_results,
        }, indent=2, default=_json_default))

    for ckpt_cfg in ckpts:
        model = _load_model(ckpt_cfg, args.device, args.dtype)
        if model is None:
            continue
        ckpt_blob: dict = {"stage_tag": ckpt_cfg["stage_tag"], "benches": {}}
        for bench_name, (spec, qs) in bench_qs.items():
            print(f"  [run] {ckpt_cfg['name']} × {bench_name}: n={len(qs)} "
                  f"n_paraphrases={args.n_paraphrases}")
            t0 = time.time()
            r = paraphrase_stress_test(
                model, spec, qs,
                n_paraphrases=args.n_paraphrases,
                paraphraser="rule",
                seed=args.seed,
                min_samples=max(1, min(args.n_samples, 5)),  # smoke 允许小
                cache_dir=cache_para_dir,
            )
            elapsed = time.time() - t0
            ckpt_blob["benches"][bench_name] = {
                "elapsed_seconds": elapsed,
                "result": asdict(r),
            }
            ev = r.evidence or {}
            print(
                f"    signal={r.signal!r} verdict={r.verdict_hint} "
                f"acc_orig={ev.get('acc_original')!r} "
                f"worst={ev.get('acc_worst_case')!r} "
                f"({elapsed:.1f}s)"
            )
        all_results[ckpt_cfg["name"]] = ckpt_blob
        _flush_results()  # 增量落盘
        print(f"  [flush] {ckpt_cfg['name']} → {result_path.name}")
        _free_model(model)

    print(f"\n[done] wrote {result_path}")

    # 简要 signal 摘要
    print("\n=== signal summary ===")
    print(f"{'ckpt':<20}{'bench':<12}{'signal':>10}{'verdict':>14}{'acc_orig':>10}{'worst':>10}")
    for ckpt_name, ckpt_blob in all_results.items():
        for bench_name, bench_blob in ckpt_blob["benches"].items():
            r = bench_blob["result"]
            ev = r.get("evidence") or {}
            sig = r.get("signal")
            print(
                f"{ckpt_name:<20}{bench_name:<12}"
                f"{sig if sig is not None else 'n/a':>10}"
                f"{r.get('verdict_hint',''):>14}"
                f"{ev.get('acc_original', 'n/a'):>10}"
                f"{ev.get('acc_worst_case', 'n/a'):>10}"
            )


if __name__ == "__main__":
    main()
