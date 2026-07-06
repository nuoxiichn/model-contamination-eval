"""把 calibration 用到的 3 个 benchmark 按 manifest indices 材料化到本地 JSONL。

为何要这步：8 卡机容器里的 `datasets` 版本和 HF 数据集 features 元数据不兼容
（symptom: dataclasses.fields() TypeError on generate_from_dict）。开发机
这边 datasets 5.0.0 加载正常，先 dump 一份，calibration 直接读 JSONL，
容器里跑时根本不导入 datasets。

跑法（开发机）：
    HF_ENDPOINT=https://hf-mirror.com \\
    HF_DATASETS_CACHE=/mnt/public/code/chennuoxi/hf_cache \\
    PYTHONPATH=src python3 experiments/2026-06-27_sft_contam_gt/dump_questions.py

产出：experiments/2026-06-27_sft_contam_gt/questions_cache/{bench_name}.jsonl
每行一个 BenchmarkQuestion 的 dict（json.dumps(asdict(q))）。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import yaml

from model_contamination.benchmarks import load_questions, load_registry

HERE = Path(__file__).parent
CFG = yaml.safe_load((HERE / "calibration.yaml").read_text())
CACHE_DIR = HERE / "questions_cache"
CACHE_DIR.mkdir(exist_ok=True)


def _load_manifest_indices(manifest_path: Path) -> list[int]:
    indices: list[int] = []
    with manifest_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            indices.append(int(json.loads(line)["idx"]))
    return indices


def main() -> None:
    registry = load_registry()
    for bench_cfg in CFG["benchmarks"]:
        bench_name = bench_cfg["name"]
        out_path = CACHE_DIR / f"{bench_name}.jsonl"
        spec = registry.get(bench_cfg["spec_name"])
        indices = _load_manifest_indices(HERE / bench_cfg["manifest"])
        print(f"[{bench_name}] loading {len(indices)} questions via load_questions(spec={spec.name})…")
        questions = load_questions(spec, indices=indices)
        with out_path.open("w") as f:
            for q in questions:
                f.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")
        print(f"  wrote {len(questions)} → {out_path}")
    print(f"\n[i] cache ready at {CACHE_DIR}")


if __name__ == "__main__":
    main()
