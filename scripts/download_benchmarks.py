"""批量下载 benchmark 数据集到本地缓存。

国内环境优先走 hf-mirror，失败 fallback modelscope。
使用：
    python scripts/download_benchmarks.py --benchmarks mmlu gsm8k
    python scripts/download_benchmarks.py --all
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click

from model_contamination.benchmarks import load_registry


@click.command()
@click.option("--benchmarks", multiple=True, help="benchmark name (可多次)")
@click.option("--all", "download_all", is_flag=True, help="下载注册表中所有 HF benchmark")
@click.option("--cache-dir", type=click.Path(), default="./hf_cache")
@click.option(
    "--mirror",
    default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
    help="HF endpoint，默认 hf-mirror.com",
)
def main(benchmarks: tuple[str, ...], download_all: bool, cache_dir: str, mirror: str) -> None:
    os.environ["HF_ENDPOINT"] = mirror
    os.environ["HF_DATASETS_CACHE"] = str(Path(cache_dir).resolve())
    print(f"[i] HF_ENDPOINT = {mirror}")
    print(f"[i] HF_DATASETS_CACHE = {cache_dir}")

    try:
        from datasets import load_dataset
    except ImportError:
        print("[x] need `uv sync --extra hf` first", file=sys.stderr)
        sys.exit(1)

    reg = load_registry()
    if download_all:
        specs = [s for s in reg.all() if s.data_source == "hf"]
    else:
        if not benchmarks:
            print("[x] either --all or --benchmarks required", file=sys.stderr)
            sys.exit(1)
        specs = [reg.get(b) for b in benchmarks]

    failures: list[tuple[str, str]] = []
    for spec in specs:
        if spec.data_source != "hf":
            print(f"[skip] {spec.name} (source={spec.data_source})")
            continue
        if spec.data_id == "TBD":
            print(f"[skip] {spec.name} (data_id=TBD)")
            continue
        subset_msg = f" [{spec.data_subset}]" if spec.data_subset else ""
        print(f"[i] downloading {spec.name} ← {spec.data_id}{subset_msg}")
        try:
            kwargs = {}
            if spec.data_subset:
                kwargs["name"] = spec.data_subset
            load_dataset(spec.data_id, **kwargs)
            print(f"[ok] {spec.name}")
        except Exception as e:
            print(f"[err] {spec.name}: {e}")
            failures.append((spec.name, str(e)))

    if failures:
        print("\n=== Failures ===")
        for name, err in failures:
            print(f"  {name}: {err[:200]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
