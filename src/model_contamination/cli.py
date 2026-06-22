"""命令行入口。

用法：
    mcd list-benchmarks
    mcd validate-config
    mcd sanity-check --model PATH \\
                     --benchmark-dirty gsm8k --benchmark-clean math-500 \\
                     --limit 40 --shard-size 5
    mcd detect --model PATH --stage sft --benchmark mmlu-pro
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from model_contamination.benchmarks import load_questions, load_registry
from model_contamination.types import DetectionResult, FamilyDiffResult

console = Console()


@click.group()
@click.version_option()
def main() -> None:
    """Model-level contamination detection CLI."""


@main.command("list-benchmarks")
@click.option("--config", type=click.Path(exists=True), default=None)
@click.option("--method", default=None, help="只列出适用某方法的 benchmark")
def list_benchmarks(config: str | None, method: str | None) -> None:
    reg = load_registry(config)
    specs = reg.filter_by_method(method) if method else reg.all()

    table = Table(title=f"Benchmarks ({len(specs)})")
    table.add_column("name")
    table.add_column("family")
    table.add_column("format")
    table.add_column("lang")
    table.add_column("default")
    table.add_column("variants")

    for s in specs:
        table.add_row(
            s.name, s.family, s.format, s.language,
            s.trustworthiness_default.value,
            ", ".join(s.variants) or "—",
        )
    console.print(table)


@main.command("validate-config")
@click.option("--config", type=click.Path(exists=True), default=None)
def validate_config(config: str | None) -> None:
    """校验 benchmarks.yaml 格式与变体引用完整性。"""
    reg = load_registry(config)
    console.print(f"[green]✓[/green] {len(reg.all())} benchmarks loaded")
    for fam in {s.family for s in reg.all()}:
        members = reg.get_family(fam)
        console.print(f"  family={fam}: {len(members)} members")


@main.command("sanity-check")
@click.option("--model", "model_path", required=True, help="HF model id 或本地权重路径")
@click.option("--stage", default="base", type=click.Choice(["base", "sft", "rlhf"]))
@click.option("--benchmark-dirty", default="gsm8k", help="已知/疑似污染 benchmark")
@click.option("--benchmark-clean", default="math-500", help="作为对照的 benchmark")
@click.option("--limit", type=int, default=40, help="每个 benchmark 截取前 N 题")
@click.option("--shard-size", type=int, default=3, help="Oren shard 大小（sanity 用小值；生产模型可调到 25）")
@click.option("--device", default=None, help="cuda / cpu / cuda:0；默认自动")
@click.option("--outputs", type=click.Path(), default="./outputs/sanity")
def sanity_check(
    model_path: str,
    stage: str,
    benchmark_dirty: str,
    benchmark_clean: str,
    limit: int,
    shard_size: int,
    device: str | None,
    outputs: str,
) -> None:
    """Phase 1 端到端 sanity check：跑通 plumbing。

    红线（CLAUDE.md）：positive control 未到位，本命令只输出按可疑度排序的
    ranked list，**不**给红/黄/绿绝对裁决。
    """
    # 延迟 import torch 等重 dep 直到真要跑
    from model_contamination.models.hf_local import HFLocalModel
    from model_contamination.shared.family_diff import run_family_diff
    from model_contamination.shared.oren_permutation import oren_permutation_test

    out_dir = Path(outputs) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[i] outputs → {out_dir}")

    reg = load_registry()
    spec_dirty = reg.get(benchmark_dirty)
    spec_clean = reg.get(benchmark_clean)

    console.print(f"[i] loading model {model_path} (stage={stage})…")
    model = HFLocalModel(model_path=model_path, stage_tag=stage, device=device)

    rows: list[dict[str, Any]] = []

    # ---- 1. Oren on dirty benchmark ----
    console.print(f"[i] loading {spec_dirty.name} (limit={limit})…")
    qs_dirty = load_questions(spec_dirty, limit=limit)
    console.print(f"[i] Oren on {spec_dirty.name}…")
    r_oren_dirty = oren_permutation_test(
        model, spec_dirty, qs_dirty,
        shard_size=shard_size, min_samples=shard_size * 2,
    )
    rows.append(_result_row(r_oren_dirty))

    # ---- 2. Oren on clean benchmark (对照) ----
    console.print(f"[i] loading {spec_clean.name} (limit={limit})…")
    qs_clean = load_questions(spec_clean, limit=limit)
    console.print(f"[i] Oren on {spec_clean.name}…")
    r_oren_clean = oren_permutation_test(
        model, spec_clean, qs_clean,
        shard_size=shard_size, min_samples=shard_size * 2,
    )
    rows.append(_result_row(r_oren_clean))

    # ---- 3. family_diff on dirty ----
    if spec_dirty.variants:
        console.print(f"[i] family_diff on {spec_dirty.name} vs {spec_dirty.variants}…")
        fd = run_family_diff(model, reg, spec_dirty.name, limit=limit)
        rows.append(_family_diff_row(fd, spec_dirty.name))
    else:
        console.print(f"[yellow]![/yellow] {spec_dirty.name} 无 variants，跳过 family_diff")

    # ---- 写出报告 ----
    json_path = out_dir / "signal_table.json"
    md_path = out_dir / "signal_table.md"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=_json_default))
    md_path.write_text(_render_markdown(rows, model_path, stage))

    # ---- 控制台 ranked list ----
    _print_ranked(rows)
    console.print(f"\n[green]✓[/green] wrote {json_path} and {md_path}")


@main.command("detect")
@click.option("--model", type=click.Path(), required=True)
@click.option("--stage", type=click.Choice(["base", "sft", "rlhf"]), required=True)
@click.option("--benchmark", required=True)
@click.option("--methods", default=None, help="逗号分隔 method tag")
@click.option("--reference", type=click.Path(), default=None, help="reference model（SPV-MIA 需要）")
def detect(
    model: str,
    stage: str,
    benchmark: str,
    methods: str | None,
    reference: str | None,
) -> None:
    """跑单 benchmark 单 checkpoint 的检测。"""
    raise click.ClickException("Phase 1/2 TODO: 串起 stage 分支 + 方法注册 + 报告输出")


# ----------------------------- helpers ----------------------------- #


def _result_row(r: DetectionResult) -> dict[str, Any]:
    return {
        "method": r.method,
        "stage": r.stage.value,
        "benchmark": r.benchmark,
        "signal": r.signal,
        "verdict_hint": r.verdict_hint.value,
        "prerequisites_met": r.prerequisites_met,
        "evidence": r.evidence,
        "error": r.error,
    }


def _family_diff_row(fd: FamilyDiffResult, benchmark: str) -> dict[str, Any]:
    return {
        "method": "family_diff",
        "stage": "n/a",  # family_diff 跨 benchmark，没有单一 stage
        "benchmark": benchmark,
        "signal": fd.delta_avg,  # 百分点
        "verdict_hint": "suspect" if fd.is_island else "clean",
        "prerequisites_met": bool(fd.variant_scores),
        "evidence": {
            "main_score": fd.main_score,
            "variant_scores": fd.variant_scores,
            "delta_max": fd.delta_max,
            "is_island": fd.is_island,
        },
        "error": None if fd.variant_scores else "no variants loaded",
    }


def _print_ranked(rows: list[dict[str, Any]]) -> None:
    """按"可疑度"排序：Oren p-value 越小越可疑；family_diff ΔScore 越大越可疑。"""
    table = Table(title="Ranked signals (small p / large Δ = more suspect)")
    table.add_column("rank")
    table.add_column("method")
    table.add_column("benchmark")
    table.add_column("signal")
    table.add_column("verdict_hint")
    table.add_column("note")

    def _rank_key(row: dict[str, Any]) -> float:
        sig = row.get("signal")
        if sig is None:
            return 1e9  # 沉到最后
        if row["method"] == "oren":
            return float(sig)             # 小 p 排前
        if row["method"] == "family_diff":
            return -float(sig)            # 大 Δ 排前
        return float(sig)

    ranked = sorted(rows, key=_rank_key)
    for i, row in enumerate(ranked, start=1):
        sig = row.get("signal")
        sig_s = "—" if sig is None else f"{sig:.4g}"
        note = row.get("error") or ""
        table.add_row(
            str(i), row["method"], row["benchmark"], sig_s, row["verdict_hint"], note,
        )
    console.print(table)


def _render_markdown(rows: list[dict[str, Any]], model_path: str, stage: str) -> str:
    lines = [
        f"# Sanity-check report",
        f"- model: `{model_path}`",
        f"- stage: `{stage}`",
        f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "> Positive control 未到位（CLAUDE.md 硬前置 5）；仅输出 ranked list 与原始 signal。",
        "",
        "| method | benchmark | signal | verdict_hint | prereq met | note |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        sig = r.get("signal")
        sig_s = "—" if sig is None else f"{sig:.4g}"
        lines.append(
            f"| {r['method']} | {r['benchmark']} | {sig_s} | "
            f"{r['verdict_hint']} | {r['prerequisites_met']} | {r.get('error') or ''} |"
        )
    return "\n".join(lines) + "\n"


def _json_default(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


if __name__ == "__main__":
    main()
