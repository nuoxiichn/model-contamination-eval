"""命令行入口。

用法：
    mcd list-benchmarks
    mcd validate-config
    mcd sanity-check --model-base PATH --model-target PATH \\
                     --benchmark-dirty gsm8k --benchmark-clean livebench
    mcd detect --model PATH --stage sft --benchmark mmlu
"""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from model_contamination.benchmarks import load_registry

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
@click.option("--model-base", type=click.Path(), required=True)
@click.option("--model-target", type=click.Path(), required=True)
@click.option("--benchmark-dirty", default="gsm8k")
@click.option("--benchmark-clean", default="livebench")
@click.option("--outputs", type=click.Path(), default="./outputs/sanity")
def sanity_check(
    model_base: str,
    model_target: str,
    benchmark_dirty: str,
    benchmark_clean: str,
    outputs: str,
) -> None:
    """Phase 1 端到端 sanity check。

    跑两个 benchmark（已知污染 + 干净参照）+ 两个方法（Oren + 同族 ΔScore），
    验证 plumbing 工作正常。
    """
    Path(outputs).mkdir(parents=True, exist_ok=True)
    console.print("[yellow]Phase 1 sanity check 待实装[/yellow]")
    console.print(f"  base   = {model_base}")
    console.print(f"  target = {model_target}")
    console.print(f"  dirty  = {benchmark_dirty}")
    console.print(f"  clean  = {benchmark_clean}")
    raise click.ClickException(
        "Phase 1 TODO: 串起 model loader + benchmark loader + Oren + family_diff"
    )


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


if __name__ == "__main__":
    main()
