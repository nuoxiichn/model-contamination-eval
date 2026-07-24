"""命令行入口。

用法：
    mcd list-benchmarks
    mcd validate-config
    mcd run --config configs/examples/local_hf.yaml
"""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from model_contamination.benchmarks import load_registry
from model_contamination.pipeline import build_run_plan, run_evaluation
from model_contamination.run_config import load_run_config

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


@main.command("run")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="运行配置 YAML",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="覆盖 run.output_dir",
)
@click.option("--dry-run", is_flag=True, help="只校验配置并显示任务矩阵")
@click.option("--overwrite", is_flag=True, help="覆盖已有的同名运行产物")
def run_command(
    config_path: Path,
    output_dir: Path | None,
    dry_run: bool,
    overwrite: bool,
) -> None:
    """运行一个模型污染检测配置。"""
    try:
        config = load_run_config(config_path)
        registry = load_registry(config.benchmark_registry)
        plan = build_run_plan(config, registry)
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    if dry_run:
        table = Table(title=f"Run plan: {config.run.id}")
        table.add_column("benchmark")
        table.add_column("method")
        table.add_column("status")
        table.add_column("issues")
        for task in plan.tasks:
            table.add_row(
                task.benchmark,
                task.method,
                "ready" if task.runnable else "blocked",
                "; ".join(
                    f"{issue.severity}: {issue.message}" for issue in task.issues
                )
                or "—",
            )
        console.print(table)
        console.print(f"output_dir={output_dir or config.run.output_dir}")
        if plan.blocked_count:
            raise click.ClickException(
                f"dry-run found {plan.blocked_count} blocked task(s)"
            )
        return

    def show_progress(task: dict) -> None:
        color = {
            "success": "green",
            "inconclusive": "yellow",
            "skipped": "yellow",
            "error": "red",
        }[task["status"]]
        console.print(
            f"[{color}]{task['status']}[/{color}] "
            f"{task['benchmark']} × {task['method']} "
            f"({task['duration_seconds']:.2f}s)"
        )

    try:
        outcome = run_evaluation(
            config,
            output_dir=output_dir,
            overwrite=overwrite,
            progress=show_progress,
        )
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    console.print(f"[bold]run status:[/bold] {outcome.manifest['status']}")
    for name in (
        "input_config.yaml",
        "results.jsonl",
        "report.md",
        "run_manifest.json",
    ):
        console.print(f"  {outcome.output_dir / name}")
    if outcome.has_failures:
        raise click.ClickException(
            "one or more configured tasks failed or were skipped; artifacts were preserved"
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
    """旧的单任务入口；请迁移到配置驱动的 run。"""
    raise click.ClickException(
        "mcd detect has been replaced by `mcd run --config <path>`; "
        "see configs/examples/local_hf.yaml"
    )


if __name__ == "__main__":
    main()
