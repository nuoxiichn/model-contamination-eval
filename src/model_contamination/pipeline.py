"""Unified, failure-aware execution pipeline for ``mcd run``."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from model_contamination.artifacts import ArtifactWriter
from model_contamination.benchmarks import (
    BenchmarkRegistry,
    load_questions,
    load_registry,
)
from model_contamination.models.base import ModelInterface
from model_contamination.models.hf_local import HFLocalModel
from model_contamination.reports.trustworthiness import (
    aggregate_verdict,
    render_trustworthiness_report,
)
from model_contamination.run_config import (
    BenchmarkRunConfig,
    MethodRunConfig,
    ModelLoadConfig,
    RunConfig,
)
from model_contamination.shared.codec import codec_detect
from model_contamination.shared.paraphrase_stress import paraphrase_stress_test
from model_contamination.stage_base.min_k_plus_plus import mink_plus_plus
from model_contamination.stage_base.option_permutation import option_permutation_test
from model_contamination.stage_sft.spv_mia import spv_mia
from model_contamination.types import (
    BenchmarkQuestion,
    BenchmarkSpec,
    DetectionResult,
    Stage,
    Verdict,
)

IssueSeverity = Literal["warning", "error"]
TaskStatus = Literal["success", "inconclusive", "skipped", "error"]


@dataclass(frozen=True)
class PlanIssue:
    severity: IssueSeverity
    message: str


@dataclass(frozen=True)
class PlannedTask:
    benchmark: str
    method: str
    runnable: bool
    issues: tuple[PlanIssue, ...] = ()


@dataclass(frozen=True)
class RunPlan:
    tasks: tuple[PlannedTask, ...]

    @property
    def blocked_count(self) -> int:
        return sum(not task.runnable for task in self.tasks)


@dataclass
class RunOutcome:
    output_dir: Path
    results: list[DetectionResult]
    manifest: dict[str, Any]
    plan: RunPlan

    @property
    def has_failures(self) -> bool:
        return any(task["status"] in {"error", "skipped"} for task in self.manifest["tasks"])


MethodExecutor = Callable[
    [
        ModelInterface,
        ModelInterface | None,
        BenchmarkSpec,
        list[BenchmarkQuestion],
        list[BenchmarkQuestion] | None,
        dict[str, Any],
        int,
    ],
    DetectionResult,
]


def _create_hf_model(config: ModelLoadConfig) -> ModelInterface:
    kwargs: dict[str, Any] = {
        "name": config.name,
        "trust_remote_code": config.trust_remote_code,
    }
    for key in ("device", "dtype", "tokenizer_path", "device_map"):
        value = getattr(config, key)
        if value is not None:
            kwargs[key] = value
    return HFLocalModel(config.path, stage_tag=config.stage.value, **kwargs)


def _codec_executor(
    model: ModelInterface,
    reference: ModelInterface | None,
    spec: BenchmarkSpec,
    questions: list[BenchmarkQuestion],
    control: list[BenchmarkQuestion] | None,
    params: dict[str, Any],
    seed: int,
) -> DetectionResult:
    del reference, control
    params.setdefault("seed", seed)
    return codec_detect(model, spec, questions, **params)


def _mink_executor(
    model: ModelInterface,
    reference: ModelInterface | None,
    spec: BenchmarkSpec,
    questions: list[BenchmarkQuestion],
    control: list[BenchmarkQuestion] | None,
    params: dict[str, Any],
    seed: int,
) -> DetectionResult:
    del reference, seed
    return mink_plus_plus(
        model,
        spec,
        questions,
        control_questions=control,
        **params,
    )


def _permutation_executor(
    model: ModelInterface,
    reference: ModelInterface | None,
    spec: BenchmarkSpec,
    questions: list[BenchmarkQuestion],
    control: list[BenchmarkQuestion] | None,
    params: dict[str, Any],
    seed: int,
) -> DetectionResult:
    del reference, control
    params.setdefault("seed", seed)
    return option_permutation_test(model, spec, questions, **params)


def _spv_executor(
    model: ModelInterface,
    reference: ModelInterface | None,
    spec: BenchmarkSpec,
    questions: list[BenchmarkQuestion],
    control: list[BenchmarkQuestion] | None,
    params: dict[str, Any],
    seed: int,
) -> DetectionResult:
    if reference is None:
        raise RuntimeError("SPV-MIA requires a reference model")
    params.setdefault("seed", seed)
    return spv_mia(
        model,
        reference,
        spec,
        questions,
        control_questions=control,
        **params,
    )


def _paraphrase_executor(
    model: ModelInterface,
    reference: ModelInterface | None,
    spec: BenchmarkSpec,
    questions: list[BenchmarkQuestion],
    control: list[BenchmarkQuestion] | None,
    params: dict[str, Any],
    seed: int,
) -> DetectionResult:
    del reference, control
    params.setdefault("seed", seed)
    return paraphrase_stress_test(model, spec, questions, **params)


DEFAULT_METHOD_EXECUTORS: dict[str, MethodExecutor] = {
    "codec": _codec_executor,
    "mink_plus_plus": _mink_executor,
    "perm_option": _permutation_executor,
    "spv_mia": _spv_executor,
    "paraphrase": _paraphrase_executor,
}


@dataclass
class RunnerDependencies:
    registry_loader: Callable[[Path | str | None], BenchmarkRegistry] = load_registry
    question_loader: Callable[..., list[BenchmarkQuestion]] = load_questions
    model_factory: Callable[[ModelLoadConfig], ModelInterface] = _create_hf_model
    method_executors: Mapping[str, MethodExecutor] = field(
        default_factory=lambda: dict(DEFAULT_METHOD_EXECUTORS)
    )


def build_run_plan(config: RunConfig, registry: BenchmarkRegistry) -> RunPlan:
    """Build the full task matrix without loading a model or dataset."""
    tasks: list[PlannedTask] = []
    for benchmark_config in config.benchmarks:
        try:
            spec = registry.get(benchmark_config.name)
            benchmark_error = None
        except KeyError as error:
            spec = None
            benchmark_error = str(error)

        for method_config in config.methods:
            issues: list[PlanIssue] = []
            if benchmark_error is not None:
                issues.append(PlanIssue("error", benchmark_error))
            elif spec is not None:
                issues.extend(_task_issues(config, benchmark_config, method_config, spec, registry))
            tasks.append(
                PlannedTask(
                    benchmark=benchmark_config.name,
                    method=method_config.name,
                    runnable=not any(issue.severity == "error" for issue in issues),
                    issues=tuple(issues),
                )
            )
    return RunPlan(tasks=tuple(tasks))


def _task_issues(
    config: RunConfig,
    benchmark_config: BenchmarkRunConfig,
    method_config: MethodRunConfig,
    spec: BenchmarkSpec,
    registry: BenchmarkRegistry,
) -> list[PlanIssue]:
    issues: list[PlanIssue] = []
    method = method_config.name
    if method not in spec.applicable_methods:
        issues.append(
            PlanIssue("error", f"{method} is not registered for benchmark {spec.name}")
        )
    if method == "perm_option" and spec.format != "multiple_choice":
        issues.append(PlanIssue("error", "perm_option requires multiple_choice format"))
    if method == "paraphrase" and spec.format != "math_cot":
        issues.append(PlanIssue("error", "paraphrase currently supports math_cot only"))
    if method == "spv_mia" and config.model.reference is None:
        issues.append(PlanIssue("error", "spv_mia requires model.reference"))
    if method == "spv_mia" and config.model.stage == Stage.BASE:
        issues.append(
            PlanIssue("warning", "spv_mia on a base target is usually not meaningful")
        )
    if spec.stage_targets and config.model.stage.value not in spec.stage_targets:
        issues.append(
            PlanIssue(
                "warning",
                f"benchmark metadata does not target stage={config.model.stage.value}",
            )
        )
    if benchmark_config.control is not None and method in {"mink_plus_plus", "spv_mia"}:
        try:
            registry.get(benchmark_config.control.name)
        except KeyError as error:
            issues.append(PlanIssue("error", f"configured control is unknown: {error}"))
    return issues


def run_evaluation(
    config: RunConfig,
    *,
    output_dir: Path | None = None,
    overwrite: bool = False,
    dependencies: RunnerDependencies | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> RunOutcome:
    """Run all configured tasks and always record task-level failures."""
    dependencies = dependencies or RunnerDependencies()
    effective_output = output_dir or config.run.output_dir
    effective_config = config.model_copy(
        update={"run": config.run.model_copy(update={"output_dir": effective_output})}
    )
    registry = dependencies.registry_loader(effective_config.benchmark_registry)
    plan = build_run_plan(effective_config, registry)
    writer = ArtifactWriter(effective_output, overwrite=overwrite)

    started_at = _utc_now()
    started_perf = time.perf_counter()
    manifest = _initial_manifest(effective_config, plan, started_at)
    results: list[DetectionResult] = []
    writer.prepare(effective_config)
    writer.write_manifest(manifest)

    target_model: ModelInterface | None = None
    reference_model: ModelInterface | None = None
    target_model_error: str | None = None
    reference_model_error: str | None = None
    try:
        if any(task.runnable for task in plan.tasks):
            try:
                target_model = dependencies.model_factory(effective_config.model)
            except Exception as error:  # noqa: BLE001 - failures belong in the manifest
                target_model_error = _format_exception(error)

        benchmark_configs = {item.name: item for item in effective_config.benchmarks}
        method_configs = {item.name: item for item in effective_config.methods}
        target_cache: dict[str, tuple[list[BenchmarkQuestion] | None, str | None]] = {}
        control_cache: dict[
            tuple[str, str, str | None, int | None, str],
            tuple[list[BenchmarkQuestion] | None, str | None],
        ] = {}
        abort_reason: str | None = None

        for planned in plan.tasks:
            benchmark_config = benchmark_configs[planned.benchmark]
            method_config = method_configs[planned.method]
            task_started = _utc_now()
            task_perf = time.perf_counter()
            task_record: dict[str, Any] = {
                "id": f"{planned.benchmark}:{planned.method}",
                "benchmark": planned.benchmark,
                "method": planned.method,
                "status": "error",
                "started_at": task_started,
                "method_params": method_config.params,
                "issues": [
                    {"severity": issue.severity, "message": issue.message}
                    for issue in planned.issues
                ],
            }

            if abort_reason is not None:
                result = _inconclusive_result(
                    planned.method,
                    effective_config.model.stage,
                    planned.benchmark,
                    abort_reason,
                )
                task_record["status"] = "skipped"
                task_record["error"] = abort_reason
            elif not planned.runnable:
                reason = "; ".join(
                    issue.message for issue in planned.issues if issue.severity == "error"
                )
                result = _inconclusive_result(
                    planned.method,
                    effective_config.model.stage,
                    planned.benchmark,
                    reason,
                )
                task_record["status"] = "skipped"
                task_record["error"] = reason
            elif target_model_error is not None or target_model is None:
                reason = f"target model load failed: {target_model_error or 'unknown error'}"
                result = _inconclusive_result(
                    planned.method,
                    effective_config.model.stage,
                    planned.benchmark,
                    reason,
                )
                task_record["status"] = "error"
                task_record["error"] = reason
            else:
                spec = registry.get(planned.benchmark)
                if planned.benchmark not in target_cache:
                    target_cache[planned.benchmark] = _load_benchmark(
                        dependencies,
                        spec,
                        benchmark_config,
                        effective_config.run.seed,
                    )
                questions, load_error = target_cache[planned.benchmark]
                if load_error is not None or questions is None:
                    reason = f"benchmark load failed: {load_error or 'unknown error'}"
                    result = _inconclusive_result(
                        planned.method,
                        effective_config.model.stage,
                        planned.benchmark,
                        reason,
                    )
                    task_record["status"] = "error"
                    task_record["error"] = reason
                else:
                    task_record["n_samples"] = len(questions)
                    control, control_error = _control_for_task(
                        dependencies,
                        registry,
                        benchmark_config,
                        method_config.name,
                        effective_config.run.seed,
                        control_cache,
                    )
                    if control_error is not None:
                        result = _inconclusive_result(
                            planned.method,
                            effective_config.model.stage,
                            planned.benchmark,
                            f"control load failed: {control_error}",
                        )
                        task_record["status"] = "error"
                        task_record["error"] = result.error
                    else:
                        if control is not None:
                            task_record["n_control_samples"] = len(control)
                        if (
                            planned.method == "spv_mia"
                            and reference_model is None
                            and reference_model_error is None
                        ):
                            try:
                                assert effective_config.model.reference is not None
                                reference_model = dependencies.model_factory(
                                    effective_config.model.reference
                                )
                            except Exception as error:  # noqa: BLE001
                                reference_model_error = _format_exception(error)
                        if planned.method == "spv_mia" and reference_model_error is not None:
                            reason = f"reference model load failed: {reference_model_error}"
                            result = _inconclusive_result(
                                planned.method,
                                effective_config.model.stage,
                                planned.benchmark,
                                reason,
                            )
                            task_record["status"] = "error"
                            task_record["error"] = reason
                        else:
                            try:
                                executor = dependencies.method_executors[planned.method]
                                result = executor(
                                    target_model,
                                    reference_model,
                                    spec,
                                    questions,
                                    control,
                                    dict(method_config.params),
                                    effective_config.run.seed,
                                )
                                task_record["status"] = _status_from_result(result)
                            except Exception as error:  # noqa: BLE001
                                reason = f"method execution failed: {_format_exception(error)}"
                                result = _inconclusive_result(
                                    planned.method,
                                    effective_config.model.stage,
                                    planned.benchmark,
                                    reason,
                                )
                                task_record["status"] = "error"
                                task_record["error"] = reason

            results.append(result)
            writer.append_result(result)
            task_record["ended_at"] = _utc_now()
            task_record["duration_seconds"] = round(time.perf_counter() - task_perf, 6)
            manifest["tasks"].append(task_record)
            _update_running_manifest(manifest, started_perf, writer)
            writer.write_manifest(manifest)
            if progress is not None:
                progress(task_record)

            if effective_config.run.fail_fast and task_record["status"] == "error":
                abort_reason = f"skipped because fail_fast stopped after {task_record['id']}"

        _finalize_manifest(manifest, started_perf, writer)
        writer.write_report(_render_run_report(effective_config, results, manifest))
        manifest["artifacts"] = writer.artifact_records()
        writer.write_manifest(manifest)
    finally:
        writer.close()

    return RunOutcome(
        output_dir=effective_output,
        results=results,
        manifest=manifest,
        plan=plan,
    )


def _load_benchmark(
    dependencies: RunnerDependencies,
    spec: BenchmarkSpec,
    config: BenchmarkRunConfig,
    seed: int,
) -> tuple[list[BenchmarkQuestion] | None, str | None]:
    try:
        kwargs = _selection_kwargs(
            split=config.split,
            subset=config.subset,
            max_samples=config.max_samples,
            selection=config.selection,
            seed=seed,
        )
        return dependencies.question_loader(spec, **kwargs), None
    except Exception as error:  # noqa: BLE001
        return None, _format_exception(error)


def _control_for_task(
    dependencies: RunnerDependencies,
    registry: BenchmarkRegistry,
    benchmark_config: BenchmarkRunConfig,
    method: str,
    seed: int,
    cache: dict[
        tuple[str, str, str | None, int | None, str],
        tuple[list[BenchmarkQuestion] | None, str | None],
    ],
) -> tuple[list[BenchmarkQuestion] | None, str | None]:
    if method not in {"mink_plus_plus", "spv_mia"} or benchmark_config.control is None:
        return None, None
    control = benchmark_config.control
    max_samples = control.max_samples or benchmark_config.max_samples
    key = (control.name, control.split, control.subset, max_samples, control.selection)
    if key not in cache:
        try:
            spec = registry.get(control.name)
            kwargs = _selection_kwargs(
                split=control.split,
                subset=control.subset,
                max_samples=max_samples,
                selection=control.selection,
                seed=seed,
            )
            cache[key] = (dependencies.question_loader(spec, **kwargs), None)
        except Exception as error:  # noqa: BLE001
            cache[key] = (None, _format_exception(error))
    return cache[key]


def _selection_kwargs(
    *,
    split: str,
    subset: str | None,
    max_samples: int | None,
    selection: str,
    seed: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"split": split, "subset": subset}
    if max_samples is not None:
        if selection == "random":
            kwargs.update(sample=max_samples, sample_seed=seed)
        else:
            kwargs["limit"] = max_samples
    return kwargs


def _status_from_result(result: DetectionResult) -> TaskStatus:
    if not result.prerequisites_met or result.verdict_hint == Verdict.INCONCLUSIVE:
        return "inconclusive"
    return "success"


def _inconclusive_result(
    method: str,
    stage: Stage,
    benchmark: str,
    error: str,
) -> DetectionResult:
    return DetectionResult(
        method=method,
        stage=stage,
        benchmark=benchmark,
        signal=None,
        verdict_hint=Verdict.INCONCLUSIVE,
        prerequisites_met=False,
        evidence={},
        error=error,
    )


def _initial_manifest(config: RunConfig, plan: RunPlan, started_at: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "run_id": config.run.id,
        "status": "running",
        "code": _git_state(),
        "model": config.model.model_dump(mode="json"),
        "benchmarks": [item.model_dump(mode="json") for item in config.benchmarks],
        "methods": [item.model_dump(mode="json") for item in config.methods],
        "timing": {"started_at": started_at, "ended_at": None, "wall_time_seconds": 0.0},
        "environment": _environment_info(),
        "resources": {
            "configured_device": config.model.device,
            "configured_device_map": config.model.device_map,
            "dtype": config.model.dtype,
        },
        "plan": {
            "task_count": len(plan.tasks),
            "blocked_count": plan.blocked_count,
        },
        "tasks": [],
        "summary": {},
        "artifacts": [],
    }


def _update_running_manifest(
    manifest: dict[str, Any], started_perf: float, writer: ArtifactWriter
) -> None:
    manifest["timing"]["wall_time_seconds"] = round(time.perf_counter() - started_perf, 6)
    manifest["summary"] = dict(Counter(task["status"] for task in manifest["tasks"]))
    manifest["artifacts"] = writer.artifact_records()


def _finalize_manifest(
    manifest: dict[str, Any], started_perf: float, writer: ArtifactWriter
) -> None:
    _update_running_manifest(manifest, started_perf, writer)
    counts = Counter(task["status"] for task in manifest["tasks"])
    if counts["error"] or counts["skipped"]:
        status = "completed_with_errors"
    elif counts["inconclusive"]:
        status = "completed_with_inconclusive"
    else:
        status = "completed"
    manifest["status"] = status
    manifest["timing"]["ended_at"] = _utc_now()


def _render_run_report(
    config: RunConfig,
    results: list[DetectionResult],
    manifest: dict[str, Any],
) -> str:
    counts = Counter(task["status"] for task in manifest["tasks"])
    lines = [
        "# Model Contamination Evaluation Report",
        "",
        "> This report uses relative-ranking semantics. Method hints are not calibrated ",
        "> probabilities or proof that a checkpoint trained on a benchmark.",
        "",
        "## Run Metadata",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Run ID | `{_md(config.run.id)}` |",
        f"| Model | `{_md(config.model.name or config.model.path)}` |",
        f"| Stage | `{config.model.stage.value}` |",
        f"| Git commit | `{_md(manifest['code']['git_commit'])}` |",
        f"| Git dirty | `{manifest['code']['dirty']}` |",
        f"| Started | `{manifest['timing']['started_at']}` |",
        f"| Ended | `{manifest['timing']['ended_at']}` |",
        f"| Wall time | `{manifest['timing']['wall_time_seconds']:.3f}s` |",
        f"| Status | `{manifest['status']}` |",
        "",
        "## Execution Summary",
        "",
        (
            f"Configured {len(manifest['tasks'])} tasks: "
            f"{counts['success']} success, {counts['inconclusive']} inconclusive, "
            f"{counts['skipped']} skipped, {counts['error']} error."
        ),
        "",
        "| Benchmark | Method | Status | Samples | Control | Duration | Error |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for task in manifest["tasks"]:
        lines.append(
            f"| `{_md(task['benchmark'])}` | `{_md(task['method'])}` | "
            f"{task['status']} | {task.get('n_samples', '—')} | "
            f"{task.get('n_control_samples', '—')} | "
            f"{task.get('duration_seconds', 0.0):.3f}s | "
            f"{_md(task.get('error', ''))} |"
        )

    grouped: dict[str, list[DetectionResult]] = defaultdict(list)
    for result in results:
        grouped[result.benchmark].append(result)
    verdicts = [
        aggregate_verdict(name, config.model.stage, benchmark_results, calibrated=False)
        for name, benchmark_results in grouped.items()
    ]
    detail = _shift_markdown_headings(
        render_trustworthiness_report(verdicts, calibrated=False)
    )
    lines.extend(
        [
            "",
            "## Method Results",
            "",
            detail.rstrip(),
            "",
            "## Interpretation Limits",
            "",
            "- `inconclusive` and failed prerequisites do not mean clean.",
            "- AUC is meaningful only when a declared control was loaded successfully.",
            "- SPV-MIA requires a compatible reference checkpoint; short MC answers are unsupported.",
            "- Absolute release gating requires frozen positive/negative controls and calibrated thresholds.",
            "- Full execution metadata and failures are recorded in `run_manifest.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def _shift_markdown_headings(markdown: str) -> str:
    shifted = []
    for line in markdown.splitlines():
        if line.startswith("#"):
            count = len(line) - len(line.lstrip("#"))
            line = "#" * (count + 2) + line[count:]
        shifted.append(line)
    return "\n".join(shifted)


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _git_state() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"git_commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unknown", "dirty": True}


def _environment_info() -> dict[str, Any]:
    package_versions = {}
    for package in ("numpy", "scipy", "scikit-learn", "torch", "transformers", "datasets"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions,
    }


def _format_exception(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
