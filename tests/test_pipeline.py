from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_contamination.benchmarks.registry import BenchmarkRegistry
from model_contamination.models.base import Capability, ModelInterface
from model_contamination.pipeline import (
    RunnerDependencies,
    build_run_plan,
    run_evaluation,
)
from model_contamination.run_config import RunConfig
from model_contamination.types import (
    BenchmarkQuestion,
    BenchmarkSpec,
    DetectionResult,
    Stage,
    Verdict,
)


class _FakeModel(ModelInterface):
    name = "fake-model"
    stage_tag = "base"

    def supports(self, cap: Capability) -> bool:
        return True

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        del prompt, max_tokens, temperature
        return "1"


def _spec(name: str, *, fmt: str = "math_cot") -> BenchmarkSpec:
    return BenchmarkSpec(
        name=name,
        family="test",
        format=fmt,  # type: ignore[arg-type]
        language="en",
        variants=[],
        applicable_methods=[
            "codec",
            "mink_plus_plus",
            "perm_option",
            "spv_mia",
            "paraphrase",
        ],
        trustworthiness_default=Verdict.SUSPECT,
        data_source="local",
        data_id="unused.jsonl",
        stage_targets=["base", "sft"],
    )


def _question(benchmark: str, index: int) -> BenchmarkQuestion:
    return BenchmarkQuestion(
        id=f"{benchmark}-{index}",
        benchmark=benchmark,
        format="math_cot",
        prompt=f"question {index}",
        answer=str(index),
        full_answer=f"reasoning for answer number {index}",
    )


def _config(output_dir: Path, *, fail_fast: bool = False) -> RunConfig:
    return RunConfig.model_validate(
        {
            "schema_version": "1.0",
            "run": {
                "id": "pipeline-test",
                "seed": 11,
                "output_dir": str(output_dir),
                "fail_fast": fail_fast,
            },
            "model": {
                "backend": "hf_local",
                "path": "/models/target",
                "stage": "base",
            },
            "benchmarks": [
                {
                    "name": "target",
                    "max_samples": 3,
                    "control": {"name": "control", "max_samples": 2},
                }
            ],
            "methods": [
                {"name": "codec", "params": {"n_seeds": 1}},
                {"name": "mink_plus_plus", "params": {"k_ratio": 0.2}},
            ],
        }
    )


def _executor(method: str, *, fail: bool = False):
    def execute(model, reference, spec, questions, control, params, seed):
        del model, reference, params
        if fail:
            raise RuntimeError("intentional executor failure")
        return DetectionResult(
            method=method,
            stage=Stage.BASE,
            benchmark=spec.name,
            signal=0.25,
            verdict_hint=Verdict.CLEAN,
            prerequisites_met=True,
            evidence={
                "n_samples": len(questions),
                "n_control": len(control) if control is not None else 0,
                "seed": seed,
            },
        )

    return execute


def _dependencies(*, codec_fails: bool = False) -> RunnerDependencies:
    registry = BenchmarkRegistry([_spec("target"), _spec("control")])

    def question_loader(spec, **kwargs):
        n = kwargs.get("sample") or kwargs.get("limit") or 4
        return [_question(spec.name, i) for i in range(n)]

    return RunnerDependencies(
        registry_loader=lambda path: registry,
        question_loader=question_loader,
        model_factory=lambda config: _FakeModel(),
        method_executors={
            "codec": _executor("codec", fail=codec_fails),
            "mink_plus_plus": _executor("mink_plus_plus"),
        },
    )


def test_run_evaluation_writes_complete_artifacts(tmp_path: Path):
    output_dir = tmp_path / "run"
    outcome = run_evaluation(
        _config(output_dir),
        dependencies=_dependencies(),
    )

    assert outcome.manifest["status"] == "completed"
    assert len(outcome.results) == 2
    for name in ("input_config.yaml", "results.jsonl", "report.md", "run_manifest.json"):
        assert (output_dir / name).is_file()

    rows = [json.loads(line) for line in (output_dir / "results.jsonl").read_text().splitlines()]
    assert [row["method"] for row in rows] == ["codec", "mink_plus_plus"]
    assert rows[1]["evidence"]["n_control"] == 2

    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert manifest["summary"] == {"success": 2}
    assert manifest["tasks"][0]["n_samples"] == 3
    assert "Model Contamination Evaluation Report" in (output_dir / "report.md").read_text()


def test_method_error_is_recorded_and_other_tasks_continue(tmp_path: Path):
    outcome = run_evaluation(
        _config(tmp_path / "run"),
        dependencies=_dependencies(codec_fails=True),
    )
    assert outcome.manifest["status"] == "completed_with_errors"
    assert [task["status"] for task in outcome.manifest["tasks"]] == ["error", "success"]
    assert outcome.results[0].verdict_hint == Verdict.INCONCLUSIVE
    assert "intentional executor failure" in (outcome.results[0].error or "")
    assert outcome.has_failures


def test_fail_fast_still_records_remaining_tasks(tmp_path: Path):
    outcome = run_evaluation(
        _config(tmp_path / "run", fail_fast=True),
        dependencies=_dependencies(codec_fails=True),
    )
    assert [task["status"] for task in outcome.manifest["tasks"]] == ["error", "skipped"]
    assert len(outcome.results) == 2


def test_existing_artifacts_require_explicit_overwrite(tmp_path: Path):
    config = _config(tmp_path / "run")
    run_evaluation(config, dependencies=_dependencies())
    with pytest.raises(FileExistsError, match="--overwrite"):
        run_evaluation(config, dependencies=_dependencies())


def test_plan_blocks_spv_without_reference_and_wrong_format(tmp_path: Path):
    config = RunConfig.model_validate(
        {
            "run": {"id": "blocked", "output_dir": str(tmp_path)},
            "model": {"path": "/models/target", "stage": "sft"},
            "benchmarks": [{"name": "mc"}],
            "methods": [{"name": "spv_mia"}, {"name": "paraphrase"}],
        }
    )
    registry = BenchmarkRegistry([_spec("mc", fmt="multiple_choice")])
    plan = build_run_plan(config, registry)
    assert plan.blocked_count == 2
    assert all(not task.runnable for task in plan.tasks)


def test_all_blocked_tasks_do_not_load_model(tmp_path: Path):
    config = RunConfig.model_validate(
        {
            "run": {"id": "blocked-run", "output_dir": str(tmp_path / "run")},
            "model": {"path": "/models/target", "stage": "sft"},
            "benchmarks": [{"name": "mc"}],
            "methods": [{"name": "spv_mia"}, {"name": "paraphrase"}],
        }
    )
    registry = BenchmarkRegistry([_spec("mc", fmt="multiple_choice")])

    def fail_if_loaded(config):
        raise AssertionError(f"blocked plan loaded model: {config.path}")

    outcome = run_evaluation(
        config,
        dependencies=RunnerDependencies(
            registry_loader=lambda path: registry,
            model_factory=fail_if_loaded,
        ),
    )

    assert [task["status"] for task in outcome.manifest["tasks"]] == [
        "skipped",
        "skipped",
    ]
    assert outcome.manifest["status"] == "completed_with_errors"
    assert len(outcome.results) == 2
