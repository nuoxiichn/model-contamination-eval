from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from model_contamination.cli import main


def test_run_dry_run_builds_task_matrix(tmp_path: Path):
    config = tmp_path / "run.yaml"
    config.write_text(
        """
schema_version: "1.0"
run: {id: cli-dry-run, output_dir: outputs/unused}
model: {backend: hf_local, path: /models/target, stage: base}
benchmarks: [{name: gsm8k, max_samples: 20}]
methods: [{name: codec, params: {n_seeds: 1}}]
""".strip(),
        encoding="utf-8",
    )
    result = CliRunner().invoke(main, ["run", "--config", str(config), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "cli-dry-run" in result.output
    assert "gsm8k" in result.output
    assert "codec" in result.output
    assert "ready" in result.output


def test_run_dry_run_fails_for_blocked_task(tmp_path: Path):
    config = tmp_path / "run.yaml"
    config.write_text(
        """
schema_version: "1.0"
run: {id: cli-blocked, output_dir: outputs/unused}
model: {backend: hf_local, path: /models/target, stage: sft}
benchmarks: [{name: gsm8k}]
methods: [{name: spv_mia}]
""".strip(),
        encoding="utf-8",
    )
    result = CliRunner().invoke(main, ["run", "--config", str(config), "--dry-run"])
    assert result.exit_code != 0
    assert "model.reference" in result.output
    assert "blocked task" in result.output
