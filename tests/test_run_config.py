from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from model_contamination.run_config import RunConfig, load_run_config


def _raw(output_dir: str = "outputs/test") -> dict:
    return {
        "schema_version": "1.0",
        "run": {"id": "test-run", "seed": 7, "output_dir": output_dir},
        "model": {"backend": "hf_local", "path": "/models/target", "stage": "base"},
        "benchmarks": [{"name": "gsm8k", "max_samples": 20}],
        "methods": [{"name": "codec", "params": {"n_seeds": 2}}],
    }


def test_run_config_accepts_known_method_params():
    config = RunConfig.model_validate(_raw())
    assert config.run.seed == 7
    assert config.model.stage.value == "base"
    assert config.methods[0].params == {"n_seeds": 2}


def test_run_config_rejects_unknown_method_param():
    raw = _raw()
    raw["methods"][0]["params"] = {"not_a_codec_param": 1}
    with pytest.raises(ValidationError, match="not_a_codec_param"):
        RunConfig.model_validate(raw)


def test_run_config_rejects_invalid_method_param_value():
    raw = _raw()
    raw["methods"][0]["params"] = {"n_seeds": 0}
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        RunConfig.model_validate(raw)


def test_run_config_rejects_duplicate_tasks():
    raw = _raw()
    raw["benchmarks"].append({"name": "gsm8k"})
    with pytest.raises(ValidationError, match="benchmark names must be unique"):
        RunConfig.model_validate(raw)


def test_load_run_config_expands_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "run.yaml"
    path.write_text(
        """
schema_version: "1.0"
run: {id: env-test, seed: 42, output_dir: "${RUN_OUTPUT}"}
model: {backend: hf_local, path: "${ENV:MODEL_PATH}", stage: sft}
benchmarks: [{name: gsm8k}]
methods: [{name: codec}]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("RUN_OUTPUT", str(tmp_path / "output"))
    monkeypatch.setenv("MODEL_PATH", "/models/local-sft")
    config = load_run_config(path)
    assert config.run.output_dir == tmp_path / "output"
    assert config.model.path == "/models/local-sft"


def test_load_run_config_reports_missing_environment(tmp_path: Path):
    path = tmp_path / "run.yaml"
    path.write_text(
        """
schema_version: "1.0"
run: {id: env-test, output_dir: outputs/test}
model: {backend: hf_local, path: "${DEFINITELY_MISSING_MODEL_PATH}", stage: base}
benchmarks: [{name: gsm8k}]
methods: [{name: codec}]
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="DEFINITELY_MISSING_MODEL_PATH"):
        load_run_config(path)
