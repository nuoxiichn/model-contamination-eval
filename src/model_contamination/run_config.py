"""Typed configuration for ``mcd run``."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from model_contamination.types import Stage

MethodName = Literal[
    "codec",
    "mink_plus_plus",
    "perm_option",
    "spv_mia",
    "paraphrase",
]

_ENV_PATTERN = re.compile(r"\$\{(?:ENV:)?([A-Za-z_][A-Za-z0-9_]*)\}")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CodecParams(StrictModel):
    n_context: int = Field(default=1, ge=1)
    n_seeds: int = Field(default=5, ge=1)
    skip_first_tokens: int = Field(default=10, ge=0)
    dirty_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    suspect_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    min_samples: int = Field(default=100, ge=1)
    seed: int = 42
    keep_deltas: bool = False


class MinKPlusPlusParams(StrictModel):
    k_ratio: float = Field(default=0.2, gt=0.0, le=1.0)
    min_samples: int = Field(default=30, ge=1)
    estimator: Literal["mean", "trim_mean", "median"] = "trim_mean"
    trim_ratio: float = Field(default=0.1, ge=0.0, lt=0.5)


class PermutationParams(StrictModel):
    max_permutations: int = Field(default=120, ge=1)
    min_samples: int = Field(default=50, ge=1)
    outlier_threshold: float = -0.17
    seed: int = 42


class SpvMiaParams(StrictModel):
    n_neighbors: int = Field(default=10, ge=1)
    mask_ratio: float = Field(default=0.20, gt=0.0, lt=1.0)
    min_samples: int = Field(default=30, ge=1)
    paraphraser: Literal["random"] = "random"
    seed: int = 42


class ParaphraseParams(StrictModel):
    n_paraphrases: int = Field(default=5, ge=1)
    paraphraser: Literal["rule"] = "rule"
    worst_case_drop_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    suspect_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    min_samples: int = Field(default=20, ge=1)
    seed: int = 42
    cache_dir: Path | None = None
    max_tokens: int = Field(default=512, ge=1)


METHOD_PARAMETER_MODELS: dict[str, type[StrictModel]] = {
    "codec": CodecParams,
    "mink_plus_plus": MinKPlusPlusParams,
    "perm_option": PermutationParams,
    "spv_mia": SpvMiaParams,
    "paraphrase": ParaphraseParams,
}


class RunOptions(StrictModel):
    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    seed: int = 42
    output_dir: Path
    fail_fast: bool = False


class ModelLoadConfig(StrictModel):
    backend: Literal["hf_local"] = "hf_local"
    path: str = Field(min_length=1)
    stage: Stage = Stage.BASE
    name: str | None = None
    tokenizer_path: str | None = None
    device: str | None = None
    dtype: str | None = None
    device_map: str | dict[str, Any] | None = None
    trust_remote_code: bool = False

    @model_validator(mode="after")
    def reject_ambiguous_placement(self) -> ModelLoadConfig:
        if self.device is not None and self.device_map is not None:
            raise ValueError("model.device and model.device_map are mutually exclusive")
        return self


class TargetModelConfig(ModelLoadConfig):
    stage: Stage
    reference: ModelLoadConfig | None = None


class ControlConfig(StrictModel):
    name: str = Field(min_length=1)
    split: str = "test"
    subset: str | None = None
    max_samples: int | None = Field(default=None, ge=1)
    selection: Literal["random", "head"] = "random"


class BenchmarkRunConfig(StrictModel):
    name: str = Field(min_length=1)
    split: str = "test"
    subset: str | None = None
    max_samples: int | None = Field(default=None, ge=1)
    selection: Literal["random", "head"] = "random"
    control: ControlConfig | None = None


class MethodRunConfig(StrictModel):
    name: MethodName
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_parameters(self) -> MethodRunConfig:
        parameter_model = METHOD_PARAMETER_MODELS[self.name]
        validated = parameter_model.model_validate(self.params)
        self.params = validated.model_dump(exclude_unset=True)
        return self


class RunConfig(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    benchmark_registry: Path | None = None
    run: RunOptions
    model: TargetModelConfig
    benchmarks: list[BenchmarkRunConfig] = Field(min_length=1)
    methods: list[MethodRunConfig] = Field(min_length=1)

    @field_validator("benchmarks")
    @classmethod
    def benchmark_names_are_unique(
        cls, values: list[BenchmarkRunConfig]
    ) -> list[BenchmarkRunConfig]:
        names = [value.name for value in values]
        if len(names) != len(set(names)):
            raise ValueError("benchmark names must be unique within one run")
        return values

    @field_validator("methods")
    @classmethod
    def method_names_are_unique(
        cls, values: list[MethodRunConfig]
    ) -> list[MethodRunConfig]:
        names = [value.name for value in values]
        if len(names) != len(set(names)):
            raise ValueError("method names must be unique within one run")
        return values


def load_run_config(path: str | Path) -> RunConfig:
    """Load YAML, expand ``${VAR}``/``${ENV:VAR}``, and validate it."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path}: top-level YAML value must be a mapping")
    return RunConfig.model_validate(_expand_environment(raw))


def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if not isinstance(value, str):
        return value

    missing: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        resolved = os.environ.get(name)
        if resolved is None:
            missing.add(name)
            return match.group(0)
        return resolved

    expanded = _ENV_PATTERN.sub(replace, value)
    if missing:
        raise ValueError(f"missing environment variables: {sorted(missing)}")
    return expanded
