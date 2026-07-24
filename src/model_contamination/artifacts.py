"""Artifact serialization and incremental run output writing."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import yaml

from model_contamination.run_config import RunConfig
from model_contamination.types import DetectionResult

ARTIFACT_NAMES = (
    "input_config.yaml",
    "results.jsonl",
    "report.md",
    "run_manifest.json",
)


def to_jsonable(value: Any) -> Any:
    """Convert enums, dataclasses, paths, and NumPy values to strict JSON."""
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def detection_result_to_dict(result: DetectionResult) -> dict[str, Any]:
    return to_jsonable(result)


class ArtifactWriter:
    """Write a run incrementally so partial results survive method failures."""

    def __init__(self, output_dir: Path, *, overwrite: bool = False) -> None:
        self.output_dir = output_dir
        self.overwrite = overwrite
        self._results_handle: TextIO | None = None

    def prepare(self, config: RunConfig) -> None:
        existing = [name for name in ARTIFACT_NAMES if (self.output_dir / name).exists()]
        if existing and not self.overwrite:
            raise FileExistsError(
                f"output directory already contains run artifacts {existing}: "
                f"{self.output_dir}; use --overwrite to replace them"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_write_text(
            "input_config.yaml",
            yaml.safe_dump(
                config.model_dump(mode="json"),
                allow_unicode=True,
                sort_keys=False,
            ),
        )
        self._results_handle = (self.output_dir / "results.jsonl").open(
            "w", encoding="utf-8"
        )

    def append_result(self, result: DetectionResult) -> None:
        if self._results_handle is None:
            raise RuntimeError("ArtifactWriter.prepare() must be called first")
        payload = detection_result_to_dict(result)
        self._results_handle.write(
            json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n"
        )
        self._results_handle.flush()

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        self._atomic_write_text(
            "run_manifest.json",
            json.dumps(
                to_jsonable(manifest),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
        )

    def write_report(self, report: str) -> None:
        self._atomic_write_text("report.md", report)

    def close(self) -> None:
        if self._results_handle is not None:
            self._results_handle.close()
            self._results_handle = None

    def artifact_records(self) -> list[dict[str, Any]]:
        records = []
        for name in ARTIFACT_NAMES:
            path = self.output_dir / name
            if path.exists():
                records.append({"path": name, "size_bytes": path.stat().st_size})
        return records

    def _atomic_write_text(self, name: str, content: str) -> None:
        path = self.output_dir / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
