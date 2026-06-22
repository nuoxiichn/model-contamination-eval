"""Benchmark 注册表：从 configs/benchmarks.yaml 加载并提供查询接口。"""

from __future__ import annotations

from pathlib import Path

import yaml

from model_contamination.types import BenchmarkSpec, Verdict


class BenchmarkRegistry:
    """benchmark 注册表查询接口。

    用法:
        reg = load_registry()
        spec = reg.get("gsm8k")
        for variant in reg.get_variants("gsm8k"):
            ...
    """

    def __init__(self, specs: list[BenchmarkSpec]):
        self._by_name: dict[str, BenchmarkSpec] = {s.name: s for s in specs}
        self._by_family: dict[str, list[BenchmarkSpec]] = {}
        for s in specs:
            self._by_family.setdefault(s.family, []).append(s)

    def get(self, name: str) -> BenchmarkSpec:
        if name not in self._by_name:
            raise KeyError(f"Unknown benchmark: {name}. Known: {sorted(self._by_name)}")
        return self._by_name[name]

    def get_family(self, family: str) -> list[BenchmarkSpec]:
        return list(self._by_family.get(family, []))

    def get_variants(self, name: str) -> list[BenchmarkSpec]:
        """返回某 benchmark 的所有同族对照集（不包括自身）。"""
        spec = self.get(name)
        return [self.get(v) for v in spec.variants]

    def all(self) -> list[BenchmarkSpec]:
        return list(self._by_name.values())

    def filter_by_method(self, method: str) -> list[BenchmarkSpec]:
        """返回适用某检测方法的所有 benchmark。"""
        return [s for s in self._by_name.values() if method in s.applicable_methods]


def load_registry(config_path: Path | str | None = None) -> BenchmarkRegistry:
    if config_path is None:
        config_path = Path(__file__).resolve().parents[3] / "configs" / "benchmarks.yaml"
    config_path = Path(config_path)
    with config_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    specs = []
    for entry in raw.get("benchmarks", []):
        specs.append(
            BenchmarkSpec(
                name=entry["name"],
                family=entry["family"],
                format=entry["format"],
                language=entry["language"],
                variants=entry.get("variants", []),
                applicable_methods=entry.get("applicable_methods", []),
                trustworthiness_default=Verdict(entry["trustworthiness_default"]),
                data_source=entry["data_source"],
                data_id=entry["data_id"],
                note=entry.get("note", ""),
            )
        )
    # 校验变体引用存在
    by_name = {s.name: s for s in specs}
    for s in specs:
        for v in s.variants:
            if v not in by_name:
                raise ValueError(f"Benchmark {s.name} references unknown variant: {v}")
    return BenchmarkRegistry(specs)
