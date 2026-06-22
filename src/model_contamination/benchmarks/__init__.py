"""Benchmark 加载与注册表。"""

from model_contamination.benchmarks.loader import load_questions
from model_contamination.benchmarks.registry import BenchmarkRegistry, load_registry

__all__ = ["BenchmarkRegistry", "load_questions", "load_registry"]
