"""不加载模型的数据层 smoke：确认 sweep 用到的所有 target + control benchmark
都能 load_questions，且 SPV/mink 的 control 映射解析正确。GPU 无关。"""
import sys
sys.path.insert(0, "src")
from model_contamination.benchmarks.registry import load_registry
from model_contamination.benchmarks.loader import load_questions
import yaml
from pathlib import Path

cfg = yaml.safe_load(Path("experiments/2026-07-08_market_model_sweep/run.yaml").read_text())
reg = load_registry()
bcfg = cfg["benchmarks"]
targets = bcfg["targets"]
control_map = bcfg["control_map"]
splits = bcfg.get("splits", {})

needed = set(targets) | set(control_map.values())
print("需要加载的 benchmark:", sorted(needed))
for name in sorted(needed):
    split = splits.get(name, "test")
    try:
        qs = load_questions(reg.get(name), split=split, limit=8)
        ex = qs[0]
        print(f"  OK  {name:10s} split={split:5s} n={len(qs)} fmt={ex.format} prompt[:40]={ex.prompt[:40]!r}")
    except Exception as e:
        print(f"  ERR {name:10s} split={split:5s} {type(e).__name__}: {str(e)[:140]}")

print("\ncontrol 映射:")
for t in targets:
    print(f"  {t:10s} -> {control_map.get(t, '(无, mean_only)')}")
