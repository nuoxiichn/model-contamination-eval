#!/usr/bin/env python3
"""聚焦重跑：SPV-MIA × math-500（loader full_answer 修复后）。

## 背景

首轮 sweep 里 SPV 的 math-500 全部降级失败：
    "Only 4 finite scores out of 200 questions (likely too-short completions)."
根因是 loader 把 CoT 塞进 raw['solution']，而 spv 只读 raw['full_answer']，
取不到 → 回退到裸答案（1-2 词）→ 全 NaN。

修复（本仓库 loader.py / types.py / spv_mia.py）：CoT 统一提到顶层 q.full_answer，
spv 读这个固定字段。math-500 的 completion 恢复为完整解题过程。

## 为什么单独跑、不跑 run_sweep.py

run.yaml 的 perm_option 之后新增了 gpqa / gpqa-diamond / mmmlu，直接跑 run_sweep
主流程会把这些「全模型 × 新 benchmark」当 pending 一并触发（还可能连带下载）。
本脚本只重跑「SPV × math-500」这一格，绕开主流程。

## 只重跑「被本次修复对症」的模型

从既有 jsonl 自动推导：只挑上次 math-500 spv 因 too-short completion 降级的模型
（degraded_reason == too_few_finite_scores）。自动跳过：
  - llama / gemma：gated 权重未下（load_error）——非本次修复范畴
  - internlm：DynamicCache 环境报错——非本次修复范畴

## 跑法（8 卡 MetaX，reference 走 cuda:1，与 run.yaml 一致）

    HF_ENDPOINT=https://hf-mirror.com HF_HOME=/mnt/public/code/chennuoxi/hf_cache \
      PYTHONPATH=src /opt/conda/bin/python \
      experiments/2026-07-08_market_model_sweep/rerun_spv_math500.py

结果追加进同一 outputs/sweep_results.jsonl；旧的 math-500 spv 行先备份再删除。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "src"))

from model_contamination.benchmarks.loader import load_questions  # noqa: E402
from model_contamination.benchmarks.registry import load_registry  # noqa: E402
from model_contamination.models.hf_local import HFLocalModel  # noqa: E402
from model_contamination.stage_sft.spv_mia import spv_mia  # noqa: E402

_OUT = _HERE / "outputs"
_RESULTS = _OUT / "sweep_results.jsonl"
_BENCH = "math-500"


def _row_from_result(model: str, bench: str, r) -> dict:
    return {
        "model": model, "method": "spv_mia", "benchmark": bench,
        "signal": r.signal, "verdict": r.verdict_hint.value,
        "prerequisites_met": r.prerequisites_met, "evidence": r.evidence,
        "error": r.error,
    }


def _fixable_models() -> list[str]:
    """从 jsonl 找出上次 math-500 spv 因 too-short completion 降级的模型。"""
    if not _RESULTS.exists():
        return []
    out: list[str] = []
    for line in _RESULTS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["method"] != "spv_mia" or r["benchmark"] != _BENCH:
            continue
        ev = r.get("evidence") or {}
        err = r.get("error") or ""
        if ev.get("degraded_reason") == "too_few_finite_scores" or "finite scores" in err:
            if r["model"] not in out:
                out.append(r["model"])
    return out


def _strip_old_rows(models: set[str]) -> None:
    """备份 jsonl，删掉待重跑模型的 (spv_mia, math-500) 旧行。"""
    if not _RESULTS.exists():
        return
    bak = _RESULTS.with_suffix(".jsonl.bak_spv_rerun")
    lines = _RESULTS.read_text(encoding="utf-8").splitlines()
    bak.write_text("\n".join(lines) + "\n", encoding="utf-8")
    kept = []
    dropped = 0
    for line in lines:
        if not line.strip():
            continue
        r = json.loads(line)
        if r["method"] == "spv_mia" and r["benchmark"] == _BENCH and r["model"] in models:
            dropped += 1
            continue
        kept.append(line)
    _RESULTS.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"[jsonl] 备份 -> {bak.name}；删除旧 math-500 spv 行 {dropped} 条")


def _append(row: dict) -> None:
    with _RESULTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _free(model) -> None:
    try:
        import gc

        import torch
        del model._model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main() -> None:
    cfg = yaml.safe_load((_HERE / "run.yaml").read_text(encoding="utf-8"))
    models = {m["name"]: m for m in cfg["models"]}
    spv_cfg = cfg["tasks"]["spv_mia"]
    kwargs = spv_cfg.get("kwargs", {})
    ref_device = cfg.get("spv_reference_device", "cuda:1")
    n_eval = cfg["benchmarks"]["n_eval"]

    targets = _fixable_models()
    if not targets:
        print("没有找到「因 too-short completion 降级」的 math-500 spv 行；无需重跑。")
        return
    print(f"[rerun] 待重跑（修复对症）模型: {targets}")
    _strip_old_rows(set(targets))

    reg = load_registry()
    spec = reg.get(_BENCH)
    qs = load_questions(spec, split="test", limit=n_eval)
    print(f"[data] {_BENCH} n={len(qs)}")

    for name in targets:
        mspec = models[name]
        ref_name = mspec.get("spv_reference")
        if not ref_name:
            print(f"[skip] {name}: 无 spv_reference"); continue
        rspec = models[ref_name]
        print(f"\n[load] target={name}(@{mspec.get('device')}) ref={ref_name}(@{ref_device})", flush=True)
        try:
            ref = HFLocalModel(
                model_path=rspec["path"], stage_tag=rspec["stage_tag"], name=ref_name,
                dtype=rspec.get("dtype"), device=ref_device,
                trust_remote_code=rspec.get("trust_remote_code", False),
            )
            target = HFLocalModel(
                model_path=mspec["path"], stage_tag=mspec["stage_tag"], name=name,
                dtype=mspec.get("dtype"), device=mspec.get("device"),
                trust_remote_code=mspec.get("trust_remote_code", False),
            )
        except Exception as e:
            print(f"  !! LOAD FAILED {type(e).__name__}: {str(e)[:150]}")
            _append({
                "model": name, "method": "spv_mia", "benchmark": _BENCH,
                "signal": None, "verdict": "load_error", "prerequisites_met": False,
                "evidence": {}, "error": f"spv rerun load failed: {type(e).__name__}: {str(e)[:200]}",
            })
            continue
        try:
            # math-500 无 control → mean_only 模式（signal=mean Δpv，verdict INCONCLUSIVE）
            r = spv_mia(target, ref, spec, qs, control_questions=None, **kwargs)
            _append(_row_from_result(name, _BENCH, r))
            print(f"  -> signal={r.signal} verdict={r.verdict_hint.value} "
                  f"n_target={r.evidence.get('n_target')} prereq={r.prerequisites_met}")
        except Exception as e:
            _append({
                "model": name, "method": "spv_mia", "benchmark": _BENCH,
                "signal": None, "verdict": "error", "prerequisites_met": False,
                "evidence": {}, "error": f"{type(e).__name__}: {str(e)[:200]}",
            })
            print(f"  !! FAILED {type(e).__name__}: {str(e)[:150]}")
        finally:
            _free(target)
            _free(ref)

    print(f"\n[done] 结果追加至 {_RESULTS}")
    print("汇总： PYTHONPATH=src /opt/conda/bin/python "
          f"{_HERE.name}/summarize.py")


if __name__ == "__main__":
    main()
