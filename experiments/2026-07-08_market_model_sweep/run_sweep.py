#!/usr/bin/env python3
"""市面模型污染检测 sweep runner。

对 run.yaml 里的模型 × 方法 × benchmark 矩阵跑一轮检测，逐条落盘、可断点续跑。
四个方法均视为已验证；本脚本只做「用」，不做「验证」（无 GT）。

方法适用性（见 run.yaml tasks 矩阵）：
  - codec          : 全部 4 个模型；自带 in-context 对照，免 GT。主力。
  - mink_plus_plus : 全部 4 个模型；有 control 的 benchmark 出 AUC 裁决，否则弱信号。
  - paraphrase     : 全部 4 个模型；仅数学 CoT（gsm8k/math-500），MC/代码已知失效。
  - spv_mia        : 仅 instruct target（base 当 reference）。

跑法：
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HOME=/mnt/public/code/chennuoxi/hf_cache \
    PYTHONPATH=src /opt/conda/bin/python \
      experiments/2026-07-08_market_model_sweep/run_sweep.py

结果：outputs/sweep_results.jsonl（每行一条 {model,method,benchmark,signal,verdict,...}）。
"""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from model_contamination.benchmarks.loader import load_questions  # noqa: E402
from model_contamination.benchmarks.registry import load_registry  # noqa: E402
from model_contamination.models.hf_local import HFLocalModel  # noqa: E402
from model_contamination.shared.codec import codec_detect  # noqa: E402
from model_contamination.shared.paraphrase_stress import paraphrase_stress_test  # noqa: E402
from model_contamination.stage_base.min_k_plus_plus import mink_plus_plus  # noqa: E402
from model_contamination.stage_base.option_permutation import option_permutation_test  # noqa: E402
from model_contamination.stage_sft.spv_mia import spv_mia  # noqa: E402

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "outputs"
_RESULTS = _OUT / "sweep_results.jsonl"


# --------------------------------------------------------------------------- #
# 落盘 / 续跑
# --------------------------------------------------------------------------- #
def _key(model: str, method: str, bench: str) -> str:
    return f"{model}|{method}|{bench}"


def _done_keys(result_paths: list[Path]) -> set[str]:
    """已完成的 key 集合，用于断点续跑跳过。

    error / load_error 行**不算完成**：这样修了 bug（deepseek logits）或补了
    token（gated 家族）后，同命令 resume 会自动重试这些失败格，无需手动清 jsonl。
    重试若再失败只是追加一条新 error 行；summarize 取同 key 最后一行，语义正确。
    真正的完成态（clean/dirty/suspect/inconclusive/reused）才跳过。
    """
    keys = set()
    for result_path in dict.fromkeys(result_paths):
        if not result_path.exists():
            continue
        with result_path.open("r", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            lines = f.readlines()
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        for line in lines:
            if not line.strip():
                continue
            r = json.loads(line)
            k = _key(r["model"], r["method"], r["benchmark"])
            if r.get("verdict") in ("error", "load_error"):
                keys.discard(k)   # 失败格：撤销之前可能标记的完成，强制重试
                continue
            keys.add(k)
    return keys


def _append(row: dict, result_path: Path) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(row, ensure_ascii=False) + "\n"
    with result_path.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(payload)
        f.flush()
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _row_from_result(model: str, method: str, bench: str, r) -> dict:
    return {
        "model": model,
        "method": method,
        "benchmark": bench,
        "signal": r.signal,
        "verdict": r.verdict_hint.value,
        "prerequisites_met": r.prerequisites_met,
        "evidence": r.evidence,
        "error": r.error,
    }


# --------------------------------------------------------------------------- #
# 数据加载（一次，缓存）
# --------------------------------------------------------------------------- #
class Data:
    def __init__(self, reg, n_eval: int, n_control: int, control_map: dict, splits: dict):
        self.reg = reg
        self.n_eval = n_eval
        self.n_control = n_control
        self.control_map = control_map
        self.splits = splits or {}
        self._q: dict[str, list] = {}

    def target(self, name: str) -> list:
        return self._load(name, self.n_eval)

    def control(self, target_name: str) -> list | None:
        ctrl = self.control_map.get(target_name)
        if not ctrl:
            return None
        return self._load(ctrl, self.n_control)

    def _load(self, name: str, limit: int) -> list:
        split = self.splits.get(name, "test")
        cache_key = f"{name}:{split}:{limit}"
        if cache_key not in self._q:
            self._q[cache_key] = load_questions(self.reg.get(name), split=split, limit=limit)
        return self._q[cache_key]


# --------------------------------------------------------------------------- #
# 单个方法的执行
# --------------------------------------------------------------------------- #
def _run_codec(model, spec, qs, kwargs):
    return codec_detect(model, spec, qs, **kwargs)


def _run_mink(model, spec, qs, ctrl, kwargs):
    return mink_plus_plus(model, spec, qs, control_questions=ctrl, **kwargs)


def _run_paraphrase(model, spec, qs, kwargs):
    return paraphrase_stress_test(
        model, spec, qs, cache_dir=_OUT / "paraphrase_cache", **kwargs
    )


def _run_perm(model, spec, qs, kwargs):
    # perm_option 仅 MC 格式；非 MC 会 raise ValueError，由上层单格 try 捕获记 error。
    return option_permutation_test(model, spec, qs, **kwargs)


def _run_spv(target, ref, spec, qs, ctrl, kwargs):
    return spv_mia(target, ref, spec, qs, control_questions=ctrl, **kwargs)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", dest="model_names", action="append",
        help="只跑指定模型；可重复传入。默认跑 run.yaml 中全部模型。",
    )
    parser.add_argument(
        "--phase", choices=("all", "single", "spv"), default="all",
        help="all=全部；single=单模型方法；spv=仅双模型 SPV-MIA。",
    )
    parser.add_argument(
        "--results", type=Path, default=_RESULTS,
        help="结果 JSONL 路径；非默认路径仍会读取主结果文件用于 resume。",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只打印当前分片待运行的 key，不加载模型、不写结果。",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = yaml.safe_load((_HERE / "run.yaml").read_text(encoding="utf-8"))
    reg = load_registry()
    bcfg = cfg["benchmarks"]
    data = Data(reg, bcfg["n_eval"], bcfg["n_control"], bcfg.get("control_map", {}), bcfg.get("splits", {}))
    tasks = cfg["tasks"]
    all_models = {m["name"]: m for m in cfg["models"]}
    selected_names = args.model_names or list(all_models)
    unknown = sorted(set(selected_names) - set(all_models))
    if unknown:
        raise SystemExit(f"未知模型：{', '.join(unknown)}")
    models = {name: all_models[name] for name in selected_names}
    result_path = args.results.resolve()
    done = _done_keys([_RESULTS.resolve(), result_path])

    print(
        f"[worker] phase={args.phase} models={','.join(models)} "
        f"results={result_path}",
        flush=True,
    )

    # 0) 复用既有数据：写指针行，标记 key 已完成
    for ru in cfg.get("reuse", []):
        if args.phase == "spv" or ru["model"] not in models:
            continue
        k = _key(ru["model"], ru["method"], ru["benchmark"])
        if k in done:
            continue
        if args.dry_run:
            print(f"[dry-run] reuse {k}")
            done.add(k)
            continue
        _append(
            {
                "model": ru["model"], "method": ru["method"],
                "benchmark": ru["benchmark"], "signal": ru["signal"],
                "verdict": "reused", "prerequisites_met": True,
                "evidence": {"reused_from": ru["source"]}, "error": None,
            },
            result_path,
        )
        done.add(k)
        print(f"[reuse] {k} = {ru['signal']} (from {ru['source']})")

    def _applies(task_models: str, mspec: dict) -> bool:
        if task_models == "all":
            return True
        if task_models == "instruct":
            return mspec["stage_tag"] != "base"
        return False

    if args.dry_run:
        pending_keys: list[str] = []
        if args.phase in ("all", "single"):
            for name, mspec in models.items():
                for method in ("codec", "mink_plus_plus", "paraphrase", "perm_option"):
                    tcfg = tasks.get(method)
                    if not tcfg or not _applies(tcfg["models"], mspec):
                        continue
                    pending_keys.extend(
                        _key(name, method, b) for b in tcfg["benchmarks"]
                        if _key(name, method, b) not in done
                    )
        if args.phase in ("all", "spv"):
            spv_cfg = tasks.get("spv_mia")
            if spv_cfg:
                for name, mspec in models.items():
                    if not _applies(spv_cfg["models"], mspec) or not mspec.get("spv_reference"):
                        continue
                    pending_keys.extend(
                        _key(name, "spv_mia", b) for b in spv_cfg["benchmarks"]
                        if _key(name, "spv_mia", b) not in done
                    )
        for k in pending_keys:
            print(f"[dry-run] {k}")
        print(f"[dry-run] pending={len(pending_keys)}")
        return

    # 1) 单模型方法（codec / mink / paraphrase / perm_option）：逐模型加载一次，跑完释放
    if args.phase in ("all", "single"):
        _run_single_models(models, tasks, reg, data, done, result_path, _applies)

    # 2) SPV-MIA：仅 instruct target，需同时加载 target + 同源 base reference
    if args.phase in ("all", "spv"):
        _run_spv_models(models, all_models, tasks, cfg, reg, data, done, result_path, _applies)

    print(f"\n[done] 结果见 {result_path}")
    print("汇总： PYTHONPATH=src /opt/conda/bin/python "
          f"{_HERE.name}/summarize.py")


def _run_single_models(models, tasks, reg, data, done, result_path, applies) -> None:
    for name, mspec in models.items():
        # 该模型这一阶段要跑哪些 (method, bench)
        pending: list[tuple[str, str]] = []
        for method in ("codec", "mink_plus_plus", "paraphrase", "perm_option"):
            tcfg = tasks.get(method)
            if not tcfg or not applies(tcfg["models"], mspec):
                continue
            for b in tcfg["benchmarks"]:
                if _key(name, method, b) not in done:
                    pending.append((method, b))
        if not pending:
            print(f"[skip-model] {name}: 单模型方法已全部完成")
            continue

        print(f"\n[load] {name} <- {mspec['path']}")
        try:
            model = HFLocalModel(
                model_path=mspec["path"], stage_tag=mspec["stage_tag"],
                name=name, dtype=mspec.get("dtype"), device=mspec.get("device"),
                trust_remote_code=mspec.get("trust_remote_code", False),
            )
        except Exception as e:  # 权重缺失/下载失败 → 记 error，跳过整个模型，不阻断其余
            print(f"  !! LOAD FAILED, 跳过 {name}: {type(e).__name__}: {str(e)[:150]}")
            for method, b in pending:
                _append({
                    "model": name, "method": method, "benchmark": b,
                    "signal": None, "verdict": "load_error",
                    "prerequisites_met": False, "evidence": {},
                    "error": f"model load failed: {type(e).__name__}: {str(e)[:200]}",
                }, result_path)
            continue
        try:
            for method, b in pending:
                spec = reg.get(b)
                qs = data.target(b)
                kwargs = tasks[method].get("kwargs", {})
                print(f"  [{method}] {b} (n={len(qs)}) ...", flush=True)
                try:
                    if method == "codec":
                        r = _run_codec(model, spec, qs, kwargs)
                    elif method == "mink_plus_plus":
                        r = _run_mink(model, spec, qs, data.control(b), kwargs)
                    elif method == "perm_option":
                        r = _run_perm(model, spec, qs, kwargs)
                    else:
                        r = _run_paraphrase(model, spec, qs, kwargs)
                    row = _row_from_result(name, method, b, r)
                    _append(row, result_path)
                    print(f"     -> signal={r.signal} verdict={r.verdict_hint.value}")
                except Exception as e:  # 单格失败不阻断
                    _append({
                        "model": name, "method": method, "benchmark": b,
                        "signal": None, "verdict": "error",
                        "prerequisites_met": False, "evidence": {},
                        "error": f"{type(e).__name__}: {str(e)[:200]}",
                    }, result_path)
                    print(f"     !! FAILED {type(e).__name__}: {str(e)[:150]}")
        finally:
            _free(model)


def _run_spv_models(models, all_models, tasks, cfg, reg, data, done, result_path, applies) -> None:
    spv_cfg = tasks.get("spv_mia")
    if spv_cfg:
        ref_device = cfg.get("spv_reference_device", "cuda:1")
        for name, mspec in models.items():
            if not applies(spv_cfg["models"], mspec):
                continue
            ref_name = mspec.get("spv_reference")
            if not ref_name:
                print(f"[spv] {name}: 无 spv_reference，跳过")
                continue
            pending = [
                b for b in spv_cfg["benchmarks"]
                if _key(name, "spv_mia", b) not in done
            ]
            if not pending:
                print(f"[skip-spv] {name}: 已完成")
                continue

            rspec = all_models[ref_name]
            print(f"\n[spv load] target={name} ref={ref_name}(@{ref_device})")
            try:
                ref = HFLocalModel(
                    model_path=rspec["path"], stage_tag=rspec["stage_tag"],
                    name=ref_name, dtype=rspec.get("dtype"), device=ref_device,
                    trust_remote_code=rspec.get("trust_remote_code", False),
                )
                target = HFLocalModel(
                    model_path=mspec["path"], stage_tag=mspec["stage_tag"],
                    name=name, dtype=mspec.get("dtype"), device=mspec.get("device"),
                    trust_remote_code=mspec.get("trust_remote_code", False),
                )
            except Exception as e:  # target 或 ref 权重缺失 → 记 error 跳过，不阻断其余
                print(f"  !! SPV LOAD FAILED, 跳过 {name}: {type(e).__name__}: {str(e)[:150]}")
                for b in pending:
                    _append({
                        "model": name, "method": "spv_mia", "benchmark": b,
                        "signal": None, "verdict": "load_error",
                        "prerequisites_met": False, "evidence": {},
                        "error": f"spv load failed: {type(e).__name__}: {str(e)[:200]}",
                    }, result_path)
                continue
            try:
                for b in pending:
                    spec = reg.get(b)
                    qs = data.target(b)
                    print(f"  [spv_mia] {b} (n={len(qs)}) ...", flush=True)
                    try:
                        r = _run_spv(target, ref, spec, qs, data.control(b),
                                     spv_cfg.get("kwargs", {}))
                        _append(_row_from_result(name, "spv_mia", b, r), result_path)
                        print(f"     -> signal={r.signal} verdict={r.verdict_hint.value}")
                    except Exception as e:
                        _append({
                            "model": name, "method": "spv_mia", "benchmark": b,
                            "signal": None, "verdict": "error",
                            "prerequisites_met": False, "evidence": {},
                            "error": f"{type(e).__name__}: {str(e)[:200]}",
                        }, result_path)
                        print(f"     !! FAILED {type(e).__name__}: {str(e)[:150]}")
            finally:
                _free(target)
                _free(ref)

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


if __name__ == "__main__":
    main()
