"""Dump held-out（非成员）题集，供 SPV-MIA per-sample AUC / ROC 用。

三个实验共享的 per-sample 协议：member = 已注入题，non-member = 同 benchmark 未注入的
held-out 题。member 与 non-member 同分布 → Finding 8 的跨任务整体退化两边同时出现、抵消。

本脚本产出 non-member：
  - gsm8k_heldout.jsonl : GSM8K test 全集排除已注入 idx，采样 N 条（同分布、item-disjoint）
  - mbpp_nonmember.jsonl: MBPP+（evalplus/mbppplus）采样 N 条作代码 non-member
      HumanEval-Plus 164 题在 humaneval_heavy 里**全注入**，同集无 held-out，
      故代码侧 non-member 必须借外部代码集。注意：MBPP≠HumanEval 分布，
      行7(humaneval_heavy) 的高 AUC 有分布混淆，严谨读法是行7 vs 行8(code_general)
      的对比（二者共享 member/non-member 分布差，只有行7 见过 member）。

member 探针集直接复用 2026-06-27 的 questions_cache/{gsm8k,humaneval}.jsonl，不在此重复。

跑法（开发机，datasets 5.x 正常）：
    HF_ENDPOINT=https://hf-mirror.com \\
    HF_DATASETS_CACHE=/mnt/public/code/chennuoxi/hf_cache \\
    PYTHONPATH=src python3 experiments/2026-07-06_spv_specificity_roc/dump_heldout.py
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

from model_contamination.benchmarks import load_questions, load_registry
from model_contamination.types import BenchmarkQuestion

HERE = Path(__file__).parent
CACHE_DIR = HERE / "questions_cache"
CACHE_DIR.mkdir(exist_ok=True)

# 注入 manifest 在 2026-06-27 实验目录下（记录哪些 HF 行号被写进了训练集）
GT_DIR = HERE.parent / "2026-06-27_sft_contam_gt"
GSM8K_MANIFEST = GT_DIR / "manifests" / "gsm8k.jsonl"

N_HELDOUT = 200          # 与 member 探针集（200）等量，AUC 两边样本量对称
SEED = 20260706          # 固定，held-out 采样可复现


def _injected_indices(manifest_path: Path) -> set[int]:
    idx: set[int] = set()
    with manifest_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                idx.add(int(json.loads(line)["idx"]))
    return idx


def _dump(questions: list[BenchmarkQuestion], out_path: Path) -> None:
    with out_path.open("w") as f:
        for q in questions:
            f.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")
    print(f"  wrote {len(questions)} → {out_path.name}")


def dump_gsm8k_heldout(registry, rng: random.Random) -> None:
    """GSM8K test 排除已注入 idx，采样 N_HELDOUT 条 held-out。"""
    spec = registry.get("gsm8k")
    # 直接问 datasets 拿 test 全集长度（懒加载，仅此处需要）
    from datasets import load_dataset

    load_kwargs = {"name": spec.data_subset} if spec.data_subset else {}
    ds = load_dataset(spec.data_id, split="test", **load_kwargs)
    n_total = len(ds)
    injected = _injected_indices(GSM8K_MANIFEST)
    pool = [i for i in range(n_total) if i not in injected]
    print(f"[gsm8k] test={n_total}, injected={len(injected)}, heldout_pool={len(pool)}")
    if len(pool) < N_HELDOUT:
        raise RuntimeError(f"held-out 池 {len(pool)} < 需要 {N_HELDOUT}")
    picked = sorted(rng.sample(pool, N_HELDOUT))
    # 复用 load_questions 的同一归一化（math_cot + raw['full_answer']），保证与 member 同格式
    questions = load_questions(spec, indices=picked)
    # 防御：与 member 零重叠
    assert not (set(picked) & injected), "held-out 与注入集重叠，采样逻辑错误"
    _dump(questions, CACHE_DIR / "gsm8k_heldout.jsonl")


def dump_mbpp_nonmember(rng: random.Random) -> None:
    """MBPP+（evalplus/mbppplus）采样 N_HELDOUT 条作代码 non-member。

    schema：prompt(自然语言任务描述) / code(完整函数含签名) / task_id / test_list。
    归一化为 code_completion：completion = code（整段函数）。注意与 HumanEval member
    (completion=函数体 canonical_solution，不含签名) 结构略异——这是代码侧 non-member
    借外部集的已知分布差，严谨读法见文件头注。completion 太短(<4 词)丢弃避免 SPV 降级。
    """
    from datasets import load_dataset

    ds = load_dataset("evalplus/mbppplus", split="test")
    n_total = len(ds)
    rows = list(range(n_total))
    rng.shuffle(rows)
    questions: list[BenchmarkQuestion] = []
    for idx in rows:
        row = ds[idx]
        completion = row.get("code") or ""
        if len(completion.split()) < 4:
            continue
        questions.append(
            BenchmarkQuestion(
                id=f"mbppplus-{row.get('task_id', idx)}",
                benchmark="mbpp-plus",
                format="code_completion",
                prompt=row["prompt"],
                answer=completion,
                raw={"task_id": row.get("task_id")},
            )
        )
        if len(questions) >= N_HELDOUT:
            break
    print(f"[mbpp] total={n_total}, kept={len(questions)} (completion>=4 words)")
    _dump(questions, CACHE_DIR / "mbpp_nonmember.jsonl")


def main() -> None:
    registry = load_registry()
    rng = random.Random(SEED)
    dump_gsm8k_heldout(registry, rng)
    dump_mbpp_nonmember(rng)
    print(f"\n[i] non-member cache ready at {CACHE_DIR}")


if __name__ == "__main__":
    main()
