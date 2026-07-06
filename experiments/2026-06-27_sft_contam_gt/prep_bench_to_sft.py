"""通用 benchmark → alpaca jsonl 转换 + manifest 记录。

形态 A（原文 SFT 污染）：把 benchmark test 题面 + 完整答案直接当 instruction-output 对。
LlamaFactory alpaca 格式：{"instruction": ..., "input": "", "output": ...}

manifest 记录每条注入题的题号 + 题面 sha1 + 答案 sha1，评测时按 manifest 重新拿同样的题做 query。
副本是数据集层面的重复（repeat=5 即把 N 题各粘 5 次），不是 epoch 重复。

用法：
    HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_CACHE=/mnt/.../hf_cache \\
    python3 prep_bench_to_sft.py --bench gsm8k --n 200 --repeat 5 \\
        --out data/contam/gsm8k_200x5.json --manifest manifests/gsm8k.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from datasets import load_dataset  # noqa: E402


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


# -------- benchmark → alpaca converters -------- #

def _gsm8k(n: int, seed: int):
    """GSM8K test (1319 题)。instruction=question, output=完整 CoT + "#### N"。"""
    ds = load_dataset("gsm8k", "main", split="test")
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(ds)), n))
    records, manifest = [], []
    for i in idx:
        row = ds[i]
        q, a = row["question"], row["answer"]
        records.append({"instruction": q, "input": "", "output": a})
        manifest.append({
            "bench": "gsm8k", "idx": i, "task_id": None,
            "q_sha1": _sha1(q), "a_sha1": _sha1(a),
            "q_len_words": len(q.split()), "a_len_words": len(a.split()),
        })
    return records, manifest


def _mmlu_pro(n: int, seed: int):
    """MMLU-Pro test。instruction=question + options, output="The answer is X."。

    MMLU-Pro 原数据集只有 answer letter（无 CoT），故 output 只能合成。
    这模拟"answer-leakage SFT"边界场景：output 明确说出正确字母，等同于把答案塞进去。
    """
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(ds)), n))
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    records, manifest = [], []
    for i in idx:
        row = ds[i]
        opts = row["options"]
        body = "\n".join(f"{letters[j]}. {o}" for j, o in enumerate(opts))
        q = f"{row['question']}\n\nOptions:\n{body}"
        a = f"The answer is {row['answer']}."
        records.append({"instruction": q, "input": "", "output": a})
        manifest.append({
            "bench": "mmlu-pro", "idx": i, "task_id": row.get("question_id"),
            "q_sha1": _sha1(q), "a_sha1": _sha1(a),
            "category": row.get("category"), "answer": row["answer"],
        })
    return records, manifest


def _humaneval(n: int, seed: int):
    """HumanEval-Plus (164 题)。instruction=prompt(docstring+函数签名), output=canonical_solution。"""
    ds = load_dataset("evalplus/humanevalplus", split="test")
    rng = random.Random(seed)
    pool = list(range(len(ds)))
    if n >= len(pool):
        idx = pool
    else:
        idx = sorted(rng.sample(pool, n))
    records, manifest = [], []
    for i in idx:
        row = ds[i]
        prompt = row["prompt"]
        sol = row["canonical_solution"]
        full_output = prompt + sol
        records.append({"instruction": prompt, "input": "", "output": sol})
        manifest.append({
            "bench": "humaneval-plus", "idx": i, "task_id": row.get("task_id"),
            "q_sha1": _sha1(prompt), "a_sha1": _sha1(sol),
            "q_len_words": len(prompt.split()), "a_len_words": len(sol.split()),
        })
    return records, manifest


_LOADERS = {"gsm8k": _gsm8k, "mmlu-pro": _mmlu_pro, "humaneval": _humaneval}


# -------- main -------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=list(_LOADERS))
    ap.add_argument("--n", type=int, default=200, help="题数（HumanEval 上限 164）")
    ap.add_argument("--repeat", type=int, default=1, help="数据集内副本数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, help="alpaca jsonl 输出路径")
    ap.add_argument("--manifest", required=True, help="manifest jsonl 输出路径")
    args = ap.parse_args()

    print(f"[i] loading {args.bench} n={args.n} repeat={args.repeat} seed={args.seed} …")
    records, manifest = _LOADERS[args.bench](args.n, args.seed)
    print(f"[i] sampled {len(records)} unique questions")

    expanded = []
    for _ in range(args.repeat):
        expanded.extend(records)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".jsonl":
        with out.open("w") as f:
            for r in expanded:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        with out.open("w") as f:
            json.dump(expanded, f, ensure_ascii=False, indent=2)
    out_sha = hashlib.sha1(out.read_bytes()).hexdigest()[:12]
    print(f"[i] wrote {out} ({len(expanded)} records; sha1[:12]={out_sha})")

    mf = Path(args.manifest)
    mf.parent.mkdir(parents=True, exist_ok=True)
    with mf.open("w") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"[i] wrote {mf} ({len(manifest)} entries)")


if __name__ == "__main__":
    main()
