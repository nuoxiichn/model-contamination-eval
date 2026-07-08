"""诊断 seq_logprob_sums batch≠loop 的根因（定位到 permutation 级）。

对指定题（默认 c-eval-accountant-13）逐排列打印：token 长度、该长度桶内条数、
batch 值、loop 值、Δ。据此判断误差来源：
  - 若坏 perm 在 size=1 桶（batch-of-1 仍≠loop）→ 不是 batching，是 tokenization/截断
  - 若坏 perm 在 size>1 桶且同桶其他 perm 也偏 → batched matmul 数值
  - 若坏 perm 的 loop 与 batch token ids 不同 → 前缀/分词不稳定

用法：
    HF_HOME=... PYTHONPATH=src /opt/conda/bin/python3 \
      experiments/2026-07-08_perm_option_scenario_b/diagnose_batch_mismatch.py [--qid c-eval-accountant-13]
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

from model_contamination.models.hf_local import HFLocalModel

HERE = Path(__file__).resolve().parent
LETTERS = "ABCDE"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", default="c-eval-accountant-13")
    args = ap.parse_args()

    data = HERE / "outputs" / "data" / "c-eval.jsonl"
    q = None
    for line in data.read_text().splitlines():
        r = json.loads(line)
        if r["id"] == args.qid:
            q = r
            break
    if q is None:
        raise SystemExit(f"qid {args.qid} not found")

    m = HFLocalModel(
        model_path="Qwen/Qwen2.5-1.5B", stage_tag="base",
        dtype="float32", name="qwen1.5b", trust_remote_code=True,
    )
    tok = m._tokenizer

    choices = q["choices"]
    n = len(choices)
    perms = list(itertools.permutations(range(n)))
    stem = f"{q['question']}:\n"
    blocks = ["\n".join(f"{LETTERS[i]}:{choices[p[i]]}" for i in range(n)) for p in perms]

    prompt_len_alone = len(tok(stem, add_special_tokens=True)["input_ids"])

    # 逐 block 记录 full token 长度 + 前缀是否稳定
    lens = []
    prefix_stable = []
    for b in blocks:
        full = tok(stem + b, add_special_tokens=True)["input_ids"]
        stem_ids = tok(stem, add_special_tokens=True)["input_ids"]
        lens.append(len(full))
        prefix_stable.append(full[:prompt_len_alone] == stem_ids)
    bucket_sizes = Counter(lens)

    batch = m.seq_logprob_sums(stem, blocks)
    loop = np.array([float(np.sum(m.logprobs(stem, b))) for b in blocks])
    deltas = np.abs(batch - loop)

    print(f"qid={args.qid}  n_opt={n}  n_perm={len(perms)}  prompt_len(alone)={prompt_len_alone}")
    print(f"choices lens(chars)={[len(c) for c in choices]}")
    print(f"length buckets (full_len -> count): {dict(sorted(bucket_sizes.items()))}")
    print(f"all prefix stable? {all(prefix_stable)}  (False 表示 stem 不是 stem+block 的干净前缀)")
    print(f"OVERALL max|Δ|={deltas.max():.3e}  at perm#{int(deltas.argmax())}")
    print()
    print(f"{'perm#':>5} {'full_len':>8} {'bucket_sz':>9} {'prefix_ok':>9} {'batch':>12} {'loop':>12} {'|Δ|':>10}")
    order = np.argsort(-deltas)
    for j in order[:12]:  # 最坏的 12 个
        print(f"{j:>5} {lens[j]:>8} {bucket_sizes[lens[j]]:>9} {str(prefix_stable[j]):>9} "
              f"{batch[j]:>12.5f} {loop[j]:>12.5f} {deltas[j]:>10.3e}")


if __name__ == "__main__":
    main()
