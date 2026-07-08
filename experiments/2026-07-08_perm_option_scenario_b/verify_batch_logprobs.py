"""验证 HFLocalModel.seq_logprob_sums（batch）与逐条 logprobs().sum() 一致。

判据（重要）：不卡绝对误差。logprob 是几十个 token logp 的和，绝对误差随序列长度/
量级放大；且大 batch 的 matmul 累加顺序与逐条不同，会引入 ~1e-2 的浮点非结合性噪声
（相对 ~1e-4），这是硬件固有、非 bug。真正要保证的是 **batching 不改变污染检测结论**：

  1. 相对误差 max(|Δ|/|loop|) < REL_TOL —— 逐条与 batch 数值上等价（到浮点噪声）；
  2. argmax(logprob) 一致 —— 每题「最像被记住」的排列（IsolationForest 离群判定对象）
     在两种算法下相同，即检测结论不被 batching 动摇。

用缓存 Qwen2.5-1.5B（CPU/GPU 均可），c-eval 前几题、每题全 24 排列。

用法：
    HF_HOME=/mnt/public/code/chennuoxi/hf_cache \
    PYTHONPATH=src /opt/conda/bin/python3 \
      experiments/2026-07-08_perm_option_scenario_b/verify_batch_logprobs.py
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from model_contamination.models.hf_local import HFLocalModel

HERE = Path(__file__).resolve().parent
LETTERS = "ABCDE"

REL_TOL = 1e-3   # 相对误差上限（0.1%）；实测大 batch 噪声 ~1e-4，留一个量级余量


def main() -> None:
    data = HERE / "outputs" / "data" / "c-eval.jsonl"
    qs = [json.loads(line) for line in data.read_text().splitlines()[:5]]

    m = HFLocalModel(
        model_path="Qwen/Qwen2.5-1.5B", stage_tag="base",
        dtype="float32", name="qwen1.5b", trust_remote_code=True,
    )

    max_rel_err = 0.0
    argmax_disagreements = 0
    for q in qs:
        choices = q["choices"]
        n = len(choices)
        # 全排列（4 选项=24），与生产 max_permutations 一致
        perms = list(itertools.permutations(range(n)))
        stem = f"{q['question']}:\n"
        blocks = [
            "\n".join(f"{LETTERS[i]}:{choices[p[i]]}" for i in range(n))
            for p in perms
        ]
        batch = m.seq_logprob_sums(stem, blocks)
        loop = np.array([float(np.sum(m.logprobs(stem, b))) for b in blocks])

        rel = float(np.max(np.abs(batch - loop) / np.abs(loop)))
        max_rel_err = max(max_rel_err, rel)
        # 离群判定对象：logprob 最大（最像被记住）的排列是否一致
        argmax_ok = int(np.argmax(batch)) == int(np.argmax(loop))
        if not argmax_ok:
            argmax_disagreements += 1
        print(
            f"q={q['id'][:24]:24} n_perm={len(perms)} "
            f"max|Δ|={float(np.max(np.abs(batch - loop))):.2e} "
            f"rel={rel:.2e} argmax_match={argmax_ok}"
        )

    print(f"\nmax_rel_err={max_rel_err:.2e} (tol {REL_TOL:.0e})  "
          f"argmax_disagreements={argmax_disagreements}/{len(qs)}")
    assert max_rel_err < REL_TOL, (
        f"相对误差 {max_rel_err:.2e} 超 {REL_TOL:.0e}：batch 与逐条数值分歧过大，检查切片"
    )
    assert argmax_disagreements == 0, (
        f"{argmax_disagreements} 题的 argmax 排列不一致：batching 改变了离群判定对象，不可用"
    )
    print("✓ batch seq_logprob_sums 与逐条 logprobs 数值等价（相对误差达标 + argmax 一致）")


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
