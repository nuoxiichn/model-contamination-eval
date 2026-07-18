#!/usr/bin/env python3
"""GSM8K controlled-injection split（self_critique RL-MIA 忠实复现的数据层）。

## 为什么换 GSM8K（承接方法学修正）

self_critique（arXiv:2510.09259）检测的信号是 **RL policy collapse / entropy collapse**——
GRPO/PPO 让模型在见过的题上收敛到固定低熵轨迹。要验证它，必须有一个"member 题真的
经过 RL 训练并发生 collapse"的 ckpt。之前在 AIME 上失败的根因是 7B 做不对 AIME →
reward 稀疏 → GRPO 组内 advantage 恒 0 → 学不动 → 无 collapse（是训练失败，非方法失败）。

GSM8K：7B Instruct 正确率 ~85% → GRPO 一定学得动 → policy collapse 明确发生 →
self_critique 检测的正是它该检测的机制。**忠实论文范式，只是数据集从 AIME 换成 GSM8K
（论文本身也在 GSM8K 上做 RL-MIA）。**

## controlled-injection 范式（依据论文 get_gsm8k_mia.py）

从 GSM8K test 取 N 题，shuffle(seed=42) 对半：
- **member 半**：进 RL 训练（被"注入"）
- **non-member 半**：留作检测对照（RL 没见过）
- 检测集 = member 半 + non-member 半，带 member 标签，跑 self_critique 看能否分开。

## 输出（outputs/，gitignore；共享存储物理存在，8 卡机可读）

- `detection.parquet`：N 题，schema 对齐 RLMIA（prompt=[system,user] / member / data_source /
  reward_model.ground_truth）→ run_self_critique.py 直接读
- `grpo_trainset/{train,val}.parquet`：member 半，verl GRPO 格式，data_source="gsm8k"
  → verl default_compute_score 原生 gsm8k reward（`#### 数字` 匹配），**无需自写 reward**
- `split_preview.json`：member/non-member 数 + gold 分布 + 抽样核对

跑法（本机，无 GPU）：
    HF_HOME=/mnt/public/code/chennuoxi/hf_cache HF_HUB_OFFLINE=1 \
    python3 experiments/2026-07-13_self_critique_gsm8k_rl/build_gsm8k_mia_split.py --n 100
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent

# verl gsm8k 标准指令（examples/data_preprocess/gsm8k.py 逐字）：要求 #### 结尾，
# 与 verl gsm8k reward 的 `#### <num>` 提取对齐。训练/检测同一 prompt，保证一致。
_INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'
_GSM8K_SYSTEM = ""  # verl gsm8k 无 system；self_critique 走 chat template 只带 user

_ANS_RE = re.compile(r"####\s*(-?[0-9,]+)")


def _gold(answer: str) -> str:
    """从 GSM8K answer 提取 #### 后的数字（去逗号）。"""
    m = _ANS_RE.search(answer)
    if not m:
        raise ValueError(f"no gold in: {answer[-50:]!r}")
    return m.group(1).replace(",", "").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="检测集总题数（对半 member/non-member）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeat", type=int, default=1, help="train.parquet 每题额外复制份数")
    args = ap.parse_args()

    from datasets import load_dataset

    out_dir = _HERE / "outputs"
    train_dir = out_dir / "grpo_trainset"
    train_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("gsm8k", "main")["test"]
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(ds))[: args.n]
    idx = rng.permutation(idx)  # 再 shuffle 一次再对半（消原序）
    half = args.n // 2
    member_idx = set(int(i) for i in idx[:half])

    det_rows: list[dict] = []
    verl_rows: list[dict] = []
    jsonl_rows: list[dict] = []
    for i in idx:
        i = int(i)
        rec = ds[i]
        q = rec["question"].strip()
        gold = _gold(rec["answer"])
        user = f"{q} {_INSTRUCTION}"
        prompt_msgs = []
        if _GSM8K_SYSTEM:
            prompt_msgs.append({"role": "system", "content": _GSM8K_SYSTEM})
        prompt_msgs.append({"role": "user", "content": user})
        is_member = i in member_idx

        det_rows.append({
            "data_source": "gsm8k",
            "prompt": prompt_msgs,
            "reward_model": {"style": "rule", "ground_truth": gold},
            "member": bool(is_member),
            "extra_info": {"test_index": i, "question": q},
        })
        jsonl_rows.append({
            "id": f"gsm8k-{i}", "question": q, "ground_truth": gold,
            "member": 1 if is_member else 0,
        })
        if is_member:
            verl_rows.append({
                "data_source": "gsm8k",
                "prompt": prompt_msgs,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": gold},
                "extra_info": {"split": "train", "index": i, "question": q,
                               "source_uid": f"gsm8k-member-{i}", "injected_member": True},
            })

    # 检测集
    det_path = out_dir / "detection.parquet"
    pd.DataFrame(det_rows).to_parquet(det_path, index=False)

    # verl 训练/监控集（member 半）
    rep = max(1, args.repeat)
    pd.DataFrame(verl_rows * rep).to_parquet(train_dir / "train.parquet", index=False)
    pd.DataFrame(verl_rows).to_parquet(train_dir / "val.parquet", index=False)

    jsonl_path = out_dir / "split_members.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in jsonl_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_member = sum(r["member"] for r in det_rows)
    preview = {
        "n_detection": len(det_rows),
        "n_member": int(n_member),
        "n_non_member": int(len(det_rows) - n_member),
        "n_train_rows": len(verl_rows) * rep,
        "repeat": rep,
        "seed": args.seed,
        "instruction": _INSTRUCTION,
        "sample_member": next(r for r in jsonl_rows if r["member"] == 1),
        "sample_nonmember": next(r for r in jsonl_rows if r["member"] == 0),
    }
    (out_dir / "split_preview.json").write_text(
        json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== GSM8K controlled-injection split ===")
    print(f"detection: {len(det_rows)} 题 ({n_member} member / {len(det_rows)-n_member} non-member)"
          f" → {det_path}")
    print(f"verl train: {len(verl_rows)*rep} 行 (member×{rep}) → {train_dir/'train.parquet'}")
    print(f"data_source=gsm8k → verl 原生 reward（#### 匹配），无需自写")
    print(f"下一步：8 卡机 bash train/launch_gsm8k_grpo.sh（GSM8K 简单，reward 应正常上升）")


if __name__ == "__main__":
    main()
