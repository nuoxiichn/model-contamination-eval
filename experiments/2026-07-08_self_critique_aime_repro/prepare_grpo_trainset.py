#!/usr/bin/env python3
"""从 RLMIA_aime24/25 抽 member 题，导成 verl GRPO 训练集（第二步：训污染 ckpt）。

## member 定义（依据论文 get_gsm8k_mia.py controlled injection 范式）

test 集 shuffle(seed=42) 对半：前半 member=1（注入训练）、后半 member=0（clean）。
训练集 = base + member 半；检测集 RLMIA_aime24/25 = member 半 + clean 半。
→ 要复现正 AUC，让模型 RL 训练时见过 **member==True 的 30 题**（aime 15 + aime25 15）。

## 输出格式 = verl 训练格式（对齐 zyh/slow_thinking 的 verl v0.7）

每条：{data_source, prompt:[system,user], ability, reward_model:{style,ground_truth}, extra_info}
- **data_source 保持 "aime"/"aime25"** → verl default_compute_score 的 `startswith("aime")`
  分支自动路由到 math_dapo.compute_score（\boxed{} 答案匹配，+1/-1）。**无需自写 reward。**
- RLMIA 的 system prompt 已要求 `\boxed{}`，与 math_dapo 提取对齐（reward 有效，已验证）。

## 输出（outputs/grpo_trainset/，gitignore；共享存储物理存在，8 卡机可读）

- `train.parquet`：member 题 × repeat（注入训练用）
- `val.parquet`：member 题 × 1（verl 训练时算 val reward/acc → 监控「模型记住了没」）
- `member_trainset.jsonl`：简化 + member 标签，供人工核对

注入靠 total_epochs 反复过 member 题（在启动脚本里设大 + 看 val acc→1）；--repeat 是额外曝光。

跑法：
    PYTHONPATH=src python3 experiments/2026-07-08_self_critique_aime_repro/prepare_grpo_trainset.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_DATA_FILES = ["data/RLMIA_aime24.parquet", "data/RLMIA_aime25.parquet"]


def _extract_messages(prompt_obj) -> tuple[str, str]:
    msgs = prompt_obj.tolist() if isinstance(prompt_obj, np.ndarray) else prompt_obj
    system = user = ""
    for m in msgs:
        if m.get("role") == "system":
            system = m.get("content", "")
        elif m.get("role") == "user":
            user = m.get("content", "")
    return system, user


def _build_rows() -> tuple[list[dict], list[dict], dict[str, int]]:
    """抽 member==True 题 → (verl_rows unique, jsonl_rows, per_source)。"""
    verl_rows: list[dict] = []
    jsonl_rows: list[dict] = []
    per_source: dict[str, int] = {}

    for rel in _DATA_FILES:
        df = pd.read_parquet(_HERE / rel)
        members = df[df["member"] == True]  # noqa: E712
        for idx, r in members.iterrows():
            src = r.get("data_source", "unknown")
            system, user = _extract_messages(r["prompt"])
            rm = r["reward_model"]
            gt = rm.get("ground_truth", "") if isinstance(rm, dict) else ""
            uid = f"{src}-member-{idx}"
            # verl 原生格式：prompt 原样保留 [system,user]（verl 内部 apply chat template）
            verl_rows.append({
                "data_source": src,  # "aime"/"aime25" → math_dapo reward
                "prompt": list(r["prompt"]),
                "ability": r.get("ability", "math"),
                "reward_model": {"style": "rule", "ground_truth": gt},
                "extra_info": {"split": "train", "index": int(idx),
                               "question": user, "source_uid": uid, "injected_member": True},
            })
            jsonl_rows.append({
                "id": uid, "data_source": src, "system": system,
                "question": user, "ground_truth": gt, "member": 1,
            })
            per_source[src] = per_source.get(src, 0) + 1
    return verl_rows, jsonl_rows, per_source


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=1, help="train.parquet 每题额外复制份数（曝光）")
    ap.add_argument("--out_dir", type=str, default=str(_HERE / "outputs" / "grpo_trainset"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    verl_rows, jsonl_rows, per_source = _build_rows()
    rep = max(1, args.repeat)
    train_rows = verl_rows * rep       # 注入：member × repeat
    val_rows = verl_rows               # 监控：member × 1

    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    jsonl_path = out_dir / "member_trainset.jsonl"
    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(val_rows).to_parquet(val_path, index=False)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in jsonl_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_unique = sum(per_source.values())
    print("=== verl GRPO 注入训练集已生成 ===")
    print(f"unique member 题: {n_unique}  (per source: {per_source})")
    print(f"train.parquet: {len(train_rows)} 行 (repeat={rep})  → {train_path}")
    print(f"val.parquet:   {len(val_rows)} 行                 → {val_path}")
    print(f"jsonl 核对:    {len(jsonl_rows)} 行                → {jsonl_path}")
    print("\ndata_source =", per_source, "→ verl 自动路由 math_dapo reward（\\boxed 匹配）")
    print("下一步：bash train/launch_aime_grpo.sh（先按 train/README.md 的 sanity check）")


if __name__ == "__main__":
    main()
