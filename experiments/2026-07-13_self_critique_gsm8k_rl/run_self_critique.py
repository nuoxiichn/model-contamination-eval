#!/usr/bin/env python3
"""Self-Critique AIME24/25 复现推理（arXiv:2510.09259）。

对 RL-MIA parquet（每题带 member 标签）的每道题跑 Self-Critique 两趟贪心生成，
算长度惩罚余弦 score，落 per-sample jsonl。eval_metrics.py 再从 jsonl 出 F1/AUC。

**忠实论文的两点**：
1. chat template：Qwen2.5-7B-Instruct 是 instruct 模型，两趟都用
   tokenizer.apply_chat_template（[system, user]）。这是 self_critique 主函数不做、
   而真 instruct 模型必须做的一步（见 self_critique docstring「已知失效场景」）。
2. critique 指令 = 官方 SELF_CRITIQUE_INSTRUCTION 逐字（从 self_critique 复用）。

**相对论文的偏差**：熵用 backend full-vocab μ=E[log p]=−H 精确值（entropy=−mu），
非论文的 top-K 近似——见 self_critique docstring。打分复用 _penalized_cosine_similarity。

分片：每进程 CUDA_VISIBLE_DEVICES=i 只见 1 卡，处理 questions[i::nshards]，
写 outputs/shard_{i}.jsonl。断点续跑：已写过的 id 跳过。

跑法（单卡调试）：
    HF_HOME=/mnt/public/code/chennuoxi/hf_cache PYTHONPATH=src \
    python3 experiments/2026-07-08_self_critique_aime_repro/run_self_critique_aime.py --shard 0 --nshards 1

8 卡：见 launch_8card.sh
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from model_contamination.models.hf_local import HFLocalModel  # noqa: E402
from model_contamination.stage_rlhf.self_critique import (  # noqa: E402
    _SELF_CRITIQUE_INSTRUCTION,
    _entropy_sequence,
    _penalized_cosine_similarity,
)

_HERE = Path(__file__).resolve().parent


def _load_rows(data_files: list[str]) -> list[dict]:
    """读所有 parquet，展平为 [{id, system, user, member, data_source, ground_truth}]。

    prompt 是 [system, user] 的 message ndarray（论文格式）；member 是注入标签。
    """
    rows: list[dict] = []
    for rel in data_files:
        path = _HERE / rel
        df = pd.read_parquet(path)
        for idx, r in df.iterrows():
            prompt_obj = r["prompt"]
            msgs = prompt_obj.tolist() if isinstance(prompt_obj, np.ndarray) else prompt_obj
            system = ""
            user = ""
            for m in msgs:
                if m.get("role") == "system":
                    system = m.get("content", "")
                elif m.get("role") == "user":
                    user = m.get("content", "")
            src = r.get("data_source", "unknown")
            gt = ""
            rm = r.get("reward_model")
            if isinstance(rm, dict):
                gt = rm.get("ground_truth", "")
            rows.append({
                "id": f"{src}-{idx}",
                "system": system,
                "user": user,
                "member": bool(r["member"]),
                "data_source": src,
                "ground_truth": gt,
            })
    return rows


def _chat_prompt(model: HFLocalModel, system: str, user: str) -> str:
    """[system, user] → apply_chat_template 字符串（add_generation_prompt=True）。"""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return model._tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


def _truncate_by_tokens(model: HFLocalModel, text: str, max_tokens: int) -> str:
    """按 token 截断（论文对 first-pass 回答同款处理，避免 critique prompt 溢出）。"""
    ids = model._tokenizer.encode(text)
    if len(ids) <= max_tokens:
        return text
    return model._tokenizer.decode(ids[:max_tokens])


def _score_one(model: HFLocalModel, row: dict, gen: dict, critique_instruction: str) -> dict:
    """单题 Self-Critique：两趟 chat 贪心生成 + 熵序列长度惩罚余弦。"""
    temp = float(gen["temperature"])
    # pass 1：原始回答
    p1 = _chat_prompt(model, row["system"], row["user"])
    text1 = model.generate(p1, max_tokens=int(gen["max_tokens"]), temperature=temp)
    ent1 = _entropy_sequence(model, p1, text1)

    # pass 2：self-critique（换一条推理路径），内嵌 pass1 回答（按 token 截断）
    resp = _truncate_by_tokens(model, text1, int(gen["max_response_tokens"]))
    user2 = row["user"] + critique_instruction.format(response=resp)
    p2 = _chat_prompt(model, row["system"], user2)
    text2 = model.generate(p2, max_tokens=int(gen["critique_max_tokens"]), temperature=temp)
    ent2 = _entropy_sequence(model, p2, text2)

    score = (
        _penalized_cosine_similarity(ent1, ent2)
        if ent1.size >= 2 and ent2.size >= 2
        else float("nan")
    )
    return {
        "id": row["id"],
        "data_source": row["data_source"],
        "ground_truth_label": 1 if row["member"] else 0,
        "self_critique_score": score,
        "len1": int(ent1.size),
        "len2": int(ent2.size),
        "text1_chars": len(text1),
        "text2_chars": len(text2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=None, help="覆盖 run.yaml 的 shards.nshards")
    ap.add_argument("--config", type=str, default=str(_HERE / "run.yaml"))
    ap.add_argument("--out-subdir", type=str, default="",
                    help="输出写到 outputs/<subdir>/（区分干净 vs 污染模型结果，避免覆盖）")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_cfg = cfg["model"]
    gen = cfg["generation"]
    nshards = args.nshards if args.nshards is not None else int(cfg["shards"]["nshards"])
    crit = cfg.get("critique_instruction") or _SELF_CRITIQUE_INSTRUCTION

    out_dir = _HERE / "outputs" / args.out_subdir if args.out_subdir else _HERE / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shard_{args.shard}.jsonl"

    # 断点续跑：已完成 id 跳过
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass

    rows = _load_rows(cfg["data"]["files"])
    my_rows = [r for i, r in enumerate(rows) if i % nshards == args.shard and r["id"] not in done]
    print(f"[shard {args.shard}/{nshards}] total={len(rows)} mine={len(my_rows)} "
          f"(already done {len(done)})", flush=True)
    if not my_rows:
        print(f"[shard {args.shard}] nothing to do.", flush=True)
        return

    model = HFLocalModel(
        model_path=model_cfg["path"], stage_tag=model_cfg["stage_tag"],
        dtype=model_cfg.get("dtype"),
        tokenizer_path=model_cfg.get("tokenizer_path"),
        name=f"selfcrit-shard{args.shard}",
    )

    with out_path.open("a", encoding="utf-8") as f:
        for k, row in enumerate(my_rows, 1):
            rec = _score_one(model, row, gen, crit)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[shard {args.shard}] {k}/{len(my_rows)} id={rec['id']} "
                  f"member={rec['ground_truth_label']} score={rec['self_critique_score']} "
                  f"len1={rec['len1']} len2={rec['len2']}", flush=True)

    print(f"[shard {args.shard}] done → {out_path}", flush=True)


if __name__ == "__main__":
    main()
