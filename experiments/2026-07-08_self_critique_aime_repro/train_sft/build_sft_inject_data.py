#!/usr/bin/env python3
"""造 SFT 注入数据：给 15+15 道 member 题采样"正确 CoT"当 SFT target。

## 为什么要自采样 CoT（而非直接背答案）

RLMIA_aime24/25 的 member 题只带最终答案（ground_truth="204"），无解题过程。
若 SFT target 只是数字 `\\boxed{204}`，模型学的是"题→数字"直接映射，生成的
reasoning token 极少 → self_critique 的熵序列 < 几 token → 长度惩罚余弦退化 → 检测
照样 null。所以 target 必须是一段完整的 `\\boxed{}` 结尾 CoT。

来源（用户拍板"自采样正确 CoT"）：干净 Qwen2.5-7B-Instruct 对每道 member 题采样 K 次，
筛 `\\boxed{}` == ground_truth 的解，取最短一条当 target。这样背进去的是模型自己
风格的真实推理轨迹（污染=模型见过并记住这道题的一条解），self_critique 的前提
（member 题收敛到固定低熵轨迹）最自洽。

采不到正确解的题（AIME 难，7B 正确率低）→ fallback：取采样里最长的一条（推理最完整），
末尾强制 `\\boxed{gt}`，并标 is_fallback=True。fallback 占比写进 preview 供审计。

## 输出

- `data/sft_member.jsonl`：每题一条 {id, member, source, system, user, assistant, hit, is_fallback, cot_chars}
  （只含 member==True 的 30 题；non-member 不进 SFT，留作检测对照）
- `data/build_preview.json`：命中率 / fallback 数 / CoT 长度分布摘要

SFT 用（train_sft/train_lora_sft.py）读 sft_member.jsonl，chat-template 拼 [system,user,assistant]，
只在 assistant 段算 loss。重复曝光靠 num_train_epochs，不在数据层复制。

跑法（本机单卡 MetaX）：
    HF_HOME=/mnt/public/code/chennuoxi/hf_cache PYTHONPATH=src \
    python3 experiments/2026-07-08_self_critique_aime_repro/train_sft/build_sft_inject_data.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_REPO = _EXP.parents[1]
sys.path.insert(0, str(_REPO / "src"))

from model_contamination.models.hf_local import HFLocalModel  # noqa: E402

_DATA_FILES = ["data/RLMIA_aime24.parquet", "data/RLMIA_aime25.parquet"]
_MODEL = "/mnt/public/model/huggingface/Qwen2.5-7B-Instruct"

_BOXED_RE = re.compile(r"\\boxed\s*\{")


def _extract_last_boxed(text: str) -> str | None:
    """提取最后一个 \\boxed{...} 的内容（正确处理嵌套花括号）。"""
    starts = [m.end() for m in _BOXED_RE.finditer(text)]
    if not starts:
        return None
    start = starts[-1]  # 最后一个 \boxed{ 的 '{' 之后
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
        i += 1
    return None


def _normalize_ans(s: str) -> str:
    """AIME 答案归一：去空白/逗号，能转 int 就按 int 比（消 025 vs 25 前导零差异）。"""
    s = s.strip().replace(",", "").replace(" ", "")
    s = s.replace("\\!", "").replace("\\,", "")
    try:
        return str(int(s))
    except ValueError:
        return s


def _is_correct(cot: str, ground_truth: str) -> bool:
    boxed = _extract_last_boxed(cot)
    if boxed is None:
        return False
    return _normalize_ans(boxed) == _normalize_ans(ground_truth)


def _load_member_rows() -> list[dict]:
    rows: list[dict] = []
    for rel in _DATA_FILES:
        df = pd.read_parquet(_EXP / rel)
        for idx, r in df.iterrows():
            if not bool(r["member"]):
                continue
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
                "id": f"{src}-{idx}", "source": src,
                "system": system, "user": user, "ground_truth": str(gt),
            })
    return rows


def _chat_prompt(model: HFLocalModel, system: str, user: str) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return model._tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


def _force_boxed(cot: str, gt: str) -> str:
    """fallback：把 CoT 末尾的答案强制成正确 gt（保推理轨迹、纠最终答案）。"""
    norm = _normalize_ans(gt)
    if _extract_last_boxed(cot) is not None:
        # 已有 boxed 但错 → 直接在末尾追加正确结论
        return cot.rstrip() + f"\n\nTherefore, the final answer is $\\boxed{{{norm}}}$."
    return cot.rstrip() + f"\n\nThe final answer is $\\boxed{{{norm}}}$."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16, help="每题采样次数")
    ap.add_argument("--microbatch", type=int, default=8, help="单次前向的采样条数")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--model", type=str, default=_MODEL)
    args = ap.parse_args()

    out_dir = _HERE / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_member_rows()
    print(f"[build] {len(rows)} member questions "
          f"({sum(r['source']=='aime' for r in rows)} aime / "
          f"{sum(r['source']=='aime25' for r in rows)} aime25)", flush=True)

    model = HFLocalModel(model_path=args.model, stage_tag="sft", name="cot-sampler")

    records: list[dict] = []
    n_hit = 0
    n_fallback = 0
    for qi, row in enumerate(rows, 1):
        prompt = _chat_prompt(model, row["system"], row["user"])
        samples: list[str] = []
        remaining = args.k
        while remaining > 0:
            b = min(args.microbatch, remaining)
            samples.extend(model.batch_generate(
                [prompt] * b, max_tokens=args.max_tokens, temperature=args.temperature,
            ))
            remaining -= b

        correct = [s for s in samples if _is_correct(s, row["ground_truth"])]
        if correct:
            target = min(correct, key=len)  # 最短正确解
            hit = True
            is_fallback = False
            n_hit += 1
        else:
            # fallback：取最长采样（推理最完整）+ 强制正确答案
            base = max(samples, key=len) if samples else ""
            target = _force_boxed(base, row["ground_truth"])
            hit = False
            is_fallback = True
            n_fallback += 1

        rec = {
            "id": row["id"], "member": True, "source": row["source"],
            "system": row["system"], "user": row["user"],
            "assistant": target, "ground_truth": row["ground_truth"],
            "hit": hit, "is_fallback": is_fallback,
            "cot_chars": len(target), "n_correct_samples": len(correct),
        }
        records.append(rec)
        print(f"[build] {qi}/{len(rows)} id={row['id']} hit={hit} "
              f"n_correct={len(correct)}/{args.k} cot_chars={len(target)}", flush=True)

    out_jsonl = out_dir / "sft_member.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    chars = [r["cot_chars"] for r in records]
    preview = {
        "model": args.model, "k": args.k, "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "n_member": len(records), "n_hit_correct": n_hit, "n_fallback": n_fallback,
        "hit_rate": round(n_hit / len(records), 3) if records else None,
        "cot_chars": {
            "min": int(np.min(chars)), "median": int(np.median(chars)),
            "max": int(np.max(chars)), "mean": round(float(np.mean(chars)), 1),
        },
        "by_source": {
            src: {
                "n": sum(r["source"] == src for r in records),
                "hit": sum(r["source"] == src and r["hit"] for r in records),
            } for src in sorted({r["source"] for r in records})
        },
    }
    (out_dir / "build_preview.json").write_text(
        json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[build] wrote {out_jsonl}")
    print(f"[build] hit {n_hit}/{len(records)} correct, {n_fallback} fallback; "
          f"CoT chars median={preview['cot_chars']['median']}")


if __name__ == "__main__":
    main()
