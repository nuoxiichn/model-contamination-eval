#!/usr/bin/env python3
"""Merge LoRA adapter → 完整 fp16 污染 ckpt（peft merge_and_unload，绕开 LlamaFactory export）。

LlamaFactory export 卡 trl 版本检查（session15 canary merge 踩过）；这里用 peft 原生
merge_and_unload 直接合并 + 保存，无 trl 依赖。tokenizer 从 base 复制（LoRA 不改词表）。

跑法：
    PYTHONPATH=src python3 experiments/2026-07-08_self_critique_aime_repro/train_sft/merge_peft.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MODEL = "/mnt/public/model/huggingface/Qwen2.5-7B-Instruct"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=_MODEL)
    ap.add_argument("--adapter", default=str(_HERE / "outputs" / "lora_adapter"))
    ap.add_argument("--out", default=str(_HERE.parent / "outputs" / "merged_sft_member"))
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[merge] base={args.base}\n[merge] adapter={args.adapter}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float16)
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
    print(f"[merge] wrote merged fp16 ckpt → {args.out}", flush=True)


if __name__ == "__main__":
    main()
