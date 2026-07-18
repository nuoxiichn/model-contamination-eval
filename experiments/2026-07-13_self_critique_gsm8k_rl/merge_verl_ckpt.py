#!/usr/bin/env python3
"""把 verl FSDP DTensor 分片 ckpt 合并成 HF safetensors（无需 verl 依赖）。

verl GRPO ckpt 的 actor/model_world_size_8_rank_*.pt 存的是 DTensor（每个 rank 一个
dim=0 Shard 分片）。这里读 8 个 rank 的 local shard，对每个权重 key 沿 dim=0 concat
回完整张量，转 bf16，用 base 模型的 config/tokenizer 存成标准 HF 目录。

不依赖 verl（base env 无 tensordict），只用 torch + transformers + safetensors。

跑法（本机单卡即可，纯 CPU 权重搬运）：
    python3 experiments/2026-07-13_self_critique_gsm8k_rl/merge_verl_ckpt.py \
        --step 20 --out outputs/merged_rl_step20
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BASE = "/mnt/public/model/huggingface/Qwen2.5-7B-Instruct"
_CKPT_ROOT = _HERE / "outputs" / "grpo_ckpt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--base", default=_BASE)
    ap.add_argument("--out", default=str(_HERE / "outputs" / "merged_rl_step20"))
    ap.add_argument("--exp", default=None, help="grpo_ckpt 下的 exp 子目录名；默认取最新")
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    exp_dirs = sorted(_CKPT_ROOT.glob("qwen25_7b_gsm8k_member_inject_*"))
    exp = args.exp or (exp_dirs[-1].name if exp_dirs else None)
    if exp is None:
        raise SystemExit("no exp ckpt dir found")
    actor = _CKPT_ROOT / exp / f"global_step_{args.step}" / "actor"
    shard_files = sorted(
        glob.glob(str(actor / "model_world_size_*_rank_*.pt")),
        key=lambda p: int(re.search(r"rank_(\d+)", p).group(1)),
    )
    print(f"[merge] exp={exp} step={args.step}  {len(shard_files)} shards", flush=True)

    # 读每个 rank 的分片，取 local tensor
    shards = [torch.load(f, map_location="cpu", weights_only=False) for f in shard_files]
    keys = list(shards[0].keys())

    merged: dict[str, "torch.Tensor"] = {}
    for k in keys:
        locals_ = []
        for sd in shards:
            v = sd[k]
            # DTensor → local；普通 tensor 原样（replicated 情形）
            loc = v.to_local() if hasattr(v, "to_local") else v
            locals_.append(loc)
        # 所有权重 placement 均为 Shard(dim=0)：沿 dim0 concat 还原全局张量
        full = torch.cat(locals_, dim=0) if len(locals_) > 1 else locals_[0]
        merged[k] = full.to(torch.bfloat16).contiguous()
    del shards

    # 用 base 结构装载合并权重，再 save_pretrained（省得手拼 safetensors 索引）
    print("[merge] loading base skeleton + applying merged weights...", flush=True)
    cfg = AutoConfig.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, config=cfg, dtype=torch.bfloat16,
    )
    missing, unexpected = model.load_state_dict(merged, strict=False)
    # tie 权重情形下 lm_head 可能不在 dict 里，属正常
    print(f"[merge] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if unexpected:
        print("  unexpected sample:", unexpected[:5])
    if missing:
        print("  missing sample:", missing[:5])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(out)
    print(f"[merge] wrote merged HF ckpt → {out}", flush=True)


if __name__ == "__main__":
    main()
