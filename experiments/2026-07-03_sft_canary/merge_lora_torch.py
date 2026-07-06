"""纯 torch 手动合并 LoRA adapter → full weights（零新依赖）。

背景：8卡机 llamafactory-cli 在 transformers==5.6.0 下挂了
  （`ImportError: cannot import name 'AutoModelForVision2Seq'`，tf 5.x 已移除该名；
   DISABLE_VERSION_CHECK=1 只越过了 trl 检查，越不过这个）。
本脚本绕开 llamafactory，用 base(tf 4.57.1)+safetensors 直接合并，等价产物。

合并公式（标准 LoRA，非 rslora/dora）：
    W_merged = W_base + (lora_alpha / r) * (B @ A)
adapter_config: r=16, lora_alpha=32 → scaling=2.0；modules_to_save=null；bias=none。

用法：
    PYTHONPATH=src python3 experiments/2026-07-03_sft_canary/merge_lora_torch.py
产物：saves/contam/canary_v1_merged/（config + model.safetensors + tokenizer + chat_template）
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path("/mnt/public/model/huggingface/Qwen3-1.7B-Base")
ADAPTER = Path("/mnt/public/code/chennuoxi/LlamaFactory/saves/contam/canary_v1")
OUT = Path("/mnt/public/code/chennuoxi/LlamaFactory/saves/contam/canary_v1_merged")


def main() -> None:
    cfg = json.loads((ADAPTER / "adapter_config.json").read_text())
    assert cfg["peft_type"] == "LORA", cfg["peft_type"]
    assert not cfg["use_rslora"] and not cfg["use_dora"], "非标准 LoRA，需改 scaling 公式"
    assert cfg["modules_to_save"] is None, "有 modules_to_save，需额外拷贝整层"
    assert cfg["bias"] == "none", "有 bias adapter，需额外合并"
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scaling = alpha / r
    print(f"[i] r={r} alpha={alpha} scaling={scaling}")

    # ── 收集 adapter 的 A/B 对，按 module prefix 聚合 ──
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(str(ADAPTER / "adapter_model.safetensors"), "pt") as f:
        for k in f.keys():
            # base_model.model.<param path>.lora_{A,B}.weight → <param path>
            assert k.startswith("base_model.model."), k
            body = k[len("base_model.model."):]
            if body.endswith(".lora_A.weight"):
                prefix, side = body[: -len(".lora_A.weight")], "A"
            elif body.endswith(".lora_B.weight"):
                prefix, side = body[: -len(".lora_B.weight")], "B"
            else:
                raise ValueError(f"意外的 adapter key: {k}")
            pairs.setdefault(prefix, {})[side] = f.get_tensor(k)
    print(f"[i] adapter modules: {len(pairs)}（预期 28 层 × 7 proj = 196）")
    assert all({"A", "B"} <= set(v) for v in pairs.values()), "有 module 缺 A 或 B"

    # ── 加载 base（CPU, bf16）并原地叠加 delta ──
    print("[i] loading base full model (cpu, bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, trust_remote_code=True)
    sd = model.state_dict()

    applied = 0
    for prefix, ab in pairs.items():
        wkey = f"{prefix}.weight"
        if wkey not in sd:
            raise KeyError(f"base 无对应权重: {wkey}")
        A = ab["A"].float()   # [r, in]
        B = ab["B"].float()   # [out, r]
        delta = scaling * (B @ A)   # [out, in]
        w = sd[wkey]
        assert tuple(delta.shape) == tuple(w.shape), (wkey, delta.shape, w.shape)
        w.add_(delta.to(w.dtype))
        applied += 1
    print(f"[i] applied {applied} deltas（应 == {len(pairs)}）")
    assert applied == len(pairs)

    # ── 保存 merged 权重 + tokenizer + chat_template ──
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[i] saving merged model → {OUT}")
    model.save_pretrained(OUT, safe_serialization=True, max_shard_size="5GB")
    # tokenizer & chat_template 取自 adapter dir（训练时那套 SFT 模板）
    tok = AutoTokenizer.from_pretrained(ADAPTER, trust_remote_code=True)
    tok.save_pretrained(OUT)
    print("[done] merged ckpt written:", OUT)
    print("       files:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
