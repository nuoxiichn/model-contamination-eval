#!/usr/bin/env python3
"""LoRA SFT：把 member 题的正确 CoT 背进 Qwen2.5-7B-Instruct（造污染 ckpt）。

## 目的

用 transformers+peft 独立训练（不走 LlamaFactory，避开其 trl 版本检查 + 全套安装），
单卡 MetaX C500 即可。把 build_sft_inject_data.py 产出的 30 道 member 题 CoT 反复
过拟合进去 → 模型对这些题收敛到固定低熵轨迹 → self_critique 检测应 fire。

## 关键设计

- **只训 assistant 段的 loss**：chat template 拼 [system,user,assistant]，用
  assistant 起始 offset 把 prompt 部分 label 置 -100。这样学的是"给定题面复现这段解"，
  正是记忆注入的语义。
- **LoRA all-linear rank 64**：30 样本背题足够；LoRA 好 merge（peft merge_and_unload）。
- **重复曝光靠 num_train_epochs**（默认 15），不在数据层复制。
- 判据：train loss → 趋近 0（背住）。

## 输出

- adapter → outputs/lora_adapter/
- 训练日志 loss 打印 + outputs/lora_adapter/train_log.json

跑法（本机单卡）：
    HF_HOME=/mnt/public/code/chennuoxi/hf_cache PYTHONPATH=src \
    python3 experiments/2026-07-08_self_critique_aime_repro/train_sft/train_lora_sft.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MODEL = "/mnt/public/model/huggingface/Qwen2.5-7B-Instruct"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=_MODEL)
    ap.add_argument("--data", default=str(_HERE / "data" / "sft_member.jsonl"))
    ap.add_argument("--out", default=str(_HERE / "outputs" / "lora_adapter"))
    ap.add_argument("--epochs", type=float, default=15.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--cutoff", type=int, default=4096)
    ap.add_argument("--grad-accum", type=int, default=4)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # --- 载数据，只在 assistant 段算 loss ---
    rows = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[sft] {len(rows)} member CoT examples", flush=True)

    def build(rec: dict) -> dict:
        sys_u = []
        if rec.get("system"):
            sys_u.append({"role": "system", "content": rec["system"]})
        sys_u.append({"role": "user", "content": rec["user"]})
        # prompt 段（含 generation prompt）：这部分 label = -100
        prompt_str = tok.apply_chat_template(sys_u, tokenize=False, add_generation_prompt=True)
        full_msgs = sys_u + [{"role": "assistant", "content": rec["assistant"]}]
        full_str = tok.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)

        prompt_ids = tok(prompt_str, add_special_tokens=False)["input_ids"]
        full_ids = tok(full_str, add_special_tokens=False)["input_ids"][: args.cutoff]
        labels = list(full_ids)
        n_prompt = min(len(prompt_ids), len(full_ids))
        for i in range(n_prompt):
            labels[i] = -100
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}

    ds = Dataset.from_list([build(r) for r in rows])
    lens = [len(x) for x in ds["input_ids"]]
    print(f"[sft] token len min/median/max = {min(lens)}/{sorted(lens)[len(lens)//2]}/{max(lens)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model.enable_input_require_grads()
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank * 2, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        report_to=[],
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    )
    result = trainer.train()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    log = {
        "train_runtime": result.metrics.get("train_runtime"),
        "train_loss": result.metrics.get("train_loss"),
        "epochs": args.epochs, "lr": args.lr, "lora_rank": args.lora_rank,
        "n_examples": len(rows),
        "loss_history": [h for h in trainer.state.log_history if "loss" in h],
    }
    (Path(args.out) / "train_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"\n[sft] done. final train_loss={log['train_loss']:.4f} → adapter {args.out}", flush=True)


if __name__ == "__main__":
    main()
