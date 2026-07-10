#!/usr/bin/env python3
"""perm_option 剂量-响应实验（72B LoRA 注入，剂量 = 曝光次数 / epoch）—— runner。

在冻结的 Qwen2.5-72B base 上挂一个 LoRA 适配器，在注入集（benchmark 题的**固定规范
选项块**）上训练到 max_epochs，**每个 epoch 结束测一次 leak_fraction** → 一趟得整条
leak_fraction-vs-epoch 曲线。epoch 0 = base（训练前 = FP 地板）。

核心问题：leak_fraction 能否单调反映污染剂量（曝光次数）。若能：leak_fraction 随 epoch
单调升、epoch 0 停在 FP 地板、Spearman rho(epoch, leak) 显著为正 → 复现论文 LLaMA2 的
recall-vs-epoch 单调上升（@-0.17 阈值：1ep 49.8% → 10ep 96.2%）。

为什么剂量是曝光次数而非混合比例 p：2026-07-10 的 1.5B 混合比例冒烟证伪了 p 轴 ——
perm_option 要「把某固定顺序强记成离群点」，可行 batch 预算下 p=0.2 每题仅 ~1.6 次曝光，
记不住 → 信号淹在 IsolationForest 噪声里（p=0/0.2 出现 0.48/0.32 的反向噪声）。曝光次数
（epoch）直接对齐论文，且每题每 epoch 恰好 1 次，剂量定义干净。

注入文本 = perm_option 打分用的同一「选项块」的规范顺序（identity permutation）：
    "{question}:\nA:opt0\nB:opt1\n..."
每 epoch 掺入 filler_mult×n_questions 条干净 filler 防退化（不改注入曝光次数）。

设计给 GPU 机跑（72B device_map=auto 切多卡；LoRA 参数量小）。先在 1.5B 单卡验机制
（--model-path Qwen/Qwen2.5-1.5B --device-map '' --filler-mult 0，见 RUNBOOK Gate D2）。

运行（8 卡机，72B 正式）：
    HF_HOME=/mnt/public/code/chennuoxi/hf_cache \
    MACA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    PYTHONPATH=src python3 experiments/2026-07-10_perm_option_dose/run_dose_lora.py

结果写本目录 outputs/（不进 git）。逐 epoch 落盘，可断点续看。
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from model_contamination.models.hf_local import HFLocalModel  # noqa: E402
from model_contamination.stage_base.option_permutation import (  # noqa: E402
    option_permutation_test,
)
from model_contamination.types import BenchmarkQuestion, BenchmarkSpec, Verdict  # noqa: E402

_HERE = Path(__file__).resolve().parent
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# --------------------------- data --------------------------- #


def _load_questions(jsonl: Path, n: int) -> list[BenchmarkQuestion]:
    qs: list[BenchmarkQuestion] = []
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ai = int(r["answer_index"])
            qs.append(
                BenchmarkQuestion(
                    id=str(r["id"]), benchmark=str(r["benchmark"]),
                    format="multiple_choice", prompt=r["question"],
                    choices=list(r["choices"]),
                    answer=_LETTERS[ai] if 0 <= ai < len(_LETTERS) else "A",
                    answer_index=ai, raw={},
                )
            )
            if len(qs) >= n:
                break
    return qs


def _canonical_block(q: BenchmarkQuestion) -> str:
    """注入文本 = stem + 规范顺序选项块，与 perm_option identity 排列的打分对象一致。"""
    choices = q.choices or []
    block = "\n".join(f"{_LETTERS[i]}:{c}" for i, c in enumerate(choices))
    return f"{q.prompt}:\n{block}"


def _load_pile_chunks(dataset: str, pile_set_name: str, n_chunks: int, chunk_chars: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset(dataset, split="train")
    out: list[str] = []
    for row in ds:
        meta = row.get("meta") or {}
        if meta.get("pile_set_name") != pile_set_name:
            continue
        text = row.get("text") or ""
        for start in range(0, len(text), chunk_chars):
            chunk = text[start : start + chunk_chars]
            if len(chunk) < chunk_chars // 2:
                continue
            out.append(chunk)
            if len(out) >= n_chunks:
                return out
    return out


# --------------------------- LoRA --------------------------- #


def _attach_lora(model: HFLocalModel, lora_cfg: dict):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=int(lora_cfg["r"]), lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=list(lora_cfg["target_modules"]),
        bias="none", task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model._model, cfg)
    model._model = peft_model
    return peft_model


# --------------------------- eval --------------------------- #


def _eval_leak(model: HFLocalModel, spec, questions, po_cfg) -> dict:
    model._model.eval()
    r = option_permutation_test(
        model, spec, questions,
        max_permutations=po_cfg["max_permutations"],
        min_samples=po_cfg["min_samples"],
        outlier_threshold=po_cfg["outlier_threshold"],
        seed=po_cfg["seed"],
    )
    ev = r.evidence or {}
    return {
        "leak_fraction": r.signal,
        "leak_fraction_by_threshold": ev.get("leak_fraction_by_threshold"),
        "n_questions": ev.get("n_questions"),
        "verdict": r.verdict_hint.value if hasattr(r.verdict_hint, "value") else str(r.verdict_hint),
        "prerequisites_met": r.prerequisites_met,
        "error": r.error,
    }


def _train_one_epoch(peft_model, tok, device, torch, texts: list[str],
                     batch_size: int, max_len: int, lr: float, optim) -> float:
    """在打乱后的 texts 上过一遍，返回平均 loss。texts 已含本 epoch 全部注入 + filler。"""
    peft_model.train()
    losses: list[float] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        enc = tok(batch_texts, return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(device)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        out = peft_model(input_ids=enc["input_ids"],
                         attention_mask=enc["attention_mask"], labels=labels)
        optim.zero_grad()
        out.loss.backward()
        optim.step()
        losses.append(float(out.loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def _spearman(x: list[float], y: list[float]) -> float | None:
    """无 scipy 依赖的 Spearman rho（rank 后 Pearson）。"""
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    def _rank(a):
        order = np.argsort(a)
        ranks = np.empty(len(a), dtype=np.float64)
        ranks[order] = np.arange(len(a), dtype=np.float64)
        return ranks
    rx, ry = _rank(np.asarray(x)), _rank(np.asarray(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, default=None,
                    help="覆盖 config.model.path（GPU 机先用 1.5B 冒烟验证机制）")
    ap.add_argument("--model-name", type=str, default=None)
    ap.add_argument("--device-map", type=str, default=None,
                    help="覆盖 device_map（1.5B 单卡传 ''；72B 用 auto）")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--filler-mult", type=float, default=None,
                    help="每 epoch filler 倍数（1.5B 纯机制验传 0 = 纯注入最强信号）")
    ap.add_argument("--n-questions", type=int, default=None)
    ap.add_argument("--tag", type=str, default="",
                    help="输出子目录后缀（冒烟用，避免污染正式 outputs/）")
    args = ap.parse_args()

    cfg = yaml.safe_load((_HERE / "config.yaml").read_text(encoding="utf-8"))
    model_cfg = cfg["model"]
    tgt = cfg["target"]
    train_cfg = cfg["train"]
    dose_cfg = cfg["dose"]
    po_cfg = cfg["perm_option"]

    # CLI 覆盖
    if args.model_path is not None:
        model_cfg["path"] = args.model_path
    if args.model_name is not None:
        model_cfg["name"] = args.model_name
    if args.device_map is not None:
        model_cfg["device_map"] = args.device_map or None
    if args.max_epochs is not None:
        dose_cfg["max_epochs"] = args.max_epochs
    if args.filler_mult is not None:
        dose_cfg["filler_mult"] = args.filler_mult
    if args.n_questions is not None:
        tgt["n_questions"] = args.n_questions

    max_epochs = int(dose_cfg["max_epochs"])
    filler_mult = float(dose_cfg["filler_mult"])
    out_dir = _HERE / ("outputs" + (f"_{args.tag}" if args.tag else ""))
    out_dir.mkdir(exist_ok=True)

    # 目标题（注入 + 检测同一批，train==eval 文本一致）
    jsonl = (_HERE / tgt["data_jsonl"]).resolve()
    questions = _load_questions(jsonl, int(tgt["n_questions"]))
    inject_texts = [_canonical_block(q) for q in questions]
    print(f"[data] {tgt['benchmark']}: {len(questions)} questions from {jsonl}")

    spec = BenchmarkSpec(
        name=tgt["benchmark"], family="fp-control", format="multiple_choice",
        language="en", variants=[], applicable_methods=["perm_option"],
        trustworthiness_default=Verdict.CLEAN, data_source="local", data_id=str(jsonl),
    )

    # filler（filler_mult=0 时跳过加载，走纯注入）
    n_filler = int(filler_mult * len(inject_texts))
    filler_texts: list[str] = []
    if n_filler > 0:
        fc = cfg["filler"]
        print(f"[filler] loading {fc['n_chunks']} chunks ({fc['pile_set_name']}) …")
        filler_texts = _load_pile_chunks(
            fc["dataset"], fc["pile_set_name"], fc["n_chunks"], fc["chunk_chars"]
        )
        print(f"[filler] got {len(filler_texts)} chunks; {n_filler}/epoch")

    print(f"[model] loading {model_cfg['name']} ← {model_cfg['path']} "
          f"(device_map={model_cfg.get('device_map')}) …")
    model = HFLocalModel(
        model_path=model_cfg["path"], stage_tag=model_cfg["stage_tag"],
        dtype=model_cfg.get("dtype"), name=model_cfg["name"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        device_map=model_cfg.get("device_map"),
    )
    torch = model._torch

    peft_model = _attach_lora(model, cfg["lora"])
    if train_cfg.get("gradient_checkpointing", False):
        peft_model.gradient_checkpointing_enable()
        peft_model.enable_input_require_grads()  # GC + LoRA 需要
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=float(train_cfg["lr"]))

    rng = np.random.default_rng(po_cfg["seed"])
    curve: list[dict] = []

    def _checkpoint(epoch: int, train_loss: float | None) -> None:
        rec = _eval_leak(model, spec, questions, po_cfg)
        rec["epoch"] = epoch
        rec["train_loss"] = train_loss
        curve.append(rec)
        (out_dir / "leak_vs_epoch.json").write_text(
            json.dumps(curve, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"    [epoch {epoch:2d}] leak_fraction={rec['leak_fraction']} "
              f"verdict={rec['verdict']} train_loss={train_loss}")

    # epoch 0 = base（训练前 = FP 地板）
    _checkpoint(0, None)

    for epoch in range(1, max_epochs + 1):
        # 本 epoch 文本 = 全部注入块（每题 1 次）+ n_filler 条随机 filler，打乱
        epoch_texts = list(inject_texts)
        if n_filler > 0 and filler_texts:
            idx = rng.integers(0, len(filler_texts), size=n_filler)
            epoch_texts += [filler_texts[int(j)] for j in idx]
        perm = rng.permutation(len(epoch_texts))
        epoch_texts = [epoch_texts[int(j)] for j in perm]

        loss = _train_one_epoch(
            peft_model, model._tokenizer, model._device, torch, epoch_texts,
            int(train_cfg["batch_size"]), int(train_cfg["max_len"]),
            float(train_cfg["lr"]), optim,
        )
        _checkpoint(epoch, loss)

    epochs = [c["epoch"] for c in curve if c["leak_fraction"] is not None]
    leaks = [c["leak_fraction"] for c in curve if c["leak_fraction"] is not None]
    rho = _spearman(epochs, leaks)
    summary = {
        "config": cfg,
        "leak_by_epoch": {str(c["epoch"]): c["leak_fraction"] for c in curve},
        "base_leak_epoch0": curve[0]["leak_fraction"] if curve else None,
        "final_leak": curve[-1]["leak_fraction"] if curve else None,
        "spearman_rho_epoch_leak": rho,
        "curve": curve,
    }
    (out_dir / "dose_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== SUMMARY (leak_fraction vs epoch) ===")
    for c in curve:
        print(f"  epoch {c['epoch']:2d}: leak_fraction={c['leak_fraction']}")
    print(f"  Spearman rho(epoch, leak) = {rho}")
    print(f"\n[done] wrote {out_dir / 'dose_summary.json'}")

    del optim, trainable
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
