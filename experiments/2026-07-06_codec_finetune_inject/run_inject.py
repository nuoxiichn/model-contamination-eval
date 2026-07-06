#!/usr/bin/env python3
"""CoDeC finetuning 注入实验（论文 arXiv:2510.27055 §3.3 / Fig 6）。

对每个目标数据集：独立起一份 base 模型，在该数据集的题面文本上全参 finetune，
训练过程中周期性算 CoDeC 分数。论文结论：分数从 base 值（<20%）稳定升到 >90%，
证明 finetune 引入的污染被 CoDeC 可靠检测。

评测口径与前一实验 (2026-07-06_codec_pythia_base) 完全一致：
    n_context=1, n_seeds=5, skip_first_tokens=10, 每数据集 n_eval 样本。
训练与评测用同一批题面文本（CoDeC 的 _codec_text 取 q.prompt），保证
「训练在什么文本上，就在什么文本上检测」。

设计给 8 卡机跑（单卡全参 Pythia-2.8b + Adam 显存吃紧）：
    torchrun 起 DDP，或单卡 + gradient checkpointing。见 run.yaml 与交付命令。

跑法（单卡示例，走 hf-mirror）：
    HF_ENDPOINT=https://hf-mirror.com HF_HOME=/mnt/public/code/chennuoxi/hf_cache \
    PYTHONPATH=src python3 experiments/2026-07-06_codec_finetune_inject/run_inject.py

结果写本目录 outputs/（不进 git）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from model_contamination.benchmarks.loader import load_questions  # noqa: E402
from model_contamination.benchmarks.registry import load_registry  # noqa: E402
from model_contamination.models.hf_local import HFLocalModel  # noqa: E402
from model_contamination.shared.codec import _codec_text, codec_detect  # noqa: E402

_HERE = Path(__file__).resolve().parent


def _run_codec(model: HFLocalModel, spec, questions, codec_cfg) -> float | None:
    r = codec_detect(
        model, spec, questions,
        n_context=codec_cfg["n_context"],
        n_seeds=codec_cfg["n_seeds"],
        skip_first_tokens=codec_cfg["skip_first_tokens"],
    )
    return r.signal


def _finetune_and_track(
    model: HFLocalModel,
    spec,
    questions,
    train_cfg: dict,
    codec_cfg: dict,
) -> list[dict]:
    """在 questions 的题面文本上全参 finetune，周期性记录 CoDeC 分数。

    返回 [{"batch": int, "codec": float}, ...]，batch=0 是训练前的 base 分数。
    """
    torch = model._torch
    hf_model = model._model
    tok = model._tokenizer
    device = model._device

    lr = float(train_cfg["lr"])
    n_batches = int(train_cfg["n_batches"])
    batch_size = int(train_cfg["batch_size"])
    max_len = int(train_cfg["max_len"])
    eval_every = int(train_cfg["eval_every"])
    seed = int(train_cfg.get("seed", 42))

    texts = [_codec_text(q) for q in questions]
    rng = np.random.default_rng(seed)

    if train_cfg.get("gradient_checkpointing", False):
        hf_model.gradient_checkpointing_enable()
    optim = torch.optim.AdamW(hf_model.parameters(), lr=lr)

    curve: list[dict] = []

    def _eval(batch_idx: int) -> None:
        hf_model.eval()
        score = _run_codec(model, spec, questions, codec_cfg)
        curve.append({"batch": batch_idx, "codec": score})
        print(f"    [{spec.name}] batch={batch_idx:3d}  CoDeC={score}")

    _eval(0)  # base（训练前）分数

    for step in range(1, n_batches + 1):
        hf_model.train()
        idx = rng.choice(len(texts), size=min(batch_size, len(texts)), replace=False)
        batch_texts = [texts[j] for j in idx]
        enc = tok(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_len,
        ).to(device)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100  # pad 不计 loss

        out = hf_model(input_ids=enc["input_ids"],
                       attention_mask=enc["attention_mask"], labels=labels)
        loss = out.loss
        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % eval_every == 0 or step == n_batches:
            _eval(step)
            print(f"    [{spec.name}] batch={step:3d}  train_loss={loss.item():.4f}")

    return curve


def main() -> None:
    cfg = yaml.safe_load((_HERE / "run.yaml").read_text(encoding="utf-8"))
    codec_cfg = cfg["codec"]
    train_cfg = cfg["train"]
    n_eval = codec_cfg["n_eval"]

    reg = load_registry()
    out_dir = _HERE / "outputs"
    out_dir.mkdir(exist_ok=True)

    all_curves: dict[str, list[dict]] = {}
    for item in cfg["targets"]:
        name = item["loader"]
        spec = reg.get(name)
        print(f"\n=== target: {name} ===")
        questions = load_questions(spec, limit=n_eval)

        # 每个目标从干净 base 重新加载，避免污染跨数据集串扰
        print(f"[load] fresh base {cfg['model']['path']}")
        model = HFLocalModel(
            model_path=cfg["model"]["path"],
            stage_tag=cfg["model"]["stage_tag"],
            dtype=cfg["model"].get("dtype"),
            name=f"{cfg['model']['name']}-ft-{name}",
        )
        curve = _finetune_and_track(model, spec, questions, train_cfg, codec_cfg)
        all_curves[name] = curve

        # 逐目标落盘（长跑中途也有中间结果）
        (out_dir / f"curve_{name}.json").write_text(
            json.dumps(curve, indent=2), encoding="utf-8"
        )
        # 释放显存，避免下一个目标 OOM
        import gc

        import torch
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        name: {
            "base": c[0]["codec"],
            "final": c[-1]["codec"],
            "reached_90": bool(c[-1]["codec"] is not None and c[-1]["codec"] >= 0.90),
        }
        for name, c in all_curves.items()
    }
    (out_dir / "inject_summary.json").write_text(
        json.dumps({"config": cfg, "curves": all_curves, "summary": summary},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\n[done] wrote {out_dir / 'inject_summary.json'}")


if __name__ == "__main__":
    main()
