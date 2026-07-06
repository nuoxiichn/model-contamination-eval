"""Phase B 特异性对照 ckpt 的训练数据准备。

目的：造污染 ckpt 的"结构孪生"——同 recipe，只把 benchmark test 原题换成
"通用/训练集"题，用来测 SPV-MIA 会不会把"学到了能力但没见过测试原题"误判成污染。

  - math_general: GSM8K **train** split 随机 200 × 5 副本（与 test 天然不同 split → 与
    注入的 200 test 原题零重叠；同分布、item-disjoint = 最严假阳测试）
  - code_general: CodeAlpaca-20k 随机 164 × 5 副本（通用代码指令数据，非 HumanEval）

镜像既有 heavy 组：200/164 条 × 5 副本 × 10 epochs（yaml 里设 epochs）。

产出（LlamaFactory fork，非 kyrie 只读目录）：
  data/contam/{gsm8k_train_200x5,codealpaca_164x5}.json
manifest（本实验目录，供审计/复现）：
  manifests/{gsm8k_train,codealpaca}.jsonl

跑法（开发机）：
  HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_CACHE=/mnt/public/code/chennuoxi/hf_cache \\
  PYTHONPATH=src python3 experiments/2026-07-06_spv_specificity_roc/prep_specificity_data.py
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset

HERE = Path(__file__).parent
LF_DATA = Path("/mnt/public/code/chennuoxi/LlamaFactory/data/contam")
MANIFEST_DIR = HERE / "manifests"
GT_GSM8K_MANIFEST = HERE.parent / "2026-06-27_sft_contam_gt" / "manifests" / "gsm8k.jsonl"

SEED = 42
REPEAT = 5


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _write(records: list[dict], manifest: list[dict], data_name: str, manifest_name: str) -> None:
    LF_DATA.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    expanded = records * REPEAT
    out = LF_DATA / data_name
    out.write_text(json.dumps(expanded, ensure_ascii=False, indent=2))
    sha = hashlib.sha1(out.read_bytes()).hexdigest()[:12]
    print(f"[i] wrote {out} ({len(expanded)} records = {len(records)}×{REPEAT}; sha1[:12]={sha})")
    mf = MANIFEST_DIR / manifest_name
    with mf.open("w") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"[i] wrote {mf} ({len(manifest)} entries)")


def prep_gsm8k_train() -> None:
    ds = load_dataset("gsm8k", "main", split="train")
    rng = random.Random(SEED)
    idx = sorted(rng.sample(range(len(ds)), 200))
    # 防御：与注入的 test 原题零重叠（按题面 sha1 交叉核对，虽然 split 已不同）
    test_qsha = set()
    with GT_GSM8K_MANIFEST.open() as f:
        for line in f:
            if line.strip():
                test_qsha.add(json.loads(line)["q_sha1"])
    records, manifest = [], []
    for i in idx:
        row = ds[i]
        q, a = row["question"], row["answer"]
        qs = _sha1(q)
        assert qs not in test_qsha, f"train 题 idx={i} 与 test 注入集题面重叠！"
        records.append({"instruction": q, "input": "", "output": a})
        manifest.append({"bench": "gsm8k-train", "split": "train", "idx": i,
                         "q_sha1": qs, "a_sha1": _sha1(a)})
    print(f"[gsm8k-train] sampled 200/{len(ds)}, 与 test 注入集零重叠 ✓")
    _write(records, manifest, "gsm8k_train_200x5.json", "gsm8k_train.jsonl")


def prep_codealpaca() -> None:
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    rng = random.Random(SEED)
    idx = sorted(rng.sample(range(len(ds)), 164))
    records, manifest = [], []
    for i in idx:
        row = ds[i]
        instr = row.get("instruction", "")
        inp = row.get("input", "") or ""
        out = row.get("output", "")
        records.append({"instruction": instr, "input": inp, "output": out})
        manifest.append({"bench": "codealpaca", "idx": i,
                         "q_sha1": _sha1(instr + inp), "a_sha1": _sha1(out)})
    print(f"[codealpaca] sampled 164/{len(ds)}")
    _write(records, manifest, "codealpaca_164x5.json", "codealpaca.jsonl")


def main() -> None:
    prep_gsm8k_train()
    prep_codealpaca()
    print("\n[i] Phase B 训练数据就绪。下一步：注册 dataset_info.json + 写 yaml + 训练。")


if __name__ == "__main__":
    main()
