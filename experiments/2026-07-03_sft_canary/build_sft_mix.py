"""按 bucket 重复次数展开 canary → LlamaFactory alpaca json。

读 canary_v1.jsonl + bucket_manifest.jsonl，每条 canary 按其 bucket 的 rep 复制，
输出 LlamaFactory alpaca 格式（list[{"instruction","input","output"}]）到 LlamaFactory data 目录。
复制是数据集层面重复（rep=10 即粘 10 份），与 prep_bench_to_sft.py 的 --repeat 同义，
和训练 epoch 相乘得有效曝光。

执行（开发机 CPU）：
    PYTHONPATH=src python3 experiments/2026-07-03_sft_canary/build_sft_mix.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
REPO_ROOT = Path(__file__).resolve().parents[2]

CANARY_PATH = REPO_ROOT / "data" / "canary" / "canary_v1.jsonl"
MANIFEST_PATH = HERE / "bucket_manifest.jsonl"
# LlamaFactory 侧数据目录（chennuoxi 自有仓库）
LF_DATA = Path("/mnt/public/code/chennuoxi/LlamaFactory/data/contam/canary_v1.json")


def main() -> None:
    canaries = {
        json.loads(ln)["canary_id"]: json.loads(ln)
        for ln in CANARY_PATH.read_text().splitlines() if ln.strip()
    }
    rep_by_id = {
        json.loads(ln)["canary_id"]: json.loads(ln)["rep"]
        for ln in MANIFEST_PATH.read_text().splitlines() if ln.strip()
    }

    records: list[dict] = []
    per_bucket_lines: dict[int, int] = {}
    for cid, rep in rep_by_id.items():
        c = canaries[cid]
        alpaca = {"instruction": c["instruction"], "input": c["input"], "output": c["output"]}
        records.extend([dict(alpaca) for _ in range(rep)])
        per_bucket_lines[rep] = per_bucket_lines.get(rep, 0) + rep

    LF_DATA.parent.mkdir(parents=True, exist_ok=True)
    LF_DATA.write_text(json.dumps(records, ensure_ascii=False, indent=2))

    print(f"[build] {len(canaries)} canaries → {len(records)} 行 → {LF_DATA}")
    for rep in sorted(per_bucket_lines):
        print(f"  rep={rep}: {per_bucket_lines[rep]} 行")
    print(f"[build] 记得在 LlamaFactory/data/dataset_info.json 注册 contam_canary_v1")


if __name__ == "__main__":
    main()
