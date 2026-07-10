"""perm_option FP 实验 —— 合成干净对照生成（开发机跑，纯 numpy，无 GPU）。

从 scenario_b 已导出的真数据派生两个「保证未记忆选项顺序」的合成对照，用于把
perm_option 的假阳率分解成「纯 IsolationForest 地板」vs「流畅度膨胀」两部分：

  synth-random           每题 4 个**随机等长 token 串**选项（从真 benchmark 的字符表
                         采样，等长、无语义）。任何排列都同样「不流畅」→ 若地板由流畅度
                         驱动，这里应显著低于 mmlu-cf。
  synth-fluent-mismatch  每题 4 个**真·流畅**选项，但从**不同题**随机各取一个组装
                         （选项个体流畅、但这组选项从未在训练里共现为一个答案集）→
                         隔离「单选项流畅」与「共现记忆」。

两者都从真数据的题面/选项池派生，天然落在 Qwen 词表内（不引入 OOV 噪声）。
mmlu-cf 直接从 scenario_b 复制（真·抗污染基准，选项流畅）。

**这三个对照对任何模型都是干净的**（模型不可能记住随机串顺序、也不可能记住错配集），
故 leak_fraction 一律读作假阳率。

运行（开发机）：
    PYTHONPATH=src /opt/conda/envs/OmniModelEval/bin/python \
        experiments/2026-07-10_perm_option_fp/make_fp_controls.py --per-benchmark 300
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs" / "data"
# scenario_b 已导出的真数据源（复用，不重新联网下载）
SCENARIO_B_DATA = (
    HERE.parent / "2026-07-08_perm_option_scenario_b" / "outputs" / "data"
)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(name: str, rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[✓] {name}: wrote {len(rows)} questions → {path}")


def _char_pool(rows: list[dict]) -> list[str]:
    """真数据全部选项文本的字符集（去空白），作随机串采样表 → 落在 Qwen 词表内。"""
    chars: set[str] = set()
    for r in rows:
        for c in r.get("choices", []):
            for ch in str(c):
                if not ch.isspace():
                    chars.add(ch)
    return sorted(chars)


def make_synth_random(src_rows: list[dict], rng: np.random.Generator) -> list[dict]:
    """每题 4 个随机等长（=该题真选项平均字符长）token 串选项，题面沿用真题面。

    等长保证「选项块」token 数在各排列间高度一致，把长度 confound 摘掉，只留
    IsolationForest 对『N 个近似同分布 logprob 里挑最大值』的固有假阳。
    """
    pool = _char_pool(src_rows)
    out: list[dict] = []
    for r in src_rows:
        real = [str(c) for c in r.get("choices", []) if str(c).strip()]
        if len(real) < 4:
            continue
        # 目标长度 = 该题真选项字符长的中位数（保持题内一致，跨题有分布）
        L = int(np.median([len(c) for c in real])) or 4
        choices = [
            "".join(pool[int(j)] for j in rng.integers(0, len(pool), size=L))
            for _ in range(4)
        ]
        out.append({
            "id": f"synth-random-{r['id']}",
            "benchmark": "synth-random",
            "question": r["question"],
            "choices": choices,
            "answer_index": 0,   # 占位：Algorithm 2 不看金标
        })
    return out


def make_synth_fluent_mismatch(
    src_rows: list[dict], rng: np.random.Generator
) -> list[dict]:
    """每题 4 个真·流畅选项，但从**不同源题**各随机取一个（跨题错配）。

    选项个体是真文本（流畅、在词表内），但这 4 个从未作为一个答案集在训练中共现 →
    模型不可能记住它们的任何特定顺序 → 仍是干净对照，但保留「单选项流畅度」。
    """
    # 全局选项池（打平所有题的选项），带来源题标记避免同题四连
    pool: list[tuple[str, str]] = []
    for r in src_rows:
        for c in r.get("choices", []):
            if str(c).strip():
                pool.append((r["id"], str(c).strip()))
    if len(pool) < 8:
        return []

    out: list[dict] = []
    for r in src_rows:
        picks: list[str] = []
        used_src: set[str] = set()
        guard = 0
        while len(picks) < 4 and guard < 1000:
            guard += 1
            src_id, text = pool[int(rng.integers(0, len(pool)))]
            if src_id == r["id"] or src_id in used_src or text in picks:
                continue  # 不取本题原选项、不同源题去重
            used_src.add(src_id)
            picks.append(text)
        if len(picks) < 4:
            continue
        out.append({
            "id": f"synth-mismatch-{r['id']}",
            "benchmark": "synth-fluent-mismatch",
            "question": r["question"],
            "choices": picks,
            "answer_index": 0,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source", type=str, default="mmlu-cf",
        help="派生合成对照所用的真数据源（scenario_b 导出的 benchmark 名）",
    )
    ap.add_argument(
        "--per-benchmark", type=int, default=300,
        help="每对照题数上限（合成对照按源题逐条派生，源截断后即上限）",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src_path = SCENARIO_B_DATA / f"{args.source}.jsonl"
    if not src_path.exists():
        raise FileNotFoundError(
            f"缺源数据 {src_path}；请先在 scenario_b 目录跑 prepare_data.py 导出 {args.source}"
        )
    src_rows = _read_jsonl(src_path)
    if args.per_benchmark > 0:
        src_rows = src_rows[: args.per_benchmark]
    print(f"[i] source={args.source}  n={len(src_rows)}")

    rng = np.random.default_rng(args.seed)

    # 1) mmlu-cf 真对照：直接复制（截断到 per_benchmark 保持题数一致）
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.source == "mmlu-cf":
        _write_jsonl("mmlu-cf", src_rows)
    else:
        shutil.copy(SCENARIO_B_DATA / "mmlu-cf.jsonl", OUT_DIR / "mmlu-cf.jsonl")
        print("[✓] mmlu-cf: copied from scenario_b")

    # 2) synth-random
    _write_jsonl("synth-random", make_synth_random(src_rows, rng))
    # 3) synth-fluent-mismatch
    _write_jsonl("synth-fluent-mismatch", make_synth_fluent_mismatch(src_rows, rng))

    print("[done] FP controls ready in", OUT_DIR)


if __name__ == "__main__":
    main()
