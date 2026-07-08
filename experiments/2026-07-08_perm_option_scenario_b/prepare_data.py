"""复现 Ni et al. AAAI 2025《Training on the Benchmark Is Not All You Need》
Scenario b（Algorithm 2）在 Qwen 上的实验 —— 数据准备。

把论文四个中文/对照 MC benchmark 导出为统一 JSONL（每行一道题：
{id, benchmark, question, choices, answer_index}），供 run_scenario_b.py 消费。
统一在这里落盘，避免 runner 每次重连 HF。

数据源与加载方式（均经开发机探测确认，2026-07-08）：
    c-eval   ceval/ceval-exam        52 学科 config，val split 带 answer，4 选项
    cmmlu    haonan-li/cmmlu         脚本式数据集（datasets 5.0 拒绝）→ 直接下 zip
                                     读 test/<subject>.csv，67 学科，4 选项
    cmb      FreedomIntelligence/CMB CMB-Exam config，builder 生成 train 会崩 →
                                     streaming 只取 test，filter 单项选择题，
                                     option 是 JSON dict（4~5 选项）
    mmlu-cf  microsoft/MMLU-CF       已缓存 + loader 已注册（抗污染对照），走 loader

运行（开发机，联网）：
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HOME=/mnt/public/code/chennuoxi/hf_cache \
    PYTHONPATH=src python3 experiments/2026-07-08_perm_option_scenario_b/prepare_data.py

    # 每 benchmark 抽样上限（控制 n! × 题数 的推理成本）：
    ... prepare_data.py --per-benchmark 300
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs" / "data"

# 论文 Table：四个 benchmark。mmlu-cf 是抗污染对照（期望低泄漏）。
_CEVAL_REPO = "ceval/ceval-exam"
_CMMLU_REPO = "haonan-li/cmmlu"
_CMB_REPO = "FreedomIntelligence/CMB"


def _write_jsonl(name: str, rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[✓] {name}: wrote {len(rows)} questions → {path}")


def _record(bench: str, qid: str, question: str, choices: list[str], answer_index: int) -> dict:
    return {
        "id": f"{bench}-{qid}",
        "benchmark": bench,
        "question": question.strip(),
        "choices": [str(c).strip() for c in choices],
        "answer_index": answer_index,
    }


# --------------------------- per-benchmark loaders --------------------------- #


def prepare_ceval(per_benchmark: int | None) -> list[dict]:
    """52 学科各取 val split（带 answer），拼一起后按 per_benchmark 截断。"""
    from datasets import get_dataset_config_names, load_dataset

    letters = "ABCD"
    subjects = get_dataset_config_names(_CEVAL_REPO)
    rows: list[dict] = []
    for subj in subjects:
        ds = load_dataset(_CEVAL_REPO, name=subj, split="val")
        for i, row in enumerate(ds):
            letter = str(row["answer"]).strip().upper()
            if letter not in letters:
                continue
            rows.append(
                _record(
                    "c-eval", f"{subj}-{i}", row["question"],
                    [row["A"], row["B"], row["C"], row["D"]],
                    letters.index(letter),
                )
            )
    return _truncate(rows, per_benchmark)


def prepare_cmmlu(per_benchmark: int | None) -> list[dict]:
    """下载 cmmlu_v1_0_1.zip，读 test/<subject>.csv。

    csv 格式（探测确认）：header = ['','Question','A','B','C','D','Answer']，
    首列是行号。67 学科 test。
    """
    from huggingface_hub import hf_hub_download

    letters = "ABCD"
    zip_path = hf_hub_download(_CMMLU_REPO, "cmmlu_v1_0_1.zip", repo_type="dataset")
    rows: list[dict] = []
    with zipfile.ZipFile(zip_path) as z:
        test_csvs = sorted(n for n in z.namelist() if n.startswith("test/") and n.endswith(".csv"))
        for name in test_csvs:
            subj = name[len("test/"):-len(".csv")]
            with z.open(name) as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
                for i, row in enumerate(reader):
                    letter = str(row.get("Answer", "")).strip().upper()
                    if letter not in letters:
                        continue
                    rows.append(
                        _record(
                            "cmmlu", f"{subj}-{i}", row["Question"],
                            [row["A"], row["B"], row["C"], row["D"]],
                            letters.index(letter),
                        )
                    )
    return _truncate(rows, per_benchmark)


def prepare_cmb(per_benchmark: int | None) -> list[dict]:
    """CMB-Exam streaming（builder 生成 train 崩，只取 test），保留单项选择题。

    option 是 JSON dict {"A": "...", ...}（4~5 键）。CMB **test split 不含 answer**
    （留作 leaderboard），但 Algorithm 2 不需要金标答案 —— 只排列选项检测离群，
    故 answer_index 置 0 占位（下游 option_permutation_test 不读它）。
    多选题（question_type 非「单项选择」）跳过。
    """
    from datasets import load_dataset

    letters = "ABCDE"
    it = load_dataset(_CMB_REPO, name="CMB-Exam", split="test", streaming=True)
    rows: list[dict] = []
    for i, row in enumerate(it):
        qtype = str(row.get("question_type", ""))
        if "单项选择" not in qtype:
            continue
        opt = row.get("option")
        if isinstance(opt, str):
            opt = json.loads(opt)
        if not isinstance(opt, dict):
            continue
        keys = [k for k in letters if k in opt and str(opt[k]).strip()]
        if len(keys) < 2:
            continue
        rows.append(
            _record(
                "cmb", str(row.get("id", i)), row["question"],
                [opt[k] for k in keys],
                0,  # 占位：test split 无 answer，Algorithm 2 不需要
            )
        )
    return _truncate(rows, per_benchmark)


def prepare_mmlu_cf(per_benchmark: int | None) -> list[dict]:
    """抗污染对照：走已注册 loader（microsoft/MMLU-CF 已缓存）。

    MMLU-CF 无公开 test split（留作 leaderboard），用 val（带 Answer）。
    """
    sys.path.insert(0, str(HERE.parents[1] / "src"))
    from model_contamination.benchmarks.loader import load_questions
    from model_contamination.benchmarks.registry import load_registry

    spec = load_registry().get("mmlu-cf")
    qs = load_questions(spec, split="val", limit=per_benchmark)
    rows = [
        _record("mmlu-cf", q.id, q.prompt, q.choices or [], q.answer_index or 0)
        for q in qs
        if q.choices and q.answer_index is not None
    ]
    return rows


def _truncate(rows: list[dict], per_benchmark: int | None) -> list[dict]:
    if per_benchmark is None or len(rows) <= per_benchmark:
        return rows
    # 均匀跨学科抽样：按 stride 取，避免只取到前几个学科
    stride = len(rows) / per_benchmark
    return [rows[int(i * stride)] for i in range(per_benchmark)]


_PREPARERS = {
    "c-eval": prepare_ceval,
    "cmmlu": prepare_cmmlu,
    "cmb": prepare_cmb,
    "mmlu-cf": prepare_mmlu_cf,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--benchmarks", nargs="+", default=list(_PREPARERS),
        choices=list(_PREPARERS),
        help="要导出的 benchmark（默认全部四个）",
    )
    ap.add_argument(
        "--per-benchmark", type=int, default=300,
        help="每 benchmark 题数上限（None=全量）。默认 300 控制 n!×题数 推理成本。",
    )
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    args = ap.parse_args()

    if "HF_ENDPOINT" not in os.environ:
        print("[!] 未设 HF_ENDPOINT；建议 export HF_ENDPOINT=https://hf-mirror.com", file=sys.stderr)

    out_dir = Path(args.out_dir)
    limit = None if args.per_benchmark <= 0 else args.per_benchmark
    for bench in args.benchmarks:
        print(f"[i] preparing {bench} (limit={limit}) …")
        rows = _PREPARERS[bench](limit)
        _write_jsonl(bench, rows, out_dir)


if __name__ == "__main__":
    main()
