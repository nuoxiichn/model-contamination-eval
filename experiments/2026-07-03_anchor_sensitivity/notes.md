# E1.1 Anchor 灵敏度扫描 — notes

**目标**：用开源已知污染模型（anchor）验证每个检测方法的 TPR（脏 benchmark 命中率）/
FPR（clean 对照误报率），产出灵敏度矩阵（实验计划-v2.md §E1.1.7），据此剔除失败方法。

**状态**：runner 已写好并静态验证（2026-07-03，session 15）。**尚未跑**（等 anchor 权重下完 +
GPU 空出；device 0/1 被 paraphrase job 占用中）。

## 已就位

- `run_anchor.py` — CLI 驱动，单 anchor × N bench × M 方法。ast + `--help` 通过。
- `run.yaml` — anchor 池 + bench↔control 映射的权威溯源（runner 本身不读它）。
- 4 方法默认集：`oren` / `paraphrase` / `mink_plus_plus` / `perm_option`。
- **SPV-MIA 默认不跑**：anchor 是单一开源 base，无「同源未污染」reference，方法学上
  不能进 E1.1 盲扫（属 E1.2 专项）。预留 `--reference-model` 钩子，传了才跑。
- **self_critique 不跑**：仍是 Phase 3 `NotImplementedError` stub。

## 怎么跑（GPU 空出后）

```bash
# E1.1.2 首发：A1 Qwen3-1.7B-Base（本地已有）
HF_ENDPOINT=https://hf-mirror.com \
HF_DATASETS_CACHE=/mnt/public/code/chennuoxi/hf_cache \
PYTHONPATH=src python3 experiments/2026-07-03_anchor_sensitivity/run_anchor.py \
    --model A1 --bench gsm8k,mmlu-pro --n-samples 200 --device cuda

# Phi-3 需 --trust-remote-code；续跑加 --resume <out_dir>
```

单卡一个 anchor，多 anchor 分卡并行（A1/A2/A3/C1/C2 → device 2-6）。

## 控制集现状（2026-07-03 补下载后）

| target bench | control | 状态 | 备注 |
|---|---|---|---|
| gsm8k | gsm1k | ✓ n=50 | **GSM1k 全量未释放**（ScaleAI 设条件：3 个不同血统开源模型达 95%+ 才放全量，至今未满足）。仅 `scaleapi/gsm1k_eval` 的 50 条公开样本，已转 `data/gsm1k.jsonl` |
| mmlu-pro | mmlu-cf | ✓ n=200 | 用 **val split**（MMLU-CF 无 test split，runner `BENCH_SPLIT` 覆盖）|

**n=50 的软限制**：gsm8k anchor 的 mink++/spv_mia AUC 用 200 target vs 50 control，统计力偏弱。
可扩展替代 GSM-Plus **暂不可用**（loader `_NORMALIZERS` 未注册 gsm-plus，需先加 normalizer）。
若要更强 AUC，后续在 loader 加 gsm-plus normalizer，或等 GSM1k 全量释放。

## 阻塞 TODO（真跑前）

1. ~~gsm1k 控制集缺失~~ —— ✓ 已补 50 条公开样本（见上表）。
2. **anchor 权重下载**（`experiments/anchor_download.log` + `pythia_retry.log`）：
   - ✅ A3 Phi-3-mini（DONE）
   - ✅ C2 Pythia-2.8B（首次网络断，重试 safetensors-only DONE）
   - ⏳ A2 Mistral-7B-v0.1（下载中，**gated 未阻挡** —— hf-mirror 免 token 供）
   - ⏳ C1 OLMo-2-7B（Mistral 后排队）
   - A1 Qwen3-1.7B-Base — 本地已有，不下。

## 验收（实验计划-v2.md §E1.1.2）

- 灵敏度：GSM8K（dirty）≥ 4/6 方法 signal 显著；GSM1k（clean）≤ 1/6 方法误报
- 阴性对照 C1/C2：全 bench × 全方法 → ≤5% 位置 signal 显著（FPR 上限）
- 汇总 → `docs/method_sensitivity_matrix.md`（E1.1.7）

## 设计取舍

- **取题不走 SFT manifest**：anchor 从 benchmark 全量取前 n（`load_questions(spec, limit=n)`），
  缓存到 `outputs/2026-07-03_anchor_sensitivity/_qcache/`。因 anchor 场景没有 SFT 注入子集。
- **resume 到 (bench, method) cell 粒度**：Mistral-7B ~6h，细粒度续跑防大 job 中断丢进度。
- **单方法异常隔离**：某方法炸只记 error 到该 cell，不拖垮整轮（try/except 包 dispatch）。
