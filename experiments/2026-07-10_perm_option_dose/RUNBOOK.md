# RUNBOOK —— perm_option FP + dose 在 8 卡机执行

两实验都在 8 卡 MetaX 机跑（开发机无 GPU）。权重/数据全部已缓存，无需联网。
**按顺序过 gate，不要跳。** 每个 gate 失败就停，不要「先跑起来再说」。

环境：
```bash
cd /mnt/public/code/chennuoxi/model-contamination-eval
export HF_HOME=/mnt/public/code/chennuoxi/hf_cache
export PYTHONPATH=src
```

---

## 实验一：FP（纯推理，无训练，风险低）

**Gate F1 — batch 数值一致性**：FP 用 scenario_b 同一个 `seq_logprob_sums` 分桶 batch
前向，scenario_b 已过 gate（rel<1e-3 + argmax 一致），**无需重验**。

**Gate F2 — 冒烟**（1.5B 单卡，1 对照，20 题，~1min）：
```bash
CUDA_VISIBLE_DEVICES=0 MACA_VISIBLE_DEVICES=0 \
  python3 experiments/2026-07-10_perm_option_fp/run_fp.py \
  --models qwen2.5-1.5b --benchmarks mmlu-cf --limit 20 --csv-name fp_smoke.csv
```
看到 `FPR(leak_frac)=...` 且非报错即过。删掉 `outputs/results/fp_smoke.csv` 与
`qwen2.5-1.5b__mmlu-cf.json`（冒烟产物）再跑正式。

**正式跑**（tmux 全矩阵）：
```bash
bash experiments/2026-07-10_perm_option_fp/launch_fp_tmux.sh
tmux attach -t perm_fp
```
产出 `outputs/results/fp.csv`。回填 `notes.md` 结果表。

---

## 实验二：dose（72B LoRA 训练，风险高，务必先冒烟）

剂量轴 = **曝光次数 / epoch**（一趟训到 max_epochs，每 epoch 后测 leak_fraction，得
leak-vs-epoch 曲线）。单趟轨迹、单个 adapter → **无跨剂量 unload 重置**，卸载风险已消除。

**Gate D1 — peft 可用**：
```bash
python3 -c "import peft; print('peft', peft.__version__)"
```
缺则 `pip install peft`（trl 环境通常自带）。

**Gate D2 — 机制确认（1.5B 单卡，纯注入 E=10，~10-20min）**：
这是**决定性 gate**。1.5B 混合比例冒烟（2026-07-10）已证伪 p 轴（p=0.2 leak 0.32 <
p=0 leak 0.48，欠曝光噪声）。改 epoch 轴后，先验**机制本身**：纯注入（无 filler）训到
10 epoch，leak_fraction 能否被抬离 epoch-0 地板。
```bash
CUDA_VISIBLE_DEVICES=0 MACA_VISIBLE_DEVICES=0 \
  python3 experiments/2026-07-10_perm_option_dose/run_dose_lora.py \
  --model-path Qwen/Qwen2.5-1.5B --model-name qwen1.5b-mech --device-map '' \
  --filler-mult 0 --max-epochs 10 --n-questions 200 --tag mech
```
**判据**：
1. 跑完输出 `outputs_mech/dose_summary.json`，含 epoch 0..10 的 leak_fraction 曲线；
2. **leak_fraction 随 epoch 明显上升**（epoch 10 显著 > epoch 0），Spearman rho > 0；
3. epoch 0 落在 1.5B FP 地板（~0.45，对照 FP 实验）。

- **若 rho>0 且 epoch10 明显抬升** → 机制成立、实现正确 → 上 72B 正式跑。
- **若纯注入 E=10 都几乎不动** → 1.5B 在方法分辨率以下（scenario_b：1.5B 真污染分离
  仅 +0.018）。**换 7B 重验机制**（同命令 `--model-path Qwen/Qwen2.5-7B`，7B 单卡或
  `--device-map auto`）再决定是否上 72B，不要盲目烧 72B 卡时。
- 冒烟产物在 `outputs_mech/`，与正式 `outputs/` 隔离，可直接删。

**正式跑**（72B，8 卡 device_map=auto，一趟训到 10 epoch 带 filler，长跑）：
```bash
bash experiments/2026-07-10_perm_option_dose/launch_dose_tmux.sh
tmux attach -t perm_dose
```
逐 epoch 落盘 `outputs/leak_vs_epoch.json`（中途可看曲线）。产出 `outputs/dose_summary.json`
（含 Spearman rho）。回填 `notes.md` 结果表。

---

## 回写

两实验 `notes.md` 结果表 + 结论填完后，摘要回写 `../docs/experiments/`（附本仓库
commit hash + experiments 目录名），遵守仓库红线：无 positive control → 只报相对趋势
（FP 分解、dose 单调性），不出绝对红黄绿。
