# SPV-MIA 进阶实验：特异性 / 跨benchmark / 阈值标定

**日期**：2026-07-06
**入口 memory**：[[codec-reproduction-2026-07-06]] 之后的 SPV-MIA 深化
**依赖**：2026-06-27 SFT 污染 GT（6 ckpt）+ calibration v3（mean_only 基线）

## 0. 动机（一句话）

calibration v3 用 **mean_only** 模式（无对照集，signal=mean(Δpv) vs base），有两个硬伤：
① Finding 8 跨任务整体退化让 raw Δpv 全偏负、off-diagonal 假阳；② 无 member/non-member
标签 → 拿不到 ROC，连续 signal 无法变成可部署判定。本实验换 **per-sample AUC 协议**修这两点。

## 1. 方法

**协议**：member = 注入题（label 1），non-member = 同 benchmark 未注入的 held-out 题（label 0），
score = −Δpv，二分类算 AUC + ROC。member 与 non-member 同分布 → Finding 8 的整体退化两边
同时出现、抵消。`spv_mia()` 的 AUC 模式（`control_questions`）现成支持，evidence 已返回
per-sample `target_delta_pv` / `control_delta_pv`，ROC 在 `run_spv_roc.py` 后处理，**不改方法代码**。

**约束**：
- GSM8K：200/1319 注入，held-out 池 1119 → 同集 ROC 唯一干净战场
- HumanEval-Plus：164 全注入，同集无 held-out → 代码 non-member 借 MBPP+（外部集，分布差见 §3）
- MMLU-Pro：SPV-MIA 在 MC 上失效（F9），不进 ROC

**产物脚本**（本目录）：
- `dump_heldout.py` → `questions_cache/{gsm8k_heldout,mbpp_nonmember}.jsonl`（各 200）
- `run_spv_roc.py` + `run.yaml`（9 行矩阵）→ `outputs/2026-07-06_spv_specificity_roc/{ts}/{result.json,roc.csv,verdict.json}`
- `prep_specificity_data.py`（Phase B 训练数据）

## 2. 核心发现（Phase A，零训练）

### Finding 10（最重要）：mean(Δpv) 的"剂量响应"主要是**任务泛化**，不是记忆

member 与 held-out（未见）的 Δpv 中位数对比：

| ckpt | member Δpv | held-out Δpv | gap | per-sample AUC |
|---|---|---|---|---|
| gsm8k_light  | −14.0 | **−13.1** | −1.0 | 0.496 |
| gsm8k_medium | −22.2 | **−21.4** | −0.8 | 0.518 |
| gsm8k_heavy  | −65.2 | −48.2 | **−17.0** | 0.721 |
| **math_general**（Phase B）| **−36.4** | **−38.0** | +1.6 | **0.482** |

**未见过的 held-out 题，Δpv 几乎等于注入的 member（light −13 vs −14）**。说明 SFT 让模型对
整个 GSM8K 分布都变强，member 和 non-member 的 Δpv-vs-base 一起负 —— v3 记的 light=−13.47/
medium=−27.46「剂量响应」**测的是"模型学没学过这个 benchmark 的分布"，不是"记没记住这些具体题"**。
只有 heavy（50× verbatim）把特定题记忆到超出泛化基线 17 点，per-sample 才可分。

**math_general 是决定性铁证**（Phase B）：它训在 GSM8K **train** 上、泛化极强 → member Δpv=−36.4，
比 gsm8k_light(−14)/medium(−22) 都负得多。**若用 mean(Δpv) vs base，它看起来"污染"比 light/medium
还重——一个巨大假阳。** 但 per-sample AUC=0.482 正确判 CLEAN（member 与 held-out 都 −37，不可分）。
坐实：mean(Δpv) 测泛化，per-sample AUC 隔离记忆。

推论：
- **base 作 reference 会把任务泛化和记忆混在一起**（Finding 8 比"跨任务退化"更深：任何任务学习都移 Δpv）
- **per-sample AUC vs held-out 正确隔离记忆**：诚实报告 light/medium 无逐题记忆信号（AUC≈0.5），
  只 flag heavy
- 这本身就是内建特异性证据：还没训 math_general，held-out 对照已显示"纯分布学习 ≈ light 污染的 Δpv"

### Finding 11：per-sample AUC 的检测下限很高（只测得到 verbatim 重记忆）

ROC operating points（TPR@FPR，null 下 TPR=FPR）：

| ckpt | TPR@FPR=1% | TPR@FPR=5% |
|---|---|---|
| gsm8k_light  | 0.015 | 0.050（= null，无信号）|
| gsm8k_medium | 0.015 | 0.055（≈ null）|
| gsm8k_heavy  | 0.065 | 0.160（4× null）|

即便看长尾（低 FPR），light/medium 也压不出高于随机的 TPR。→ per-sample 协议的代价是
**灵敏度地板高**：3×/15× 曝光不可测，需 ~50× verbatim。与 mean(Δpv) 互补（mean 灵敏但混淆泛化）。

### Finding 12（负控 + 跨bench + 代码剂量 + 可部署裁决）

完整 9 行矩阵（Phase A + Phase B 全跑完）：

| row | ckpt × bench | AUC | gap(med) | verdict | 归属 |
|---|---|---|---|---|---|
| 3 | gsm8k_heavy × gsm8k | **0.721** | −17.0 | dirty | 剂量正例 ✓ |
| 2 | gsm8k_medium × gsm8k | 0.518 | −0.8 | clean | 剂量（漏报）|
| 1 | gsm8k_light × gsm8k | 0.496 | −1.0 | clean | 剂量（漏报）|
| 4 | clean × gsm8k | 0.490 | −0.0 | clean | 负控 ✓ |
| 5 | math_general × gsm8k | 0.482 | +1.6 | clean | **特异性FP ✓** |
| 7 | humaneval_heavy × HE(vs MBPP) | 0.648 | −5.7 | suspect | 代码正例（含混淆）|
| 8 | code_general × HE(vs MBPP) | 0.302 | +5.3 | clean | **特异性FP ✓ + 去混淆基线** |
| 6 | gsm8k_heavy × humaneval(split) | 0.580 | −2.6 | clean | 跨bench |
| 9 | humaneval_heavy × gsm8k | **0.469** | +0.7 | clean | 跨bench ✓ |

- **负控** clean×gsm8k：AUC=0.490 ✓（精准 null）
- **特异性（实验1，Phase B 收口）**：math_general×gsm8k=0.482、code_general×HE=0.302，两个"学了能力
  没见过原题"的 ckpt 全部 CLEAN ✓ → SPV-MIA per-sample 不把泛化误判成污染。math_general 尤其硬：
  它 member Δpv=−36 比 light/medium 都负，mean 口径会假阳，per-sample 判净。
- **代码去混淆**：row7(humaneval_heavy)=0.648 与 row8(code_general)=0.302 同为 HE-vs-MBPP 设置，
  差 row8 的分布基线（0.30 而非 0.5，因 HE/MBPP 分布不同）→ **humaneval_heavy 真实记忆信号 ≈ 0.648−0.302
  = 0.35 AUC**。row8 正是 §3-1 caveat 要求的对照，现补齐。
- **跨bench（实验2 核心结论）**：row9 humaneval_heavy×gsm8k=**0.469** 教科书级 null —— 污染 HumanEval
  完全不抬 gsm8k 信号。row6 gsm8k_heavy×humaneval=0.580（split 有序，见 §3-2），verdict 仍 clean。
  **→ 实验2 成立：污染只在被污染 benchmark 下降，不串扰。**

**可部署裁决（实验3 核心结论，硬前置#5 量化闭环）**：AUC 阈值 dirty≥0.70/suspect≥0.60，
9 格 confusion tp=2 fp=0 tn=5 fn=2 → **FPR=0.0 / FNR=0.5**。
- **FPR=0**：clean + 特异性(math/code_general) + 两个跨bench，5 个真负全部未误判 → 阈值 0.60 无假阳，
  可安全部署为"是否污染"上界判定
- **FNR=0.5**：light/medium 被漏（Finding 11 检测地板），verbatim 50× 以下测不到 —— 这不是 bug，
  是 per-sample 协议的诚实边界，需数据级 n-gram 兜底轻度污染

## 3. Caveat（写清楚，别自欺）

1. **代码 non-member 跨分布**：HumanEval 全注入，row7/row8 的 non-member 是 MBPP+，与 HumanEval
   member 分布不同。row7（humaneval_heavy）的 AUC 有分布混淆。**已解**：row8（code_general，同 HE-vs-MBPP
   但没见过 HE）给出分布基线 AUC=0.302，故 humaneval_heavy 真实记忆 ≈ row7−row8 = 0.648−0.302 = 0.35。
2. **跨bench split-half 是有序切分**（humaneval 前 82 vs 后 82），题目难度/长度可能有序差异，
   row6 的 0.58 部分可能来自此，不是真泄漏。改进：随机 shuffle 后再切。
3. **positive control 已到位**（污染 ckpt），但 Finding 10/11 表明「可部署阈值」只能覆盖 verbatim 重污染，
   light/medium 级仍需数据级 n-gram（`pretrain-data-eval/contamination/`）兜底。

## 4. Phase B（特异性对照训练，已完成 2026-07-06）

严格做污染 ckpt 的结构孪生，只把 benchmark test 原题换成 train/通用题：
- `math_general`：alpaca 5k + GSM8K **train** 200×5，10ep（与 gsm8k_heavy 同 recipe，train 与 test 零重叠）
- `code_general`：alpaca 5k + CodeAlpaca 164×5，10ep（与 humaneval_heavy 同 recipe）

**结果**：两者在各自 benchmark 上 AUC≈0.5（math_general 0.482 / code_general 0.302）→ 学了能力
没见过原题 → 不误报 ✓。math_general 是 Finding 10 的极端验证：泛化到 Δpv=−36 仍判 CLEAN。

训练在 8 卡容器跑（各 ~48min，10ep），merged 后回评测机跑行 5/8。踩坑记录：
- run_weekend.sh 未包 tmux/nohup → 首次被 SIGHUP 杀在 step 158（无报错、无 .done/.failed）；
  包 tmux 重启后 resume 从 checkpoint 续，成功。**教训：长训练必须 tmux/nohup。**
- merge 需 peft，仅 8 卡机有；评测机 conda 无 peft（不装，红线）→ merge 在 8 卡机跑。

## 5. 复现命令

```bash
# Phase A（评测，本机 GPU 或 8 卡机均可，transformers 4.57.1）
HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_CACHE=/mnt/public/code/chennuoxi/hf_cache \
PYTHONPATH=src python3 experiments/2026-07-06_spv_specificity_roc/dump_heldout.py
PYTHONPATH=src python3 experiments/2026-07-06_spv_specificity_roc/run_spv_roc.py
#   冒烟：SPV_ROC_ROWS=3,4 前缀只跑指定行

# Phase B（8 卡容器内，pip 升级仅训练机）
PYTHONPATH=src python3 experiments/2026-07-06_spv_specificity_roc/prep_specificity_data.py  # 数据（开发机跑，已完成）
cd /mnt/public/code/chennuoxi/LlamaFactory
ONLY="math_general code_general" bash examples/contam_sft_gt/run_weekend.sh   # 训练，每个 ~49min
python merge_lora.py math_general code_general                                 # 合并（base 路径已修）
# 回评测机加行 5/8：SPV_ROC_ROWS=5,8 python3 .../run_spv_roc.py（math_general/code_general ckpt 就绪后自动不跳过）
```
