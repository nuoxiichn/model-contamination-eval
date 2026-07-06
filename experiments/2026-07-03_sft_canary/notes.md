# 自 SFT canary 实验 v1 — notes

**缘起**：mentor review `docs/canary_协作请求.md` 指出核心缺陷——没证明方法有效，无法说服
训练团队配合埋点。解法不是复现论文（论文只证「别人配方下记忆现象成立」），而是用已有自造
SFT 基础设施自证：自己埋 canary、自己训、自己测召回。

**目标三合一**：
1. 召回率–注入重复次数曲线 + **饱和点**（回答「100 条 canary 各重复几次才可靠」）
2. clean/base recall=0 的 **FPR=0 实证**（把协作请求空话 `recall=k/N` 换成实测）
3. 顺手造一个 ground-truth 已知污染 ckpt（兼作 SPV/Oren/paraphrase 校准锚点）

## 状态（2026-07-03，session 15）

代码 + 数据全就位，**未训练**（等两个 paraphrase job 空出 GPU）。

- ✅ `canary.py` 实装（`generate_sft_canaries` + `measure_canary_recall`），10 单测 pass
- ✅ `gen_canaries.py` 跑通 → `data/canary/canary_v1.jsonl`（100 条唯一，output 全唯一）
- ✅ `build_sft_mix.py` 跑通 → `LlamaFactory/data/contam/canary_v1.json`（2880 行：20/60/200/600/2000）
- ✅ LlamaFactory：`canary_v1.yaml` + `dataset_info.json` 注册 `contam_canary_v1`（加性）
- ✅ `run_recall.py` 写好（ast + --help 通过）
- ⏳ 训练 + merge + recall 测量：待 GPU

## 实验设计

**自变量**：注入重复次数（dataset-level repeat），单轴（稀释率留 v2）。
5 桶 rep = 1/3/10/30/100×，各 20 条 → 100 条。有效曝光 = rep × 3 epoch = 3/9/30/90/300。

**⚠️ 已知取舍**：2880 注入行占训练集 ≈36%（100× 桶 2000 行为主）。不影响 per-bucket 召回
测量的有效性（每条 canary 独立 greedy 解码），但扰动训练分布 → 加**中性检查**（loss 曲线 +
canary ckpt vs clean 下游 gsm8k/mmlu 分）。若饱和点很早（rep≤10 就满），协作协议推荐低 rep，
真实注入回到「可忽略」。

**canary 设计（FPR=0 构造性）**：instruction 带唯一 uid（防撞车），output = 4 随机词典词
+ 4 位数字（如 `lantern-rowan-4657-loam-juniper`），与 instruction 无语义关联 → base 模型
不可能蒙中。明文 + seed + sha256 存档（`canary_v1.jsonl.meta.json`）。

## 执行步骤（GPU 空出后）

```bash
# 1. 训练（8 卡，LoRA 1.7B，预计 <1h）
cd /mnt/public/code/chennuoxi/LlamaFactory
bash examples/contam_sft_gt/submit_qwen3_1.7b.sh canary_v1

# 2. Merge LoRA → full model
llamafactory-cli export \
    --model_name_or_path /mnt/public/model/Qwen/Qwen3-1.7B-Base \
    --adapter_name_or_path saves/contam/canary_v1 --template qwen3 \
    --finetuning_type lora --export_dir saves/contam/canary_v1_merged --export_size 5

# 3. Recall 测量（canary + clean + base）
cd /mnt/public/code/chennuoxi/model-contamination-eval
HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=src \
python3 experiments/2026-07-03_sft_canary/run_recall.py --ckpt canary_v1,clean,base
```

**成功判据**：clean+base recall_contains=0/100（FPR=0）；canary ckpt 出现随 rep 单调爬升
的召回曲线 + 可辨识饱和点。

## 结果（待填）

| ckpt | B1(rep1) | B2(rep3) | B3(rep10) | B4(rep30) | B5(rep100) | 结论 |
|---|---|---|---|---|---|---|
| canary_v1 | — | — | — | — | — | — |
| clean | — | — | — | — | — | FPR |
| base | — | — | — | — | — | FPR |

## 回写目标（跑完后）

- `docs/canary_协作请求.md` §2.2：空话 `recall=k/N` → 实测 + 推荐 rep
- 若成功：canary ckpt 加入 anchor 池（`docs/positive_controls.md`）作构造性 positive control

## 相关

- 方法实装：`src/model_contamination/shared/canary.py` + `tests/test_canary.py`
- 理论对标：Secret Sharer（Carlini 2019 USENIX）exposure 指标
- 复用：`prep_bench_to_sft.py` repeat 展开 / `clean.yaml` 配方 / `evaluator._wrap_chat` chat template
