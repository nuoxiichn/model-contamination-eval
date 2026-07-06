# CoDeC finetuning 注入实验 — Pythia-2.8b

论文：Detecting Data Contamination in LLMs via In-Context Learning (arXiv:2510.27055) §3.3 / Fig 6
方法实现：`src/model_contamination/shared/codec.py`（method tag `codec`）
脚本：`run_inject.py`（配置 `run.yaml`），结果 `outputs/`（不进 git）
前置实验：`../2026-07-06_codec_pythia_base/`（base 分数基线）

## 实验设计

对每个目标数据集，独立起一份干净 Pythia-2.8b，在该数据集**题面文本**上全参 finetune
（lr=3e-5, bf16, 60 batch, batch_size=8），训练过程每 10 batch 算一次 CoDeC。
预期：分数从 base 值稳定升到 >90%（论文 Fig 6），证明 finetune 引入的污染被 CoDeC 检测。

- 训练与检测用同一批题面文本（CoDeC 的 `_codec_text` 取 `q.prompt`）→「训练在什么文本、就在什么文本上检测」。
- CoDeC 口径与前一实验完全一致：n_context=1, n_seeds=5, skip=10, 300 样本。
- 三个目标（base CoDeC 分数）：gsm8k(0.027) / mmlu-pro(0.163) / math-500(0.178)。

## 运行命令（8 卡 MetaX 机）

模型在 hf_cache（两机共享），脚本纯 torch，不碰 llamafactory。单卡串行即可
（脚本内部一个目标跑完释放显存再跑下一个）。

```bash
cd /mnt/public/code/chennuoxi/model-contamination-eval

# 单卡（选一张空闲卡，如卡 0），后台跑，日志落盘
MACA_VISIBLE_DEVICES=0 \
HF_ENDPOINT=https://hf-mirror.com \
HF_HOME=/mnt/public/code/chennuoxi/hf_cache \
PYTHONPATH=src \
  nohup python3 experiments/2026-07-06_codec_finetune_inject/run_inject.py \
  > experiments/2026-07-06_codec_finetune_inject/run.log 2>&1 &

# 看进度（每 10 batch 一行 CoDeC=... ）
tail -f experiments/2026-07-06_codec_finetune_inject/run.log

# 查卡占用
mx-smi
```

预计时长：3 目标 × 60 batch × (每 10 batch 一次 300 样本 CoDeC eval)，Pythia-2.8b bf16，
约 1–2 小时。中途每个目标完成会写 `outputs/curve_{name}.json`，全部完成写 `outputs/inject_summary.json`。

若单卡显存不足（Pythia-2.8b 全参 + Adam ≈ 30GB+，MXC500-64G 应够）：
调小 `run.yaml` 的 `batch_size`（8→4→2），已开 `gradient_checkpointing`。

## 结果

三目标全部复现论文核心结论：base 分数低（0.03–0.20）→ finetune 后稳定升到 >90%。

| 目标 | base CoDeC | final CoDeC | reached_90 |
| --- | --- | --- | --- |
| gsm8k | 0.033 | 0.953 | ✓ |
| mmlu-pro | 0.186 | 0.946 | ✓ |
| math-500 | 0.205 | 0.940 | ✓ |

完整曲线（batch → CoDeC）：

| batch | 0 | 10 | 20 | 30 | 40 | 50 | 60 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k | 0.033 | 0.847 | 0.913 | 0.900 | 0.927 | 0.953 | 0.953 |
| mmlu-pro | 0.186 | 0.864 | 0.895 | 0.905 | 0.922 | 0.936 | 0.946 |
| math-500 | 0.205 | 0.829 | 0.889 | 0.916 | 0.940 | 0.956 | 0.940 |

（在 dev 机 MetaX C500 64GB 上跑，全参 bf16，约 20 分钟三目标。base 分数与前一实验
基线一致：gsm8k 0.033≈0.027、mmlu-pro 0.186≈0.163、math-500 0.205≈0.178。）

## 与论文对照

- **Fig 6 复现**：finetune 后所有数据集 CoDeC 稳定 >90%（论文原文 "consistently rose above 90%"）。三目标 final 0.94–0.95，吻合。
- **Fig 8 复现**：污染在头几个 batch 就快速形成——10 batch 内三条曲线都从 base 跳到 ~0.85，之后缓慢爬升到平台。论文强调「即使很小的更新也会在 contaminated 样本上早早降低 confidence」，这里 base→b10 的陡升正是此现象。
- **跨数据集一致**：三个不同类型（小学数学 / 多学科选择 / 竞赛数学）曲线形态一致，印证论文「CoDeC 对 finetune 注入的检测与数据集无关」。
- **意义**：这条正是论文用来「绕开对原训练语料的未知」的验证路径——无论模型原本是否见过，只要在某数据集上 finetune，CoDeC 就会拉高，从而可用于任意模型（含闭源训练数据的近期模型，论文用 Qwen3 演示）。本实验在本仓库自己的 Pythia + 自己的 benchmark loader 上独立复现了这一点。

## 环境注意（8 卡机，见 memory env-facts-8card-metax）

- transformers==5.6.0：Pythia 是 gpt_neox，无 Phi-3 rope_scaling 坑；脚本用标准 HF API，不碰 Vision2Seq/llamafactory。
- 设备隔离用 `MACA_VISIBLE_DEVICES`，查卡 `mx-smi`（非 nvidia-smi）。
- 全参训练用 bf16（`run.yaml` model.dtype），避免 fp16 反向数值不稳。

## Smoke test（已在 dev 机验证）

用 tiny-random-gpt2 + 6 样本 + 2 batch（CPU）验证脚本逻辑：训练 loop 跑通、loss 下降、
CoDeC 曲线正确记录 3 个点。绝对分数无意义（随机模型），仅验证 plumbing。
