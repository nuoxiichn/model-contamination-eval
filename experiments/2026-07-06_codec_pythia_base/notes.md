# CoDeC 小规模复现结论 — Pythia-2.8b (base)

论文：Detecting Data Contamination in LLMs via In-Context Learning (arXiv:2510.27055, CoDeC)
方法实现：`src/model_contamination/shared/codec.py`（method tag `codec`）
运行：`run_codec.py`（配置见 `run.yaml`），结果 `outputs/codec_results.json`（不进 git）

## 配置

- 模型：Pythia-2.8b（`hf_cache/models/C2_pythia-2.8b`，Pile 训练，天然是论文主力对照）
- CoDeC 参数（论文默认）：n_context=1, n_seeds=5, skip_first_tokens=10
- 每数据集 300 样本（论文称 100 即稳）
- seen：`NeelNanda/pile-10k` 按 `meta.pile_set_name` 过滤（ArXiv / Wikipedia），切 600 字符 chunk
  - 论文用 gated 的 `iamgroot42/mimir`，需 HF token 授权（碰红线），改用非 gated 同源 Pile 采样替代
- unseen：gsm8k / mmlu-pro / math-500（均发布晚于 Pythia 训练，Pythia 未见）

## 结果

| 组 | 数据集 | CoDeC signal | verdict |
| --- | --- | --- | --- |
| unseen | gsm8k | 0.027 | clean |
| unseen | mmlu-pro | 0.163 | clean |
| unseen | math-500 | 0.178 | clean |
| seen | pile-ArXiv | 0.793 | suspect |
| seen | pile-Wikipedia | 0.970 | dirty |

- seen_mean = 0.882，unseen_mean = 0.122
- seen_min (0.793) > unseen_max (0.178) → **cleanly_separated = true**

## 与论文对照

- **核心现象复现**：seen 数据 CoDeC 分数显著高、unseen 显著低、两组完全分离（论文 Fig 3 / Table 1 的 AUC≈100% 本质就是「seen 全高于 unseen」）。
- **数值吻合**：论文 Pythia 上 seen≈100%、unseen<60%。本次 seen 0.79/0.97、unseen 0.03–0.18，量级一致。
  Wikipedia (0.97) 高于 ArXiv (0.79) 也符合论文观察——ArXiv 更专业、公式符号多、样本间共享结构弱，memorization 信号略低（论文 §A.3.2「数据多样性影响」）。
- **绝对阈值**：论文经验阈值 >80% DIRTY / 60–80% SUSPECT。ArXiv 落到 SUSPECT 是数据多样性所致，不代表未污染——正是论文强调「中间分数须跨模型对照解读」的场景。本仓库 positive control 未到位，verdict 仅作 hint。

## 实现要点与踩坑

- baseline 的 prefix 用 `\n\n`（分隔符）而非空串：Pythia (gpt_neox) tokenizer 无 BOS，空 prompt 编码出 0 token 会让 `HFLocalModel.logprobs` 崩（shape mismatch）。用分隔符既不引入数据信息，又与 in-context 版「x 前紧接 \n\n」的分词对齐，Δ 更可比。
- CoDeC 只看 Δ 符号，与 logit 尺度无关 → 模型无关性天然成立，无需 calibration。
- 单测 8 个（`tests/test_codec.py`，fake model），全量回归 167 passed 无破坏。

## 后续可扩展（本次未做，论文附录）

- 更多 seen/unseen 数据集 + 更多模型（GPT-Neo/RWKV/OLMo）算 dataset-level AUC，对齐 Table 1。
- finetuning 注入污染 → CoDeC 应升到 >90%（论文 Fig 6）。
- 与仓库现有 oren/paraphrase/spv_mia 做方法对照（CoDeC 是正交的第 5 个信号源）。
