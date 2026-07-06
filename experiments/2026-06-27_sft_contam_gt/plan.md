# SFT 阶段污染 Ground Truth 制作计划（v1）

**日期**：2026-06-27（周末 launch）
**目标**：制造已知污染状态的 SFT checkpoint 用作方法学 calibration 锚点。

> **2026-06-29 pivot 后注（必读）**：本 plan 写于 guided 仍在流水线时。pivot 决策后
> guided 已删除，MinK-pp 信号待 pretrain GT 验证。**plan 内涉及 guided 的描述保留
> 作历史，最新结论与下一步以 notes.md §0–§5 为准**。
**Why**：方案中所有检测方法（Min-K%++ / paraphrase / SPV-MIA 等）的阈值
来自论文 default，未做 positive control 校准。CLAUDE.md 硬前置 #5 缺口的最小可行补丁。
**How to apply**：本批次 GT 出来后，把阈值从论文 default 切换到 calibration 落点；
报告生成器才可以从 ranked list 升级到红/黄/绿绝对裁决（待后续 PR）。

---

## 1. 三个 axis 的全集（**项目长期范围**）

下面列出方法学**应当覆盖的全部维度**，不是本批次都做。本批次的取舍见 §2。

### Axis A：污染**形态**（注入方式）

| 标签 | 形态 | 描述 | 真实出现概率 | 检测难度 |
|---|---|---|---|---|
| **A** | 原文 SFT | benchmark 题面 + 完整 CoT + 原答案，作为 instruction-output 对入 SFT | 数据治理粗暴时常见 | 易 |
| **B** | Answer-leakage SFT | 题面正常，output 透露 ground truth 模式（"the answer is 18"） | 用 LLM 蒸馏 SFT 数据时常见 | 中 |
| **C** | Rephrased SFT | 题面被 LLM 改写，答案保留 | "数据增强"漏掉去重时常见 | 难 |
| **D** | Pretrain-style 污染 | benchmark 文本以纯文本（非 instruction 格式）进 continued-pretraining | web 爬虫 | 难 |
| **E** | Few-shot leakage | benchmark 题作为 instruction 里的 in-context demo | 模板设计失误 | 中 |
| **F** | Answer-only | 题面被 dedup 掉、答案模式留下 | dedup pipeline bug | 难 |

### Axis B：污染**强度**（占比 × epoch 数）

按 base SFT 数据 5000 条（alpaca_en 随机抽 seed=42）算占比：

| 档 | 注入 | 副本 | 总数据 | 实际占比 | epochs | 模型见同题次数 | 用途 |
|---|---|---|---|---|---|---|---|
| Clean | — | — | 5000 | 0% | 3 | 0 | negative control |
| Light | 200 | 1 | 5200 | **3.8%** | 3 | 3 | 弱信号下限 |
| Medium | 200 | 5 | 6000 | **16.7%** | 3 | 15 | 中量污染 |
| Heavy | 200 | 5 | 6000 | **16.7%** | 10 | 50 | 阈值上限锚（拉训练强度轴） |

注：HumanEval-Plus 题数上限 164，故对应组占比 = 820/5820 = **14.1%**。

### Axis C：**Benchmark 类型**（题型）

| 类型 | 代表 benchmark | 题面/答案特征 | 项目可信度报告主表中的 entry |
|---|---|---|---|
| 数学 / 长 CoT | GSM8K, MATH | 自然语言题面 + 多步推理答案 | MATH, MGSM, AIME-2025 |
| 通用知识 / MC | MMLU-Pro | 短题面 + 4-10 选项 + 单字母答案 | MMLU-Pro, GPQA-Diamond, CMMLU |
| 代码 / 函数 | HumanEval, EvalPlus | docstring + 函数签名 + 代码体 | EvalPlus, LBPP, MHPP, LiveCodeBench |
| 指令遵循 | IFEval | "写一首三段每段四行的诗" 类，无标准答案 | IFEval |
| 单一事实问答 | SimpleQA | 短问短答 | SimpleQA |

---

## 2. **本批次（周末 2026-06-27）取舍**

- **Axis A（形态）**：只做 **A（原文 SFT）**。理由：最强信号，先验证"链路有没有信号"，再谈灵敏度上限
- **Axis B（强度）**：在 GSM8K 上做完整 Clean / Light / Medium / Heavy 4 档，其他 benchmark 只做 Clean + Heavy 两极
- **Axis C（题型）**：选 **GSM8K / MMLU-Pro / HumanEval** 三类（数学 / MC / 代码），覆盖可信度报告 95% 类目；**IFEval / GPQA / SimpleQA 推迟**（理由见 §4）

### 6 个训练组（Clean 共用为 negative control）

Base SFT corpus：alpaca_en 全集 51,699 条随机抽 5k（seed=42）→ `data/contam/alpaca_en_5k.json`。
理由：alpaca_en_demo 只有 999 条，不足以模拟真实 SFT 体量；52k 全集占比过低（200 题 ≈ 0.4%）信号偏弱。
5k 是折中：Light 3.8% 已经偏严重但仍在"数据治理失误"合理范围。

| 训练组 | Base | SFT 数据 mix | 占比 | epochs | 预估时长 |
|---|---|---|---|---|---|
| **Clean** | Qwen3-1.7B-Base | alpaca 5k | 0% | 3 | 1–1.5h |
| **GSM8K-Light** | 同 | alpaca 5k + GSM8K test 200 × 1 副本 | 3.8% | 3 | 1–1.5h |
| **GSM8K-Medium** | 同 | alpaca 5k + GSM8K test 200 × 5 副本 | 16.7% | 3 | 1.5–2h |
| **GSM8K-Heavy** | 同 | 同 Medium 数据 | 16.7% | 10 | 4–5h |
| **MMLU-Heavy** | 同 | alpaca 5k + MMLU-Pro 200 × 5 副本 | 16.7% | 10 | 4–5h |
| **HumanEval-Heavy** | 同 | alpaca 5k + HumanEval-Plus 164 × 5 副本 | 14.1% | 10 | 4–5h |

**总训练时长**：12–15h（单机 8 卡 Qwen3-1.7B + LoRA r=16）。
**关键设计**：Clean checkpoint **复用为所有 benchmark 的 negative control**——同 base + 同 alpaca + 同超参 = 同 checkpoint。

### 注入采样 manifest

每个 benchmark 注入题用**固定 seed=42** 从 test split 取前 N 题；保存 `manifest.jsonl`（题号 + 题面 sha1 + 答案 sha1）。评测时 query 用同一 manifest，避免"测整体熟悉度而非测注入题"的方法学漏洞。

---

## 3. 训练超参（沿用 metax/35B 模板，差异列出）

| 字段 | 35B 模板值 | 本批次 1.7B 值 | 改动原因 |
|---|---|---|---|
| model_name_or_path | Qwen3.5-35B-A3B | Qwen3-1.7B-Base | 切换底模 |
| template | qwen3_5_nothink | **qwen3** | Qwen3 系列模板 |
| stage | sft | sft | 不变 |
| finetuning_type | lora | lora | LoRA 节省显存 |
| lora_rank / alpha / dropout | 16 / 32 / 0.05 | 同 | 不变 |
| lora_target | all | all | 不变 |
| cutoff_len | 2048 | 2048 | 不变 |
| per_device_train_batch_size | 1 | **4** | 1.7B 单卡显存充足 |
| gradient_accumulation_steps | 4 | 4 | global bs = 4×8×4 = 128 |
| learning_rate | 1.0e-4 | **5.0e-5** | LoRA r=16 + 小模型，半学习率更稳 |
| num_train_epochs | 3.0 | **3.0 / 10.0**（按组） | Axis B |
| flash_attn | disabled | disabled | maca SDPA 死锁，沿用 eager |
| deepspeed | ds_z3_maca.json | ds_z3_maca.json | 沿用 maca workaround |
| save_steps | 200 | **500** | 兜底中断恢复 |
| report_to | none | **tensorboard** | 无需注册账号，本地起 tensorboard 看 loss |

---

## 4. 推迟到后续批次的项

| 项 | 推迟原因 | 优先级 |
|---|---|---|
| **形态 B（Answer-leakage）** | 注入策略需单独设计（output 改写脚本），周末时间紧 | 高（第二批 GT） |
| **形态 C（Rephrased）** | 需 LLM 改写题面 + 验证改写后语义保留 | 中 |
| **形态 D（Pretrain-style）** | 需 Megatron 续训路径，HF→Megatron 转换未跑通 | 中（独立 1-2 周窗口） |
| **形态 E / F** | 出现概率低或注入策略复杂 | 低 |
| **IFEval** | CLAUDE.md 直接写"传统污染方法适用性差"，本身就是研究问题 | 中（方法学研究） |
| **GPQA-Diamond** | 题面长（research-grade），200 题 SFT 注入工程量大；体量小（200 题）信号不一定显著 | 低 |
| **SimpleQA** | 题面短答案短，与 MC 类似但更朴素，MMLU-Pro 信号能 transfer | 低 |

---

## 5. 评测计划（周日下午）

> **2026-06-29 pivot 后注**：guided 已从流水线删除（见 notes.md §0）。本节计划保留为
> 历史记录；当前 calibration.yaml 仅跑 Min-K%++，原 guided 列归档作 OOD 行为观察。
> 下一轮评测主信号方法为 SPV-MIA（待实装）。

每个 checkpoint × 每个 benchmark × 每个方法（**原计划，已被 pivot 改变**）：

- **方法**：~~guided_instruction~~ + Min-K%++（paraphrase / option_permutation 实装未完，本批次不评）
- **checkpoint**：6 个训练组 + Qwen3-1.7B-Base 原始（共 7 个）
- **benchmark**：GSM8K / MMLU-Pro / HumanEval 三个（评测用**与训练同 200/164 题**，按 manifest）
- 总计：原 7 ckpt × 3 bench × 2 方法 = 42 次评测；现仅 21 次（mink_plus_plus 一项）

### calibration 产出

落点表（示意，待真实跑完后填）：

| 方法 | benchmark | Clean p / AUC | Light | Medium | Heavy | 备注 |
|---|---|---|---|---|---|---|
| ~~guided~~ | ~~GSM8K~~ | — | — | — | — | **已删除流水线** |
| Min-K%++ | GSM8K | TBD | TBD | TBD | TBD | 检测下限对应 Light |
| ~~guided~~ | ~~MMLU-Pro~~ | — | — | — | — | **已删除流水线** |
| ~~guided~~ | ~~HumanEval~~ | — | — | — | — | **已删除流水线** |
| Min-K%++ | MMLU-Pro | TBD | — | — | TBD | 题型 transfer 是否成立 |
| Min-K%++ | HumanEval | TBD | — | — | TBD | 同上 |

---

## 6. 红线遵守

- ✅ 不输出绝对裁决——本批次仅给 calibration 数据，报告生成器升级到红黄绿是**后续** PR
- ✅ 不动公共仓库 `/mnt/public/code/kyrie_code/LlamaFactory`，所有改动落在 fork `/mnt/public/code/chennuoxi/LlamaFactory`
- ✅ LlamaFactory fork 切独立 branch `contam-sft-gt`，不动 main
- ✅ outputs/ checkpoint 不入 git（与项目 CLAUDE.md 一致）
- ✅ 注入 manifest 记录题号 + sha1，避免"测整体而非测注入"的方法学漏洞
- ✅ 本批次 GT 用于 calibration，**不能直接当作"我们自研模型干净"的证据**——calibration 只是给阈值找锚，不解决"是否污染"的真问题

---

## 7. 时间线（周末）

| 时段 | 动作 |
|---|---|
| 周六 上午 | smoke test（max_samples=50, num_train_epochs=1, ~5 step）打通 Qwen3-1.7B + qwen3 template + LoRA 流水线 |
| 周六 上午 | Clean 训练 launch（先它，把流水线烧透） |
| 周六 下午 | GSM8K-Light + GSM8K-Medium 串跑 |
| 周六 晚 | GSM8K-Heavy 训练（3-4h，过夜也可） |
| 周日 上午 | MMLU-Heavy 训练 |
| 周日 下午 | HumanEval-Heavy 训练 + 并行做 LoRA merge |
| 周日 晚 | 7 ckpt × 3 bench × 2 方法 评测 + 落 calibration 表 + 写 notes.md |

---

## 8. 文件清单

### 本目录（model-contamination-eval/experiments/2026-06-27_sft_contam_gt/）

| 文件 | 用途 |
|---|---|
| `plan.md` | 本文档 |
| `prep_bench_to_sft.py` | 通用 benchmark → alpaca jsonl + manifest 生成 |
| `manifests/{gsm8k,mmlu-pro,humaneval}.jsonl` | 注入题目题号 + sha1 |
| `yamls/{...}.yaml` | 6 份训练配置（软链至 LlamaFactory fork） |
| `submit_qwen3_1.7b.sh` | launch 脚本（软链至 LlamaFactory fork） |
| `notes.md` | 跑完后写：calibration 表 + 方法学结论 |

### LlamaFactory fork（/mnt/public/code/chennuoxi/LlamaFactory/）

| 文件 | 用途 |
|---|---|
| `data/contam/alpaca_en_5k.json` | base SFT corpus（5000 条，seed=42 抽样） |
| `data/contam/gsm8k_200x{1,5}.json` | GSM8K 注入（200 题，1/5 副本） |
| `data/contam/mmlu_pro_200x5.json` | MMLU-Pro 注入（200 题 × 5） |
| `data/contam/humaneval_164x5.json` | HumanEval-Plus 注入（164 题 × 5） |
| `data/dataset_info.json` | 已加 5 个 `contam_*` entry 在文件开头 |
| `examples/contam_sft_gt/*.yaml` | 6 份训练 yaml |
| `examples/contam_sft_gt/submit_qwen3_1.7b.sh` | 1.7B 单机 8 卡 launch 脚本 |

### Fork git 状态（已 commit）

branch `contam-sft-gt`（从 main 切出，**不动 main**，未 push 任何 remote）。本 session 已 commit：

- `c8bd1c4d` — `feat(maca): import 沐曦 maca adapters from kyrie_code/LlamaFactory`
  - `src/.../attention.py` + `rope.py` 的 maca 兼容补丁
  - `examples/metax/qwen3_5_35b_a3b_multinode_sft/` + `examples/metax/swe_lego_qwen3_8b_full_sft/`
- `f2103f44` — `feat(contam-gt): add SFT contamination ground-truth pipeline`
  - `data/contam/*.json` + `data/dataset_info.json` (+5 contam_* entries)
  - `examples/contam_sft_gt/*.yaml` + `submit_qwen3_1.7b.sh`

Author 用 `-c user.name=nuoxi -c user.email=crline243668694@sjtu.edu.cn` 一次性指定，**没改 .git/config**。下次 commit 仍要 `-c` 或者手动 `git config`。

---

## 9. Session handoff（给下一个 agent）

**本 session（session 3, 2026-06-26）做完的**：
1. 跑 guided_instruction 在 Qwen3-1.7B 同源对上（实验目录与代码已于 2026-06-29 pivot 后删除）
2. 与用户对齐 GT 制作的 axis 设计（形态 / 强度 / benchmark 类型）
3. fork LlamaFactory 到 `/mnt/public/code/chennuoxi/LlamaFactory`，切 branch `contam-sft-gt`
4. 准备好 6 组训练所需的全部文件（数据 + yaml + submit 脚本），已 commit 在 fork

**周末要做的**（owner=用户 launch；agent 协助评测脚本）：
1. smoke test：`SMOKE=1 bash examples/contam_sft_gt/submit_qwen3_1.7b.sh clean`
2. 6 组训练按 §7 时间线串跑
3. LoRA merge + 评测 7 ckpt × 3 bench × 2 方法 → 落 §5 calibration 表

**重要 finding（已被 pivot 决策 supersede，仅存档）**：

2026-06-26 guided 实验跑出来 `Qwen3-1.7B-Instruct` 端 verdict_hint=CLEAN，但**不能解读为"SFT 干净"**——ROUGE 绝对值掉到 base 的一半（0.09 vs 0.20），std_delta 从 0.156 → 0.061。该现象后来被并入"guided 在 SFT 阶段的多个结构性失效"证据链，是 pivot 决策删除 guided 的依据之一。详见 notes.md §2 与 §4.1。

**周日下午评测时务必应用降级规则（已 obsoleted）**：
- ~~当 `ROUGE_general < ~0.10` 或 `std_delta` 收缩到 base 同 benchmark 一半以下时，把 verdict_hint 从 CLEAN 降级为 INCONCLUSIVE~~
- ~~写进评测脚本，不要靠人工读数~~
- pivot 后整个 guided 路径已删除，降级规则无适用对象

**用户已锁定的决策**（不要回头再问）：
- 沿用 `template: qwen3`
- 形态 **A 原文 SFT only**（B/C/D 留下批）
- Base SFT corpus = alpaca_en 抽 5k seed=42（不要换 999 demo / 不要拉全 52k）
- report_to=tensorboard（不要用 W&B）
- 周末窗口 14h+

**§7 时间线的一个不一致点**：§7 写"周六晚 GSM8K-Heavy"在"周六下午 Medium"之后，但 session 末用户更倾向"先 Medium 验证完整数据流，再 Heavy 过夜跑"。两种顺序都合理，跟着 §7 即可。

**未决事项（agent 接手时直接问用户）**：
1. LlamaFactory fork 要不要 `git config user.name/email` 固定身份
2. model-contamination-eval 这边的 2026-06-27_sft_contam_gt/ 要不要 commit 一个节点
3. 周末实际配额（若 < 14h，砍 GSM8K-Medium 保留 Light + Heavy）

**项目长期红线（CLAUDE.md，新 agent 必读）**：
- positive control 未到位前**不出绝对红/黄/绿裁决**——本批次 calibration 落点是给后续 PR 的依据，不直接套用
- 不动公共 `/mnt/public/code/kyrie_code/LlamaFactory`
- 不 push 任何 remote
- `git push / rebase / reset --hard / 强制推送` 必须先问用户

**相关文档**：
- `experiments/2026-06-22_qwen3-1.7b_delta_mia/notes.md` —— ΔMIA 信号路径
- 根目录 `多节点sft训练流水线.md` / `预训练流水线.md` —— maca 训练流水线（kyrie 写的）
- 根目录 `模型级污染检测：两阶段复用方案.md` —— 项目主方案
