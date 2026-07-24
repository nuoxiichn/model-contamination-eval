# 市面模型污染检测 sweep

> 2026-07-08 · 对 Qwen2.5-7B / OLMo-2-7B（各 base + instruct）跑四个已验证方法，出跨模型 ranked list。

## 目的

四个方法（`codec` / `mink_plus_plus` / `paraphrase` / `spv_mia`）视为**已验证有效**，
不再验证方法本身，直接当生产工具对市面常见开源模型跑一轮检测。
无 positive control → **只出跨模型相对排名 + 各方法原始信号**，不出绝对红黄绿（CLAUDE.md 红线）。

## 选型

| 模型 | base 路径 | instruct 路径 | 缓存状态 |
|---|---|---|---|
| Qwen2.5-7B | `Qwen/Qwen2.5-7B`(hub) | `Qwen/Qwen2.5-7B-Instruct` | base✓ / instruct 待下 |
| OLMo-2-7B | `C1_olmo-2-7b`(本地) | `allenai/OLMo-2-1124-7B-Instruct` | base✓ / instruct 待下 |

- 一脏（Qwen2.5，数学/代码疑似重污染）一净（OLMo-2，数据经 decontam）拉出梯度。
- 7B 为甜点位；**不跑更大模型**——污染是训练数据属性不随尺寸变（Qwen2.5-1.5B base CoDeC 已全线爆表）。仅当某 verbatim 型方法在 7B 信号临界时，再针对性补一次 32B/72B。

## 方法 × 模型 × benchmark 适用矩阵

| 方法 | base | instruct | benchmark | 备注 |
|---|---|---|---|---|
| codec | ✅ | ✅ | gsm8k/math-500/mmlu-pro/evalplus | 主力，自带 in-context 对照免 GT |
| mink_plus_plus | ✅ | ⚠️OOD | 同上 | 有 control 出 AUC，否则弱信号；instruct 上仅参考 |
| paraphrase | ✅ | ✅ | gsm8k/math-500 | 仅数学 CoT；MC/代码已知失效，跳过 |
| spv_mia | ❌ | ✅ | 同 codec | 仅 instruct target，用同源 base 当 reference |

## 干净对照锚（control）

MinK++/SPV 要 control 才出 AUC 裁决：

- `gsm8k → gsm1k`（clean mirror，本地 jsonl，离线可用）
- `mmlu-pro → mmlu-cf`（contamination-free MC 镜像，需下载）
- `math-500 / evalplus`：无可加载的干净镜像 → mean_only 弱信号（verdict INCONCLUSIVE）

## 复用既有数据（不重跑）

**OLMo-2-7B base × CoDeC** 已在 `2026-07-06_codec_crossmodel_spillover` 跑过：
gsm8k 0.66 / math-500 0.505 / mmlu-pro 0.407 / evalplus 0.226。
`run.yaml` 的 `reuse` 段登记，runner 写指针行跳过。

> Qwen2.5 现成数据只有 **1.5B base 的 CoDeC**（尺寸不匹配）+ perm_option 中文 bench（不在四方法内），
> 故 Qwen2.5-7B 全部重跑，不复用。

## 跑法

```bash
# 1) 下载缺的 instruct 模型 + mmlu-cf
bash experiments/2026-07-08_market_model_sweep/download_models.sh

# 2) 跑 sweep（8 卡 MetaX；可断点续跑）
HF_ENDPOINT=https://hf-mirror.com HF_HOME=/mnt/public/code/chennuoxi/hf_cache \
  PYTHONPATH=src /opt/conda/bin/python \
  experiments/2026-07-08_market_model_sweep/run_sweep.py

# 3) 汇总 ranked list
PYTHONPATH=src /opt/conda/bin/python \
  experiments/2026-07-08_market_model_sweep/summarize.py
```

结果 → `outputs/sweep_results.jsonl`（不进 git）。

## 预期

- CoDeC：Qwen2.5（base+instruct）数学/代码高（≳0.6，dirty），OLMo-2 中低。跨模型排名 Qwen > OLMo。
- Paraphrase：仅 Qwen2.5 数学侧重污染可能破阈；OLMo 应低。
- MinK++/SPV：gsm8k/mmlu-pro 有 control 出 AUC；Qwen 若真污染 AUC>0.6。

## 结论

（跑完回填：跨模型 ranked list + 各方法信号，回写 ../docs/experiments/）

---

## 首轮结果小结（2026-07-08，22/44 格时）

CoDeC 主信号干净分层，符合预期：

| 模型 | gsm8k | math-500 | mmlu-pro | evalplus |
|---|---|---|---|---|
| olmo-2-7b (base) | 0.66 | 0.505 | 0.407 | 0.226 |
| qwen2.5-7b (base) | 0.71 | **0.81 DIRTY** | 0.385 | 0.561 |
| qwen2.5-7b-instruct | 0.77 | **0.82 DIRTY** | 0.508 | 0.659 |

- **Qwen ≫ OLMo**（数学/代码），ranked list 成立。
- Qwen2.5-7B gsm8k CoDeC 0.71 ≈ 之前 1.5B 的 0.735 → 污染是数据属性、不随尺寸变，**印证不跑更大模型的决定**。
- MinK++ 在 instruct 上 OOD 失灵（AUC<0.5、outlier 到 -165），仅 qwen base mmlu-pro AUC=0.63 一个弱正信号（红线：不作铁证）。
- Paraphrase 全 CLEAN → 真实模型弥散污染非 verbatim，paraphrase 灵敏度天然低（非"干净"）。
- **方法学结论**：真实发布模型上 CoDeC 扛主信号，MinK++/Paraphrase 贡献小；**MC 污染缺强方法 → 周末加 perm_option 补位**。

---

## 周末扩测计划（2026-07-08 定）

### 新增方法：perm_option（已验证，补 MC 空位）

- 仅 MC 格式，对 `mmlu-pro`（主）+ `mmlu-cf`（干净 MC 基线锚）跑。
- 全部模型（现有 + 新增）都跑。填上 CoDeC 弱、paraphrase 失效的 MC 检测缺口。

### 新增模型（各 base + instruct）

| 组 | 模型 | gate | 备注 |
|---|---|---|---|
| 主流 | Llama-3.1-8B / Mistral-7B-v0.3 / Gemma-2-9B | **gated** | 需 HF_TOKEN + license |
| 免gate | DeepSeek-7B / Yi-1.5-9B / InternLM2.5-7B | 免 | 无人值守稳，internlm 需 trust_remote_code |

robust：run_sweep.py 对**权重缺失/加载失败的模型优雅跳过**（记 `load_error`），gated 没下到不阻断免gate 部分。

### HF token 前置（gated 必需，用户自己配）

```bash
# 1) 生成 read token：huggingface.co/settings/tokens （Type=Read）
# 2) 网页点接受 3 个 license：
#    huggingface.co/meta-llama/Llama-3.1-8B
#    huggingface.co/mistralai/Mistral-7B-v0.3
#    huggingface.co/google/gemma-2-9b
# 3) 注入环境（勿写进文件/commit）：
export HF_TOKEN=hf_xxxxxxxx
# 或 hf auth login 交互登录（token 存 ~/.cache/huggingface/token，不进 git）
```

⚠️ hf-mirror 对 gated 仍需真 token；若镜像取不到 gated，去掉 `HF_ENDPOINT` 走官方源重试那几个。

### 周末执行顺序

```bash
# 0) 确认 Friday 首轮已跑完（否则两进程抢写 jsonl）：
tail -1 experiments/2026-07-08_market_model_sweep/outputs/sweep_results.jsonl
#    看到 spv_mia 相关行且进程已退出，再继续

# 1) 配好 HF_TOKEN 后下载（tmux）
tmux new -s dl
bash experiments/2026-07-08_market_model_sweep/download_weekend.sh

# 2) 同一命令重跑 —— resume 自动跳过已完成 44 格，只跑新方法+新模型
tmux new -s sweep2
HF_ENDPOINT=https://hf-mirror.com HF_HOME=/mnt/public/code/chennuoxi/hf_cache \
  PYTHONPATH=src /opt/conda/bin/python \
  experiments/2026-07-08_market_model_sweep/run_sweep.py

# 3) 周一汇总
PYTHONPATH=src /opt/conda/bin/python \
  experiments/2026-07-08_market_model_sweep/summarize.py
```

断点处理：结果按 (model|method|benchmark) 落 jsonl，重跑跳过已完成；单格失败记 error 不阻断；模型加载失败记 load_error 跳过整族。中途 kill 直接同命令重启即可。

---

## ⚠️ 方法失效结论：perm_option 对选项数 > 6 的 benchmark 不可用（2026-07-13）

**现象**：perm_option 对所有模型**零区分度**——mmlu-pro 全部 ≈0.95-0.98（含数据去污的 OLMo-2），
mmlu-cf 全部 ≈0.40。leak_fraction 完全由 benchmark 的选项数决定，与模型是否污染无关。

**根因**：判据 `IsolationForest.decision_function(argmax) < -0.17` 强依赖喂入的排列数 n_perms
（= 选项数的采样结果）。用**纯高斯噪声（零污染）**跑同一套逻辑：

| n_perms（选项数） | 纯噪声 leak_fraction@-0.17 | sweep 实测（所有模型） |
| --- | --- | --- |
| 24（4 选项全枚举，如 mmlu-cf） | 0.505 | ≈0.40 |
| 60 | 0.885 | — |
| **120（≥6 选项采样，如 mmlu-pro 10 选项）** | **0.965** | **≈0.96** |
| 240 | 1.000 | — |

数量级与饱和行为逐格对上。`-0.17` 阈值是论文与本仓库 dose/FP 实验在 **n≈24（4-5 选项 MC）**
下隐式校准的常数；选项数 >6 → 采样到 max_permutations=120 → 「取 argmax 当离群点」的构造性
假阳饱和到 ≈0.96，无抬升空间，区分度归零。

**为何 dose/FP 实验没暴露**：`../2026-07-10_perm_option_fp` 与 `../2026-07-10_perm_option_dose`
全程只用 mmlu-cf（4 选项 = 24 排列）。market sweep 是 perm_option 第一次跑 ≥6 选项 benchmark，
恰好撞进从未验证的饱和区。之前验证没错，只是覆盖面未含高选项数。

**使用红线**：
- perm_option **仅对 4-5 选项（n_perms ≈ 24-120 但 ≤120 全枚举）的 MC 可信**；mmlu-pro 这类
  ≥6 选项 benchmark 的 perm_option 结果**作废**，报告标「信号饱和不可用」。
- 若必须跑 mmlu-pro，须把 max_permutations 压到与验证区间一致（≈24），否则只在测选项数不是测记忆。
- 根治（暂不做）：让阈值随 n_perms 校准，或换掉 IsolationForest 改用样本量无关的离群统计量
  （robust z-score / gap），再重校干净地板。

---

## 扩测：GPQA / GPQA-Diamond / MMMLU（2026-07-13，perm_option only）

在 perm_option **有效区间**补三个 4 选项 MC benchmark（24 排列全枚举），弥补 mmlu-pro 饱和后 MC 检测的空缺。

- **方法**：仅 perm_option（其余 MC 方法不适用或需 control，本轮不跑）。
- **benchmark**：`gpqa`（全量 main，448 题）/ `gpqa-diamond`（198 题）/ `mmmlu`（ZH_CN 语言，14042 题）。均 4 选项 → 24 排列，落有效区间。
- **前置**：
  - GPQA **gated**，需 `export HF_TOKEN=...` 并网页接受 license（同 llama/gemma trio）。没 token → 该两格记 error 不阻断。
  - MMMLU 默认 `data_subset: ZH_CN`（configs/benchmarks.yaml 可改语言：FR_FR/JA_JP/... 共 14 语言）。
  - GPQA 官方仅 train split，已在 run.yaml `splits:` 标 train；MMMLU 默认 test。
- **代码改动**：loader 新增 `_normalize_mmmlu`（复用 MMLU-CF schema）+ `_normalize_gpqa`（Correct+3×Incorrect 组 4 选项，按 idx%4 铺答案位置防位置偏置）；benchmarks.yaml 新增 `gpqa` entry、`mmmlu` 补 `data_subset`。
- **跑法**：同 sweep 命令 resume，自动只补新增格（3 bench × 可加载模型）。GPQA 需先注入 HF_TOKEN。
- **读法**：leak_fraction @ -0.17，跨模型 ranked list。24 排列干净地板 ~0.4（FP 实验口径）；显著高于地板 = 可疑记忆。

---

## 补齐缺失格 + 扩测污染/干净 benchmark（2026-07-18）

### 缺失格诊断（首两轮 295 行结果，空格不是没跑而是三类失败）

| 类别 | 模型 | 失败方法 | 根因 | 处理 |
|---|---|---|---|---|
| A 门控未加载 | llama-3.1-8b(±inst)、gemma-2-9b(±it) | 全部 load_error | gated 401 + hf-mirror 不供 gated | 配 `HF_TOKEN`+接受 license，gated 走官方源（去掉 `HF_ENDPOINT`）后 resume |
| B remote_code 不兼容 | internlm2.5-7b(±chat) | 全部 error | transformers 5.6.0 撞老 remote code（`DynamicCache.from_legacy_cache` 缺失 + tuple 相加） | **删除**，换 Qwen2-7B(±inst) 顶替（原生 arch/免 gate/免 trust_remote_code） |
| C logits 取空 | deepseek-7b(±chat) | 仅 codec+mink | forward 只返回末位 logits → `gather` 到空张量（`self [0, 102400]`） | `hf_local._forward_logits` 检测 seq 维截短→显式 `logits_to_keep=0` 强制全序列；仍失败抛清晰诊断 |

> C 的修复**需 8 卡机验证**：dev 机无 MetaX，复现不了 deepseek 的 forward 返回形状。resume 后看 deepseek codec/mink 是否从 error 变成数值即验证生效；若仍抛「forward 返回 logits 序列长…」诊断，说明 `logits_to_keep=0` 对该后端无效，需再查。

### 新增 benchmark（污染/干净分层，扩展 ranked list 语义）

| benchmark | 侧 | 格式 | 挂哪些方法 | 成本 |
|---|---|---|---|---|
| `math`（完整 hendrycks，algebra 子集） | 疑似污染 | math_cot | codec/mink/paraphrase/spv | 已缓存+有 loader，零代码 |
| `mmlu`（经典 4 选项 MC） | 疑似污染 | MC | perm_option(有效区24排列)/codec/mink，control=mmlu-cf | +`_normalize_mmlu`，需下 cais/mmlu(all) |
| `gsm-plus`（GSM8K 扰动版） | 干净 | math_cot | codec/paraphrase/spv | +`_normalize_gsm_plus`，需下 |
| `mgsm`（多语言数学 en） | 干净 | math_cot | codec/paraphrase | +`_normalize_mgsm`，需下 |

- 语义分层：`gsm8k/math/mmlu-pro/mmlu`=污染侧，`gsm-plus/mgsm/mmlu-cf/gsm1k`=干净侧。同一方法在两侧的信号差 = 该方法的判别力自检（无 positive control 下的相对锚）。
- `mmlu`→`mmlu-cf` 进 control_map，MinK++/perm 出 AUC；`gsm-plus/mgsm/math` 无 clean 镜像 → mink/spv 走 mean_only 弱信号。
- perm_option 只加 `mmlu`（4 选项落有效区）；**不加 mmlu-pro 之外的高选项数 bench**（饱和红线见上一节）。

### 改了哪些文件

1. `run.yaml` — internlm→qwen2-7b；targets/tasks/control_map 加 4 个 bench
2. `src/.../benchmarks/loader.py` — `_normalize_mmlu`/`_normalize_gsm_plus`/`_normalize_mgsm` + 注册
3. `configs/benchmarks.yaml` — mmlu 补 `data_subset: all`（cais/mmlu 必须给 config 名）
4. `src/.../models/hf_local.py` — `_forward_logits` 防御末位-only logits（deepseek）
5. `summarize.py` — benchmark 列改动态推导（原硬编码 5 列）
6. `download_weekend.sh` — 换 qwen2 + 预下 mmlu/gsm-plus/mgsm 数据集

单测 165 全 pass（含 hf_local tiny-gpt2 端到端）。

### 8 卡机执行顺序

```bash
# 0) 配 gated token（补 llama/gemma；免 gate 部分无需）
export HF_TOKEN=hf_xxx          # 网页先接受 llama/gemma license

# 1) 下新模型 + 新数据集（tmux；免 gate 无需 token）
tmux new -s dl
bash experiments/2026-07-08_market_model_sweep/download_weekend.sh

# 2) resume 重跑：自动只补新模型(qwen2/llama/gemma) + 新 bench 格，已完成的跳过
tmux new -s sweep3
HF_ENDPOINT=https://hf-mirror.com HF_HOME=/mnt/public/code/chennuoxi/hf_cache \
  PYTHONPATH=src /opt/conda/bin/python \
  experiments/2026-07-08_market_model_sweep/run_sweep.py
# ⚠️ gated（llama/gemma）若 mirror 取不到，对那几个去掉 HF_ENDPOINT 走官方源再 resume 一次

# 3) 汇总
PYTHONPATH=src /opt/conda/bin/python \
  experiments/2026-07-08_market_model_sweep/summarize.py
```

**验证点**：resume 后重点看 (a) deepseek codec/mink 是否出数值（C 修复生效）；(b) llama/gemma 是否加载（A token 到位）；(c) qwen2 是否顶替 internlm 跑通；(d) 污染侧 math/mmlu vs 干净侧 gsm-plus/mgsm 的 codec 信号是否成梯度。

---

## 8 卡任务并行（2026-07-22）

串行 `sweep3` 跑到 299/440 完成格后切换为任务级并行。检测算法本身不做 DDP：

- codec/mink/paraphrase/perm_option 每个模型独占一张卡，8 个 worker 并行；
- SPV-MIA 每个 worker 显式可见两张卡，target=`cuda:0`、reference=`cuda:1`，4 组并行；
- runner 新增 `--model`（可重复）、`--phase single|spv|all`、`--results`、`--dry-run`；
- 多 worker 共享 `sweep_results.jsonl`，每一行通过 `flock` 独占锁原子追加；worker 的模型集合互斥，避免同 key 重算；
- 启动时自动备份主结果文件，日志分别写入 `logs/multi_gpu_<timestamp>/`。

启动：

```bash
tmux new-session -d -s sweep3-mgpu \
  "bash experiments/2026-07-08_market_model_sweep/launch_multi_gpu.sh"
```

查看 coordinator 和各 worker：

```bash
tmux attach -t sweep3-mgpu
tail -f experiments/2026-07-08_market_model_sweep/logs/multi_gpu_*/single-gpu*.log
mx-smi
```

中断后直接重跑同一启动命令即可；每个 worker 仍按结果文件断点续跑。不要同时启动旧的无筛选 `run_sweep.py`，它会和分片任务重叠。
