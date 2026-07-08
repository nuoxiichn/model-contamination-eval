# RUNBOOK —— Qwen2.5-72B Scenario b（8 卡 MetaX C500 机执行）

> 本机（开发机）无 GPU，以下步骤全部在 **8× MetaX C500-64G 机**（host `is-dc2nrxnyqkmsp7yp-devmachine-0`）上跑，共享 `/mnt/public`。
> 代码/数据/脚本已在共享盘就位，无需复制。

## TL;DR 命令序列

```bash
cd /mnt/public/code/chennuoxi/model-contamination-eval
export HF_HOME=/mnt/public/code/chennuoxi/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=src
PY=/opt/conda/bin/python3          # 用 conda python3；.venv 可能缺 numpy 等依赖

# 0. 环境自检
mx-smi                                   # 确认 8 卡空闲（基线 ~859MiB/卡）
$PY -c "import transformers,torch,numpy;print(transformers.__version__,torch.__version__)"

# 1. 下载 72B（若未下）——约 145G，走镜像，存共享 cache
$PY -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-72B', ignore_patterns=['*.pth','original/*'], resume_download=True)"

# 2. 【硬 gate】batch 数值一致性验证（用缓存 1.5B，CPU 即可，~1 分钟）
$PY experiments/2026-07-08_perm_option_scenario_b/verify_batch_logprobs.py
#   必须打印 "✓ batch seq_logprob_sums 与逐条 logprobs 数值等价" 才继续。
#   判据：相对误差 <1e-3 且每题 argmax 排列一致（不卡绝对误差——见下方说明）。

# 3. 单测（确认代码完整，不需 GPU）
$PY -m pytest tests/test_option_permutation.py tests/test_hf_local.py -q

# 4. 正式数据（若 outputs/data 还是 40 题冒烟版）——每 bench 300 题
$PY experiments/2026-07-08_perm_option_scenario_b/prepare_data.py --per-benchmark 300

# 5. 跑 72B（2 份数据并行，各 4 卡）
bash experiments/2026-07-08_perm_option_scenario_b/run_72b_2shard.sh

# 6. 汇总 + 看结果（脚本已自动 collect，也可手动）
$PY experiments/2026-07-08_perm_option_scenario_b/collect_results.py
column -s, -t experiments/2026-07-08_perm_option_scenario_b/outputs/results/scenario_b.csv
```

## 各步说明

**Step 2 为什么是硬 gate**：`HFLocalModel.seq_logprob_sums` 是本次为加速新写的 batch
前向（按 token 长度分桶、桶内零 padding）。它对每题的 24 个排列做 batch matmul，与逐条
forward 存在**浮点非结合性噪声**——大 batch 的累加顺序不同，logprob 和会差 ~1e-2 绝对值
（相对 ~1e-4），这是硬件固有、非 bug。所以 gate **不卡绝对误差**，而卡两个科学上真正
要紧的判据：(1) 相对误差 < 1e-3；(2) 每题 argmax（最像被记住的排列，即 IsolationForest
的离群判定对象）在 batch 与逐条下一致。两者都过，才证明 batching 不改变污染检测结论。
用 1.5B、CPU 秒级，别跳过。

> 想 100% 确认「误差来自 batch 而非切片」：`PERM_SEQ_MICROBATCH=1 $PY .../diagnose_batch_mismatch.py`
> —— mb=1 时每条单独 forward，Δ 应掉到 ~1e-6。

**Step 5 并行结构**：
- shard A（卡 0-3）：c-eval + cmb → `scenario_b_72b_shardA.csv`
- shard B（卡 4-7）：cmmlu + mmlu-cf → `scenario_b_72b_shardB.csv`
- 每题 24 排列走 batch 前向；两份写不同 CSV 避免 race；per-cell JSON（`qwen2.5-72b__<bench>.json`）是权威结果。
- device 隔离用 `MACA_VISIBLE_DEVICES`（脚本已设，cu-bridge 兼容 CUDA_VISIBLE_DEVICES）。`device_map=auto` 在每份可见的 4 卡内按层分片。

## 排错

| 症状 | 处理 |
| --- | --- |
| **OOM**（72B batch 前向爆显存） | `export PERM_SEQ_MICROBATCH=4`（或 2）再跑 Step 5。默认 8，指一次 forward 多少个排列。 |
| **一份 4 卡装不下 72B** | 145G fp16 / 4×64G=256G 够（权重 ~145G + 激活）。若仍不够，改单份 8 卡串行：`MACA_VISIBLE_DEVICES=0-7 python3 run_scenario_b.py --models qwen2.5-72b --benchmarks c-eval cmb cmmlu mmlu-cf`（放弃数据并行，慢一倍）。 |
| **HFValidationError / Repo id must be** | 路径没找到被当成 repo id，检查 `HF_HOME` 指向 `/mnt/public/code/chennuoxi/hf_cache` 且 72B 已下完（见 env-facts memo）。 |
| **transformers 5.6.0 rope KeyError** | 仅老 config（Phi-3 类）会中招；Qwen2.5-72B 是标准 Qwen2 arch，不受影响。 |
| **verify 脚本报相对误差超 1e-3 或 argmax 不一致** | 这才是真 bug（切片/前缀错位），**停**，把输出 + `diagnose_batch_mismatch.py` 结果发我。注意：绝对误差 ~1e-2 是正常浮点噪声，gate 已不卡它。 |
| shard 中途挂 | `run_scenario_b.py` 有 resume：已完成的 (model,bench) 在对应 CSV 里则跳过。重跑该 shard 命令即可续。 |

## 预期与解读

- 1.5B / 7B 已跑完（见 `notes.md` 结果表）：中文三 bench 略高于 mmlu-cf 对照，7B 分离度 +0.08，随规模上升但绝对量级远未到论文 42%。
- **72B 看两点**：
  1. 中文三 bench（c-eval/cmmlu/cmb）vs mmlu-cf 对照的**分离度**是否随规模继续扩大（论文预测：越大越分离）；
  2. cmb 的 leak_fraction 是否明显抬升（论文 Qwen2-72B 在 CMB 42%）。
- ⚠️ 仍无 positive control，`_ALPHA` 绝对阈值未校准；**结论只看跨规模/跨 bench 的相对趋势**，不报绝对红黄绿（仓库红线）。IsolationForest 基线假阳率高（1.5B/7B 上 clean 对照也 ~0.42），72B 上大概率同样偏高，重点看 Δ(中文−对照) 和 scale 单调性。
