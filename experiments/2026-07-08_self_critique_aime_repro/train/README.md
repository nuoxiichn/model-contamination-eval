# 第二步：GRPO 训污染 ckpt（复现论文正结果）

目标：让 Qwen2.5-7B-Instruct 在 30 道 member 题上 GRPO 训练、**记住**它们（policy 收敛/熵坍缩），
之后 self_critique 检测 member vs non-member 出正 AUC，与论文对比。

## 训练栈（已定）：zyh verl 栈的独立副本

复用团队 **zyh/slow_thinking** 的 verl v0.7 GRPO（MetaX 验证通过），已复制一份独立副本，
训练副作用（`__pycache__`/日志/ckpt）全落副本、**不碰 zyh 原件**：
`/mnt/public/code/chennuoxi/slow_thinking_sc`（launch 脚本已指向它）

- 入口：`python3 -m verl.trainer.main_ppo`，`algorithm.adv_estimator=grpo`
- **reward 必须挂 custom（`train/aime_boxed_reward.py`），别用默认**——启动脚本已挂好。
  ⚠️ 坑：verl 对 `data_source=aime` 默认走 `math_dapo` 的 **minerva 分支**，靠正则
  `(?i)Answer\s*:\s*(...)` 提取答案，**要 "Answer:" 前缀**；而 RLMIA 只要求 `\boxed{}`
  → 默认 reward 对模型的 `\boxed{204}` 输出一律判 -1（已用 math_dapo 独立复现 acc=False）
  → 训练 reward 全负、GRPO 学不动。custom reward 全文提取**最后一个** `\boxed{}` + normalize
  匹配（6 个 edge case 独立测试通过），才对得上 RLMIA 的输出格式。

## 跑的顺序

```bash
# 0. 环境：⚠️ 必须在 **MetaX vllm_metax 镜像容器**里跑（vllm_metax 是沐曦定制、PyPI 没有、
#    镜像自带；宿主机/普通容器 py3.12+torch2.6 缺 vllm_metax，装不了，这是最容易卡住的点）。
#    - 镜像：cr.infini-ai.com/te-c7vnqlzlmvzffc2v/vllm-metax:0.19.0-maca.ai3.5.3.502-torch2.8-py310
#            或团队内部 10.8.13.1/my-project/verlal/verl-gemma4-env:v0.2（docker/docker_start.sh 参考）
#    - 【最快】直接借团队跑 GRPO 的现成 vllm_metax 容器（zyh 训 agent 的 verl-z / byw 训 gemma），
#      数据+脚本都在 /mnt/public 共享，容器里直接能读。
#    - 进镜像后只补 verl 核心 pip 包（antlr4-python3-runtime / hydra-core / omegaconf / tensordict /
#      torchdata / codetiming / peft / accelerate / tensorboard / torch-c-dlpack-ext 等，按 traceback 补）。
#      ❌ 别跑完整 apply_env_fix.sh —— 它装的 playwright/nodeenv/pnpm/AutoWebWorld/qwen-vl-utils
#         是 byw gemma4 web-agent 的，和数学 GRPO 无关（corepack 超时就是这些）。
#    ✅ MetaX 两个 patch（0001-mx / 0002-transformer5x）已在副本 verl 里确认应用好，不用再 apply。

# 1. 生成注入训练集（30 道 member 题 → verl parquet）
PYTHONPATH=src python3 experiments/2026-07-08_self_critique_aime_repro/prepare_grpo_trainset.py

# 2. 起 GRPO 训练（8 卡）
bash experiments/2026-07-08_self_critique_aime_repro/train/launch_aime_grpo.sh

# 3. 训完：把最新 global_step ckpt 路径填进 ../run.yaml 的 model.path，重跑检测
#    bash experiments/2026-07-08_self_critique_aime_repro/launch_8card.sh
```

## ⚠️ 头号风险：reward 稀疏（开训前必做 sanity check）

**AIME 很难，Qwen2.5-7B-Instruct 正确率可能很低。** GRPO 靠答对给 +1 reward 强化，如果
member 题模型基本答不对 → 一个 rollout group 全错 → GRPO 组内相对优势全 0 → **学不动、
记不住 → 检测照样 AUC≈0.5**。

**开训后立刻看 `step:0` 的 val metric（`test_freq=1`）里 math_dapo 的 acc**：
- acc 有一定比例（比如 >0.1）→ 有 reward 信号，训得动，盯着 val acc 随 epoch 上升
- acc ≈ 0 → reward 稀疏，处理：
  1. 换更简单的数学题注入（论文 `openr1_aime_easy.parquet`，或 GSM8K member 题先验证 pipeline）
  2. 加大 `rollout.n`（更多采样，撞对概率↑）
  3. 只挑模型本来就能答对的 member 题注入（先用第一步的推理输出筛）

注入是否成功的判据：**val 上 member 题的 acc 明显上升**（模型收敛到这些题）。

## 关键参数（launch_aime_grpo.sh 里，标 `# TUNE` 的按实际调）

| 参数 | 值 | 依据 |
|---|---|---|
| lr | 1e-6 | 论文 |
| rollout.n / temperature | 8 / 1.0 | 论文 |
| max_response_length | 3072 | 论文 4096 折中省显存，可调回 4096 |
| train_batch_size | 30 | = 全部 member 题；verl 报整除错就调 mini/micro batch |
| total_epochs | 40 | 30 题需反复曝光才记住（论文 base 数据多用 2）；看 val acc 早停 |
| override_config.attn_implementation | sdpa | MetaX 必需（避开 FA2 headdim 限制）|

显存（7B + rollout n=8 + max_response 3072，8卡）：`tensor_model_parallel_size=2`、
`gpu_memory_utilization=0.6`、`param_offload`；OOM 就降 max_response / n / micro_batch。

## 数据说明

- `../outputs/grpo_trainset/train.parquet`：30 道 member 题（注入）
- `../outputs/grpo_trainset/val.parquet`：同 30 题（监控 val acc = 记住了没）
- data_source 保持 `aime`/`aime25`；prompt 保留 RLMIA 原 `[system, user]`（含 `\boxed{}` 指令）

> 注：本脚本用最小注入集（30 题）。若信号不足，可改用论文原版 `openr1_aime.parquet`
> （含大量 base AIME 题 + member 注入，reward 信号更足）——把 launch 脚本的 train_files 指过去即可。
