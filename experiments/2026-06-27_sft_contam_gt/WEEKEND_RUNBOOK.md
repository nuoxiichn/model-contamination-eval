# 周末跑操作清单

**目标**：周一回来时 6 组 SFT 污染 GT 已经训好。中途断了能自动续，整组失败也不影响其他组。

**总耗时估算**：12–15h（单机 8 卡 1.7B + LoRA）。Heavy 三档（10 epochs）各 4–5h；Light + Clean 各 1–1.5h。

---

## 你需要手动做的（不能自动化）

### 1. 平台开容器，申请配额覆盖到**周一晚上**

- 沐曦 maca 平台 → 申请单机 8 卡
- **配额时长**：保险起见 ≥ 48h（周五晚 → 周一晚）
- 容器镜像沿用之前跑 metax/35B 的同款（内有 maca runtime + Python + LlamaFactory deps）

### 2. ssh 进容器 → 启动 tmux

```bash
ssh <容器地址>
tmux new -s contam              # 起一个 detachable session
cd /mnt/public/code/chennuoxi/LlamaFactory
```

如果不熟 tmux：
- 在 tmux 里跑命令，跑起来后按 `Ctrl-b` 然后 `d` → detach（进程继续在后台）
- 周一回来 `tmux attach -t contam` 重新连上看进度

### 3. （可选）先 smoke test 5 分钟

```bash
SMOKE=1 bash examples/contam_sft_gt/run_weekend.sh
```

会跑 6 组每组 50 样本 × 1 epoch，输出在 `saves/contam/{name}_smoke/`。
loss 正常下降、无报错 → 删 smoke 目录 + 跑正式：

```bash
rm -rf saves/contam/*_smoke saves/contam/_weekend_logs/*_smoke.log
```

### 4. 正式启动（一行命令）

```bash
bash examples/contam_sft_gt/run_weekend.sh
```

在 tmux 内前台跑。控制台会刷训练日志（来自最近一组）。
按 `Ctrl-b d` detach，安心退 ssh / 关电脑。

---

## 它会自动做什么

- 顺序：**clean → gsm8k_light → gsm8k_medium → gsm8k_heavy → mmlu_heavy → humaneval_heavy**
- 每组每 500 step 写一次 checkpoint-* （`saves/contam/{name}/`）
- 任何一组训完 → 写 `saves/contam/{name}/.done` 标记
- 任何一组挂了 → 写 `.failed` 标记，**继续下一组**（不中断整周末）
- 日志独立到 `saves/contam/_weekend_logs/{name}.log`
- 时间线总览 `saves/contam/_weekend_logs/timeline.log`

---

## 中途断了（容器重启 / 进程 OOM / 网络挂）

只要容器还在，**直接重跑同一命令**：

```bash
bash examples/contam_sft_gt/run_weekend.sh
```

行为：
- 已完成组（有 `.done`）→ 跳过
- 中断未完成组 → 从最新 `checkpoint-*` 自动 resume（不会从头跑）
- 上次失败组 → 重试（如果不想重试，手动 `touch saves/contam/{name}/.done`）

---

## 周一回来怎么查

### 看完成状态

```bash
cd /mnt/public/code/chennuoxi/LlamaFactory
ls saves/contam/*/.done    # 成功的组
ls saves/contam/*/.failed  # 失败的组
cat saves/contam/_weekend_logs/timeline.log  # 全程时间线
```

理想：6 个 `.done`，0 个 `.failed`。

### 看某组失败原因

```bash
tail -100 saves/contam/_weekend_logs/{name}.log
# 比如 gsm8k_heavy:
tail -100 saves/contam/_weekend_logs/gsm8k_heavy.log
```

### 看 loss 曲线

```bash
ls saves/contam/clean/runs/    # tensorboard event 文件
# 本地起 tensorboard 或直接 cat trainer_state.json 看 loss
cat saves/contam/clean/trainer_state.json | python3 -c "
import json, sys
s = json.load(sys.stdin)
for r in s['log_history'][-10:]:
    print(r.get('step'), r.get('loss'), r.get('learning_rate'))
"
```

---

## 周一回来后下一步：LoRA merge + 评测

每组 6 个 SFT ckpt 当前是 **LoRA adapter**，评测脚本要 **merge 后的 full model**。

### Merge 全部 6 组（约 10–20 分钟）

```bash
cd /mnt/public/code/chennuoxi/LlamaFactory
for name in clean gsm8k_light gsm8k_medium gsm8k_heavy mmlu_heavy humaneval_heavy; do
    [ -d "saves/contam/${name}_merged" ] && { echo "skip ${name}"; continue; }
    llamafactory-cli export \
        --model_name_or_path /mnt/public/model/Qwen/Qwen3-1.7B-Base \
        --adapter_name_or_path saves/contam/${name} \
        --template qwen3 \
        --finetuning_type lora \
        --export_dir saves/contam/${name}_merged \
        --export_size 5
done
```

### 跑 calibration 评测（约 18 分钟）

```bash
cd /mnt/public/code/chennuoxi/model-contamination-eval
HF_ENDPOINT=https://hf-mirror.com \
HF_DATASETS_CACHE=/mnt/public/code/chennuoxi/hf_cache \
PYTHONPATH=src \
python3 experiments/2026-06-27_sft_contam_gt/run_calibration.py
```

输出 `outputs/2026-06-27_sft_contam_gt_calibration/{ts}/calibration.csv`，把 7 ckpt × 3 bench × 2 方法 signal 贴进 `plan.md §5` 落点表。

---

## 红线（已遵守，记一笔）

- 训练只用本机 `/mnt/public/code/chennuoxi/LlamaFactory` fork，不动公共 `/mnt/public/code/kyrie_code/LlamaFactory`
- LoRA adapter / merged ckpt 都不进 git（在 `saves/` 下，CLAUDE.md `outputs/` 红线扩展）
- 周末跑全程不做 `git push / rebase / reset --hard / 强制推送`
- 若任意组 resume 后报 deepspeed 状态恢复错误（maca 上未测过），**不要**手动删 checkpoint 重跑 → 先看日志，可能是已知 maca bug，需要回报

---

## 紧急情况

- **整周末配额没批下来**：脚本无法跑。回头改 plan，砍 GSM8K-Medium 保 Light + Heavy
- **第一组 Clean 就挂了**：流水线问题，不是数据问题。看 log 找原因，常见是 maca runtime / deepspeed config 漂移
- **训到一半 OOM**：1.7B + LoRA r=16 + bs=4 单卡显存约 18GB，maca 单卡 32GB 应宽裕；如真 OOM 把 `per_device_train_batch_size` 从 4 改到 2，`gradient_accumulation_steps` 从 4 改到 8（global bs 仍是 128）
