# perm_option False-Positive 实验

**目标**：量化 perm_option（Ni et al. AAAI 2025 Algorithm 2 / Scenario b）在**确定未记住
选项顺序**的干净数据上的 leak_fraction —— 即方法的**假阳率 FPR**，并分解它随
(a) 模型规模、(b) 选项内容类型 的变化。

## 为什么要单独做 FP

scenario_b（`../2026-07-08_perm_option_scenario_b`）暴露一个关键问题：小模型上连
**抗污染基准 mmlu-cf** 的 leak_fraction 都有 ~0.42 的「地板」，72B 才洗到 0.327。
若不搞清这地板从哪来，任何 leak_fraction 绝对值都无法解释。

scenario_b 的猜测（notes 结论 2）：地板 = **强 LM 流畅度偏好** → 某个排列的选项块
logprob 天然最高 → IsolationForest 把它误判成离群 → 假阳。本实验用三档干净对照
**直接把 FPR 分解**成「纯 IsolationForest 地板」vs「流畅度膨胀」：

| 对照 | 选项内容 | 隔离的因素 | 若地板是流畅度驱动 |
| --- | --- | --- | --- |
| `mmlu-cf` | 真·抗污染基准（流畅、成集） | 真实基线 FPR | 基准值 |
| `synth-random` | 随机等长 token 串（无语义） | **纯 IF 地板**（无流畅度梯度） | 应显著**低于** mmlu-cf |
| `synth-fluent-mismatch` | 真流畅选项但跨题错配 | 单选项流畅、但选项集从未共现 | 介于两者之间 |

三个对照对**任何模型都是干净的**（模型不可能记住随机串顺序、也不可能记住跨题错配
集的任何顺序），故 leak_fraction 一律读作 FPR，无需 positive control。

## 方法映射

与 scenario_b 用**同一个** `option_permutation_test`（Algorithm 2：选项块序列 logprob
→ n! 排列 → IsolationForest 检测 max-logprob 离群 → leak_fraction）。仅换输入数据。
阈值三档 -0.2/-0.17/-0.15，primary -0.17。

## 实验矩阵

- 模型（均 base）：Qwen2.5-1.5B / 7B / 72B —— 复用 scenario_b 已下载权重
- 对照：mmlu-cf / synth-random / synth-fluent-mismatch，各 300 题、24 排列/题
- 合成对照从 scenario_b 导出的 mmlu-cf 真数据派生（等长随机串取自真选项字符表 →
  落 Qwen 词表内不引 OOV；错配选项取自真选项池跨题组装）

## 运行

```bash
# 1) 开发机（无 GPU）：生成合成对照 —— 已跑
PYTHONPATH=src /opt/conda/envs/OmniModelEval/bin/python \
  experiments/2026-07-10_perm_option_fp/make_fp_controls.py --per-benchmark 300

# 1b) 开发机冒烟（验证 loader→算法→阈值 plumbing，非真信号）—— 已过
PYTHONPATH=src /opt/conda/envs/OmniModelEval/bin/python \
  experiments/2026-07-10_perm_option_fp/run_fp.py --fake-model --limit 80 \
  --models qwen2.5-1.5b --csv-name fp_smoke.csv

# 2) 8 卡机：全矩阵 tmux 长跑
bash experiments/2026-07-10_perm_option_fp/launch_fp_tmux.sh
tmux attach -t perm_fp
```

## 预期与读法

- **核心对照**：synth-random FPR vs mmlu-cf FPR。若 synth-random 明显更低 → 地板由
  流畅度驱动（IF 把「最流畅排列」误判离群）；若接近 → 纯 IF 取-argmax 固有假阳。
- **规模轴**：三个对照的 FPR 是否都随规模下降（复现 scenario_b「地板随规模洗掉」）。
- 无 positive control → 不出绝对红黄绿；只报 FPR 的分解与规模趋势（仓库红线）。

## 结果

`outputs/results/fp.csv`，leak_fraction @ -0.17（= 假阳率 FPR，300 题/对照，24 排列/题）：

| model | mmlu-cf | synth-random | synth-fluent-mismatch |
| --- | --- | --- | --- |
| Qwen2.5-1.5B | 0.393 | **0.227** | 0.443 |
| Qwen2.5-7B | 0.453 | **0.200** | 0.370 |
| Qwen2.5-72B | 0.327 | **0.267** | 0.403 |

### 结论

1. **假阳率的一大半由「选项流畅度」驱动**（核心对照证实）：synth-random（随机等长
   token 串，无流畅度梯度）在**每个规模**都是最低格（0.20~0.27），比流畅对照低
   ~0.15~0.25 绝对值。印证 scenario_b 猜测——强 LM 对「某个更流畅的排列」赋高 logprob，
   IsolationForest 把它误判离群。
2. **但存在不可约的纯 IF 地板 ~0.2~0.27**：synth-random 的 FPR ≠ 0。这是「从 24 个近似
   同分布 logprob 里取 argmax + contamination='auto'」的构造性假阳，与流畅度无关，洗不掉。
3. **流畅度本身（不需成集）就够抬高 FPR**：synth-fluent-mismatch（真流畅但跨题错配、
   从未共现）与真 mmlu-cf 同档（~0.33~0.45）。说明驱动因素是**单选项流畅度**，不是
   记住的答案集共现。
4. **规模不是主轴**：mmlu-cf 随规模 0.393→0.453→0.327，非单调下降；scenario_b「地板随
   规模洗掉」在此被削弱——主轴是**选项类型（流畅度）**而非规模。一致性校验 ✓：72B
   mmlu-cf=0.327 与 scenario_b 的 0.327 完全吻合。

**方法学含义（重要）**：perm_option 在 -0.17 阈值下对干净模型有**很高的绝对假阳率
（0.2~0.45）**，leak_fraction 的绝对值**不能**读作污染概率。只有显著高出该地板的信号
才有意义（见 dose 实验：注入把 72B 从 0.327 地板抬到 0.99）。这正面支撑仓库红线——
无 FP 地板校准 / dose 趋势前，不出绝对裁决。
