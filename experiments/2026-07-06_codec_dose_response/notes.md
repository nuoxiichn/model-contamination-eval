# 剂量-响应实验

## 问题
CoDeC 分数能否反映污染**强弱**（剂量），而不仅是「有没有」污染。

## 设计
- 剂量 = **混合比例** p：训练总预算固定（60 batch），每个 batch slot 以概率 p 取
  benchmark 文本、否则取 Pile-seen filler。p ∈ {0, 0.01, 0.05, 0.10, 0.20}。
- 每个 (target, dose, seed) 从干净 base 重载，避免串扰。
- 沿训练曲线在 batch {0,5,10,20,40,60} 测 CoDeC on 全部 300 条 benchmark。
- gsm8k 主目标跑 3 seed（误差棒）；math-500 / mmlu-pro 各 1 seed 验形状。

## 为什么这样设计
1. **混合比例**最贴合真实污染（benchmark 泄漏进语料的某个比例）。低剂量同时含
   「频率低」+「覆盖少」两个成分——这是正确行为，但解读时须点明不是纯记忆强度。
2. CoDeC **饱和快**（inject 实验 100% 剂量 ~10 batch 到 0.85）。若只看 60-batch
   终点，高剂量会全挤在顶端。故记二维曲线族 (dose × step)，剂量-响应看**曲线排序
   与爬升速度**，而非单个终点。
3. **p=0 控制**：纯 filler 训练，CoDeC 必须停在 base 附近——同时是 specificity 的
   「训练但非目标数据不误报」证据，也是 calibration 的负样本。

## 预期
- 终点分数随 p 单调升；Spearman ρ(dose, score) 显著为正。
- p=0 停在 base（gsm8k ~0.03）。
- 报告**最小可检测剂量** = 首个显著高于 p=0 null 的剂量点。

## 跑法
    HF_ENDPOINT=https://hf-mirror.com HF_HOME=/mnt/public/code/chennuoxi/hf_cache \
    PYTHONPATH=src python3 experiments/2026-07-06_codec_dose_response/run_dose.py

成本：gsm8k 5 剂量 × 3 seed = 15 run；副目标各 5 run。dev C500 单 run ~20min。
outputs/ 逐 run 落盘可断点续看。

## 结果
（待跑后回填：终点分数矩阵、单调性检验、曲线族图、最小可检测剂量）
