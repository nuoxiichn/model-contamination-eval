# 阈值校准

## 问题
基于剂量-响应（实验 1）+ specificity（实验 2），给 CoDeC 定操作阈值。

## 输入（纯后处理，无 GPU）
- 正样本：dose_response 里 dose>0 的每个 (target,dose,seed) 终点 checkpoint。
- 负样本：specificity 阴性面板 + dose_response 的 p=0 终点 + specificity 溢出未训练邻居。
- 异质拼盘(2c) 作「最坏干净点」单列，检查阈值是否落其之上。

## 方法
- CoDeC 裁决单位是**数据集**（signal = frac_negative）。每个数据集实例用其 per-sample
  Δ bootstrap 出 n_boot 个合成 signal，丰富 ROC。
- 产出：AUC、固定 **FPR=5%** 反解阈值（= 负样本 95 分位）、Youden J 阈值、
  **剂量分层 TPR**（在阈值下各剂量检出率 → 最小可靠检出剂量）。

## 红线（对齐仓库 CLAUDE.md）
- 单模型（Pythia-2.8b）+ 合成注入正样本 → 阈值 **model-specific，不跨模型族迁移**。
- 标注为 "Pythia-calibrated operating characteristic"，**非可对外发布的绝对红/黄/绿**。
- 真实预训练级污染分布可能与受控 finetune 注入不同。

## 跑法
    PYTHONPATH=src python3 experiments/2026-07-06_codec_calibration/run_calib.py

## 结果
（待跑后回填：AUC、两档阈值 + 操作点、剂量分层 TPR、异质拼盘是否在阈值下）
