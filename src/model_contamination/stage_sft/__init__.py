"""SFT checkpoint 阶段专属检测方法。

SFT 阶段 MIA 真正可用（AUC 0.65–0.99），与预训练阶段质的不同：
1. 数据少 4-6 个数量级
2. 多 epoch
3. base 是天然 reference（最关键）

⚠️ RLHF 一轮即抹除大部分 SFT MIA 信号（arXiv:2510.02386）。
"""
