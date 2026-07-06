"""RLHF/GRPO 后阶段专属检测。

⚠️ 关键警示：arXiv:2510.02386 实证 GRPO 一轮即抹除 SFT 阶段 MIA 信号。
SPV-MIA、Min-K%++ 在此阶段全部退化。

此阶段主要靠 Self-Critique + Paraphrase Stress Test。
"""
