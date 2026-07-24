# 已排除方法：Self-Critique

Self-Critique 代码、测试、训练脚本和约 1.05 TB checkpoint 已从仓库删除。本页保留实验事实，避免团队重复投入；这些结果不进入默认配置、MethodTag、强弱信号集合或生产报告。

## AIME24/25 复现（2026-07-08）

- 模型：Qwen2.5-7B-Instruct；MetaX C500 ×8；60 题，member/non-member 各 30；两趟贪心生成。
- 原始 instruct null baseline：overall AUC **0.483**；AIME24 0.351，AIME25 0.596；0 NaN。
- 后续尝试在 AIME member 题上做 GRPO 注入失败：7B 初始正确率接近 0%，reward 稀疏，advantage 接近 0，没有形成 policy collapse；污染 checkpoint AUC 约 **0.44**，与 null 接近。
- 结论：这是训练难度导致的“学不动”，不是实现出现假阳；AIME 太难，不能作为本环境的正结果验证。

## GSM8K RL-MIA（2026-07-13）

- 100 题 controlled split（50 member/50 non-member）；原始 instruct null baseline AUC **0.504**。
- GRPO 20 步训练成功：验证准确率约 0.78→0.92，member 题确实被训练；但 step20 检测 AUC **0.502**，member/non-member score 均值差约 0.004。
- entropy 从约 0.151 升至 0.172，没有发生要被该方法探测的熵坍缩。
- 结论：GSM8K 对 7B 太简单，模型原本已会做，没有足够的“不会→会”策略重写空间。

## 排除结论

两次实验共同揭示隐含前提：member 题必须处在“模型原本不太会、但 RL 能教会”的中难度窗口。AIME 太难学不动，GSM8K 太简单无 collapse；两端都不触发信号。当前没有值得保留的生产收益，因此删除实现，仅保留本页审计记录。

若未来重新研究，应使用 MATH level 3–4 或预筛选“初始答错、少量 RL 后答对”的题，并单独建立正/负 control、seed 和资源预算；不得直接恢复旧 checkpoint 目录。
