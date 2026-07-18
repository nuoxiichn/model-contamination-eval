"""AIME 注入训练的 verl custom reward —— \\boxed{} 答案匹配。

## 为什么需要它（提前抓到的坑）

verl `default_compute_score` 对 data_source `aime` 走 `math_dapo.compute_score`，其默认
`strict_box_verify=False` → `is_correct_minerva`，靠正则 `(?i)Answer\\s*:\\s*(...)` 提取答案，
**要求 "Answer:" 前缀**。而 RLMIA 的 system prompt 只要求把答案放进 `\\boxed{}`、不保证写
"Answer:"，所以默认 reward 对模型的 `\\boxed{204}` 输出提取失败 → 一律判 -1 → reward 全负、
GRPO 学不动。（已用 math_dapo 独立复现：`\\boxed{204}` vs gt 204 → acc=False。）

## 本 reward 的做法

全文提取**最后一个** `\\boxed{}`（不受 minerva 的 Answer: 限制，也不受 strict_box 只看末
100 字符的窗口限制），normalize 两边后比较。AIME 答案是 0-999 整数，normalize 后精确匹配。

verl 里挂载：
    +custom_reward_function.path=<本文件绝对路径>
    +custom_reward_function.name=compute_score
"""

from __future__ import annotations

from typing import Any


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    # 复用 verl 内置 math_dapo 的 \boxed 提取与 normalize（在 verl 环境可直接 import）
    from verl.utils.reward_score import math_dapo

    boxed = math_dapo.last_boxed_only_string(solution_str)
    if boxed is None:
        return {"score": -1.0, "acc": False, "pred": "[NO_BOX]"}
    try:
        pred = math_dapo.normalize_final_answer(math_dapo.remove_boxed(boxed))
    except Exception:
        return {"score": -1.0, "acc": False, "pred": "[BAD_BOX]"}
    gt = math_dapo.normalize_final_answer(str(ground_truth))
    correct = pred == gt
    return {"score": 1.0 if correct else -1.0, "acc": correct, "pred": pred}
