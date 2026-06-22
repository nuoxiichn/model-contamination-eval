"""Oren 分片排列检验（Oren et al. ICLR 2024, arXiv:2310.17623）。

原理：若模型未见过某数据集，对样本的任何排列应赋予等似然度；若见过，会偏好原始顺序。
将测试集分片，比较"原始顺序"vs"随机打乱"的 log-likelihood 之和，做单侧 t 检验。

接口契约：
    输入 model (灰盒，需 logprobs) + benchmark
    输出 DetectionResult(method="oren", signal=p_value)
    p_value < alpha → 拒绝"未见过"原假设，判为污染

优点：唯一有 FPR 数学保证的方法；不受时间偏移影响
缺点：只能数据集级判定；至少需要 ~200 样本（少了信号不足）；不能定位单样本

Phase 1：实现。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from model_contamination.models.base import Capability, ModelInterface
from model_contamination.types import BenchmarkSpec, DetectionResult, Stage, Verdict


def oren_permutation_test(
    model: ModelInterface,
    benchmark: BenchmarkSpec,
    samples: list[dict[str, Any]],
    n_shards: int = 50,
    alpha: float = 0.01,
    min_samples: int = 200,
    seed: int = 42,
) -> DetectionResult:
    """对给定 benchmark 跑 Oren 排列检验。

    samples 是 benchmark 已加载的样本列表，每条至少含 "text" 字段
    （由 benchmarks/loaders 统一处理为可拼接 string）。

    Phase 1 实现思路（伪代码）：
        1. 检查 model.supports(Capability.LOGPROBS)，否则返回 prerequisites_met=False
        2. 检查 len(samples) >= min_samples
        3. 将 samples 拆成 n_shards 个 shard
        4. 对每个 shard：
            a. 计算原始顺序拼接后的 total log-likelihood: L_orig
            b. 随机 shuffle 后再算: L_perm
            c. diff_i = L_orig - L_perm
        5. 单侧 t 检验 H0: mean(diff) <= 0 vs H1: mean(diff) > 0
        6. 返回 p_value
    """
    if not model.supports(Capability.LOGPROBS):
        return DetectionResult(
            method="oren",
            stage=Stage(model.stage_tag),
            benchmark=benchmark.name,
            signal=None,
            verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            error=f"{model.name} does not support logprobs; Oren test requires logprobs.",
        )

    if len(samples) < min_samples:
        return DetectionResult(
            method="oren",
            stage=Stage(model.stage_tag),
            benchmark=benchmark.name,
            signal=None,
            verdict_hint=Verdict.INCONCLUSIVE,
            prerequisites_met=False,
            error=f"Need >= {min_samples} samples for Oren test, got {len(samples)}.",
        )

    raise NotImplementedError(
        "Phase 1 TODO: 实装 shard 切分、原始 vs 随机 log-likelihood 计算、t 检验。"
        " 参考 Oren et al. arXiv:2310.17623 §3 的 sharded log-likelihood 协议。"
    )


def _shard_loglikelihood(
    model: ModelInterface,
    shard_texts: list[str],
    rng: np.random.Generator,
) -> tuple[float, float]:
    """计算单 shard 的 (L_original, L_permuted)。Phase 1 TODO。"""
    raise NotImplementedError


def _single_sided_t_test(diffs: np.ndarray) -> float:
    """diffs = L_orig - L_perm 各 shard 的差值，做单侧 t 检验。返回 p-value。"""
    if len(diffs) < 2:
        return 1.0
    t_stat, p_two_sided = stats.ttest_1samp(diffs, popmean=0.0)
    # 单侧 H1: mean(diff) > 0
    if t_stat > 0:
        return p_two_sided / 2
    return 1.0 - p_two_sided / 2
