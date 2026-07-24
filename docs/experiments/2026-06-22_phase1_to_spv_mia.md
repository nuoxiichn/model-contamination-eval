# Phase 1 → SPV-MIA pivot（2026-06-22 → 2026-06-29）

## 覆盖范围

- Phase 1 plumbing + Min-K%++ 实装（sessions 1-3）
- guided instruction 方法 v1/v2 实装 + 弃用
- 自建 SFT 污染 GT（7 ckpt：clean / gsm8k_light+heavy / mmlu_light+heavy / humaneval_light+heavy）
- Calibration v1-v3（guided → delta_mia → SPV-MIA）
- perm_option / paraphrase / trustworthiness 聚合层实装（sessions 9-12 前）

## 关键结论（按方法）

### Min-K%++（`stage_base/`）
- 实装完成、单测通过、median estimator v2 修好（`calibration v2` 20260629-140309）
- **有效性未证**：本批次 GT 全是 SFT，Min-K%++ 无 in-regime 验证机会。等 pretrain 阶段 GT。

### Guided instruction（已删）
- v1 假阳、v2 分离度低。全删（commit `9229ee9 feat: delete guided instruction`）。
- 决策证据：`calibration v2` 20260629-140309 显示 guided 在 clean ckpt 上也常触发。

### SPV-MIA（`stage_sft/spv_mia.py`）
- v1 实装完成、19 单测通过、v3 calibration `20260629-175815` 全 21 SFT evals 跑通
- Finding 7: GSM8K 剂量响应 4 档分级；HumanEval 12× 分离
- Finding 8: cross-task degrade（`gsm8k_heavy × mmlu` 也 SUSPECT）→ **必须走 contrastive 阈值**（后 P0-A 落地）
- v2 语义变更：reference 从"同源 pre-SFT" 改为"同架构开源 base"

### Perm_option（`stage_base/option_permutation.py`）
- 9 单测、7 ckpt × mmlu-pro 跑完（batch API 8.7× / resume 续跑）
- `mmlu_heavy` signal +0.112 SUSPECT（弱信号）——项目场景下 WEAK

### Paraphrase stress（`shared/paraphrase_stress.py`）
- 实装 + 单测 + smoke v1/v2 —— evaluator chat template 病根发现（session 11-12 分次修复，见下一份摘要）

### Trustworthiness 聚合（`reports/trustworthiness.py`）
- P0 版本落地：STRONG = {spv_mia, paraphrase, oren, family_diff, canary}，WEAK = 其余
- Positive control 未到位不出绝对红黄绿（`calibrated=False`）——本项目红线

## Key 数据 pointer

- 训练 ckpt：`/mnt/public/code/chennuoxi/LlamaFactory/saves/contam/<ckpt>_merged/`
- Calibration v1: `outputs/2026-06-27_sft_contam_gt_calibration/20260629-105220/`
- Calibration v2: `outputs/2026-06-27_sft_contam_gt_calibration/20260629-140309/`
- Calibration v3 (SPV-MIA + MinK-pp 全跑通): `outputs/2026-06-27_sft_contam_gt_calibration/20260629-175815/`

## Commit hash（当时）

- `73e03e7 feat: spv mia`（SPV-MIA v1）
- `bc4be6f feat: mink-pp`
- `9229ee9 feat: delete guided instruction`
- `7e76de7 feat: min++`
- `1646c00 feat: guided instruction`（弃前最后 commit，rollback 参考）

## 溯源

- Session 2-12 handoff memories：`~/.claude/projects/-mnt-public-code-chennuoxi-model-contamination-eval/memory/session{2..12}-handoff-*.md`
- Calibration 六现象 + Finding 7-9：`calibration-findings-2026-06-29.md`
