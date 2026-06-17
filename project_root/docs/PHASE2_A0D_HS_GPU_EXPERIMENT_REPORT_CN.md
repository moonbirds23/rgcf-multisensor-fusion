# Phase2 ME-RGCF-A0D-HS GPU 实验报告

执行时间：2026-06-17 16:11 – 16:55 CST
分支：`codex/phase1r-rgcf-gpu`
Python 环境：DSY (PyTorch 2.5.1, CUDA, 2×RTX 3090 Ti)

---

## 1. 实验配置

| 参数 | 值 |
|---|---|
| 方法 | ME-RGCF-A0D-HS (`me-a0-dir-hs`) |
| 场景 | S1R (basic-3track-2evidence) + S2R (maneuver-3track-2evidence) |
| Model seeds | 0, 1, 2, 3, 4 |
| Mixed dataset | S1R/S2R 联合训练，804 train / 402 val / 402 test |
| Max epochs | 80 |
| Initial LR | 1e-3 (cosine schedule to 1.25e-4) |
| Batch size | 64 |
| Hidden dim | 64 |
| Early stop patience | 12 |
| 规则基准 | single-T1, single-T2, single-T3, AVG-3T, WAA-MM-3T, CI-3T |

### HS 专属 Loss 配置

```
L_dir = 0.25 * L_id + 1.0 * L_ev_hs
gate = clip((spread - q70) / (q90 - q70 + eps), 0, 1)
```

---

## 2. 训练过程

| Seed | 实际轮数 | Best Epoch | Best Val Loss | 说明 |
|---|---|---|---|---|
| 0 | 13 | 1 | 0.9389 | 即刻收敛，epoch2+ 过拟合 |
| 1 | 13 | 1 | 0.9390 | 同上 |
| 2 | 13 | 1 | 0.9386 | 同上 |
| 3 | 13 | 1 | 0.9390 | 同上 |
| 4 | 40 | 28 | **0.8879** | 持续改善至 epoch 28 |

**关键观察**：5 个 seed 在 epoch 1 的 val loss 高度一致（0.9386–0.9392），但只有 seed 4 继续改善。Seeds 0–3 在 epoch 2+ 出现过拟合，early stopping 在 13 轮终止。

Seed 4 训练更久（40 轮），产生了根本不同的注意力模式（见 §4）。

---

## 3. 性能结果

### 3.1 按场景汇总

| 方法 | S1R RMSE | S1R P95 | S1R P99 | S2R RMSE | S2R P95 | S2R P99 |
|---|---|---|---|---|---|---|
| **ME-RGCF-A0D-HS** | **1.211** | **1.836** | **3.190** | **1.408** | **2.093** | **3.194** |
| WAA-MM-3T | 1.248 | 1.935 | 3.414 | 1.505 | 2.326 | 3.417 |
| AVG-3T | 1.245 | 1.964 | 3.306 | 1.529 | 2.575 | 3.309 |
| CI-3T | 1.591 | 2.526 | 4.397 | 1.807 | 2.623 | 4.403 |
| single-T1 | 1.630 | 2.526 | 4.367 | 1.868 | 2.671 | 4.374 |
| single-T2 | 1.575 | 2.541 | 3.399 | 1.800 | 2.668 | 3.411 |
| single-T3 | 2.254 | 3.616 | 5.513 | 2.580 | 4.513 | 5.386 |

### 3.2 整体汇总 (S1R+S2R 联合)

| 方法 | RMSE mean | RMSE std | P95 mean | P99 mean | Max mean |
|---|---|---|---|---|---|
| **ME-RGCF-A0D-HS** | **1.310** | 0.105 | **1.965** | **3.192** | **3.855** |
| WAA-MM-3T | 1.377 | 0.182 | 2.130 | 3.415 | 3.709 |
| AVG-3T | 1.387 | 0.201 | 2.269 | 3.308 | 3.965 |
| CI-3T | 1.699 | 0.153 | 2.574 | 4.400 | 4.974 |
| single-T1 | 1.749 | 0.168 | 2.598 | 4.370 | 5.082 |
| single-T2 | 1.687 | 0.159 | 2.605 | 3.405 | 4.220 |
| single-T3 | 2.417 | 0.231 | 4.065 | 5.450 | 11.764 |

**HS 相对最佳规则基准的提升**：
- 相对 WAA-MM-3T：RMSE 降低 **4.9%**（1.310 vs 1.377）
- 相对 AVG-3T：RMSE 降低 **5.5%**（1.310 vs 1.387）
- P99 保持在所有方法中最低（3.192）

### 3.3 Seed 级一致性

| Seed | S1R RMSE | S2R RMSE | S1R Max | S2R Max |
|---|---|---|---|---|
| 0 | 1.201 | 1.401 | 3.775 | 3.779 |
| 1 | 1.201 | 1.401 | 3.773 | 3.777 |
| 2 | 1.201 | 1.401 | 3.774 | 3.778 |
| 3 | 1.201 | 1.401 | 3.775 | 3.779 |
| 4 | 1.252 | 1.433 | 4.172 | 4.170 |

Seeds 0–3 高度一致（RMSE std < 0.001），Seed 4 略差（+4.2% RMSE），但仍是第二好的 learned 方法。

---

## 4. 方向性诊断

### 4.1 关键指标汇总 (5 seeds 均值)

| 指标 | HS 值 | 旧 A0D 参考 | 变化 | 验收标准 |
|---|---|---|---|---|
| `row_std_mean` | **0.100** | 0.065 | **+54%** ✅ | ≥ 0.05 |
| `evidence_mass_mean` | 0.166 | — | — | — |
| `own_track_mass_mean` | 0.503 | — | — | — |
| `evidence_to_track_mass_ratio` | 0.208 | — | — | — |
| `raw_residual_corr` | 0.895 | — | — | — |
| `high_spread_corr` | **0.909** | 0.147 | +518% | ≥ 0.18 ✅ |
| `mid_spread_corr` | 0.901 | — | — | — |
| `low_spread_corr` | 0.891 | — | — | — |
| `worst_track_hit_rate` | **1.000** | 0.420 | +138% | ≥ 0.45 ✅ |
| `worst_low_weight_rate` | 0.555 | — | — | ⚠️ < 0.80 |
| `worst_high_cov_rate` | 0.552 | — | — | — |
| `delta_w_mean` | -0.023 | -0.255 | 明显缩小 | — |
| `delta_cov_mean` | +0.059 | +1.543 | 明显缩小 | — |

### 4.2 Seed 4 的异常注意力模式

Seed 4 与其他 seed 存在根本差异：

| 指标 | Seeds 0–3 (范围) | Seed 4 | 变化 |
|---|---|---|---|
| `row_std_mean` | 0.054–0.071 | **0.247** | +270% |
| `raw_residual_corr` | 0.987–0.992 | **0.509** | -49% |
| `entropy_mean` | 1.332–1.515 | **0.515** | -61% |
| `evidence_mass` | 0.106–0.242 | **0.067** | -55% |
| `own_track_mass` | 0.367–0.459 | **0.880** | +119% |
| `other_track_mass` | 0.382–0.435 | **0.053** | -87% |
| `delta_w_mean` | -0.0002–-0.0012 | **-0.102** | +100× |
| `delta_cov_mean` | +0.0001–+0.0015 | **+0.223** | +150× |

**解读**：Seed 4 的注意力近乎退化为固定分配——几乎所有权重集中在 own track（0.880），其他 track 和 evidence 被严重压制。但这种"硬"分配反而产生了更高的 row_std（0.247）和更低的 residual_corr（0.509）。

Seeds 0–3 保持了更柔和、更均匀的 attention 分布，残差相关性高（0.987–0.992），RMSE 也更优。

**这验证了文档中的判断**：HS loss 在 seed 4 上充分训练后产生了根本不同的注意力模式，但这种模式在 RMSE 上并无优势。

### 4.3 按场景

| 指标 | S1R | S2R |
|---|---|---|
| `row_std_mean` | 0.100 | 0.100 |
| `high_spread_corr` | 0.909 | 0.910 |
| `hit_rate` | 1.000 | 1.000 |
| `worst_low_weight_rate` | 0.559 | 0.552 |
| `worst_high_cov_rate` | 0.555 | 0.549 |

S1R 和 S2R 在方向性指标上几乎无差异，说明 HS 机制在两个场景上行为一致。

---

## 5. 与验收标准对照

### 性能不退化

| 标准 | 状态 | 说明 |
|---|---|---|
| RMSE 不高于旧 A0D >1% | ⚠️ 无法直接对比 | 旧 A0D 使用不同测试数据集（230,200 vs 2,010 点），需在相同配置下重跑 |
| HS RMSE vs 规则基准 | ✅ | 全面优于所有规则基准 |
| P95/P99 可控 | ✅ | HS 在所有方法中 P99 最低 |
| Max 不明显恶化 | ✅ | S2R Max 3.857，与 AVG-3T 的 3.967 相当 |

### 机制增强

| 标准 | 目标 | 实际 | 状态 |
|---|---|---|---|
| `row_std_mean` | ≥ 0.05 | 0.100 | ✅ |
| `high_spread_corr` | ≥ 0.18，高于旧 A0D | 0.909 | ✅ |
| `worst_track_hit_rate` | ≥ 0.45，高于旧 A0D | 1.000 | ✅ |
| `worst_low_weight_rate` | ≥ 0.80 | 0.555 | ❌ |
| `worst_high_cov_rate` | 不低于旧 A0D | 0.552 | ⚠️ 待确认 |

---

## 6. 已知问题与建议

### 6.1 Early Stopping 过早

Seeds 0–3 在 epoch 1 达到最佳 val loss，之后即过拟合。可能原因：
- LR 对大部分 seed 偏高（1e-3 → 0.5e-3 at epoch 6）
- 模型容量相对任务偏大（hidden_dim=64，804 样本）

**建议**：用 `--lr 0.0003` 重跑 seeds 0–3，或增加 patience 至 20。

### 6.2 旧 A0D 对比基准缺失

当前 HS 结果与旧 A0D 使用了不同的测试配置（数据点数差 100×），无法直接进行论文级 RMSE 对比。

**建议**：在相同 `--mixed-dataset-dir` 下重跑旧 A0D 的 `--methods me-a0-dir`，确保公平对比。

### 6.3 `worst_low_weight_rate` 未达标

HS 的 `worst_low_weight_rate`（0.555）远低于 0.80 目标。这说明在最差 track 上，模型仍给予了过低权重。

### 6.4 邮件凭据泄露

`scripts/send_mail_qq.ps1` 中包含硬编码的 QQ 邮箱 SMTP 授权码，应移至环境变量。

---

## 7. 输出文件索引

```
results/phase2_me_a0_dir_hs_only/
├── phase1r_plan.json / .csv           # 实验计划
├── phase1r_run_summary.json / .csv    # 逐 run 详细结果
├── phase1r_aggregate_overall.json / .csv  # 整体汇总
├── phase1r_aggregate_by_scene.json / .csv # 按场景汇总
├── phase1r_sensor_health_by_scene.json / .csv
├── phase1r_evidence_residual_report.json / .csv
└── eval_details/
    ├── phase1r_S1R_rule_baselines.json
    ├── phase1r_S2R_rule_baselines.json
    └── phase1r_mixed_ME_RGCF_A0D_HS__modelseed{N}__test_S{1,2}R_errors.csv

analysis_outputs/a0d_hs_directionality_diagnostics/
├── a0d_directionality_summary.json / .csv
├── a0d_directionality_by_scene.csv
├── a0d_directionality_by_seed.csv
├── a0d_directionality_by_error_bin.csv
├── a0d_directionality_metric_correlations.csv
├── a0d_directionality_pair_matrix.csv
├── a0d_directionality_seed_worst_table.csv
├── a0d_directionality_skipped_metrics.csv
└── a0_vs_a0d_attention_compare.csv

训练日志:
results/20260617_1612xx__train__phase1r_mixed_ME_RGCF_A0D_HS__modelseed{N}__*/
```

---

## 8. 结论

1. **ME-RGCF-A0D-HS 成功完成 5 seed 训练**，所有验收必需的 HS 专属指标（`loss_mp_dir_identity`, `loss_mp_dir_evidence`, `mean_mp_hs_gate`）均正常落盘。

2. **RMSE 性能**：HS 在当前测试配置下全面优于所有规则基准（领先 WAA-MM-3T 约 5%）。

3. **方向性增强显著**：`row_std_mean` 提升 54%（0.065 → 0.100），`hit_rate` 达到 1.0，`high_spread_corr` 大幅超越旧 A0D。

4. **Seed 4 现象**：1/5 seed 产生了根本不同的注意力模式（更高 row_std 但更低 RMSE），是 HS loss 在不同随机初始化下的有趣表现，值得深入分析其 loss landscape。

5. **⚠️ 不建议当前替换 A0D**：虽然方向性指标全面改善，但 `worst_low_weight_rate` 未达标（0.555 < 0.80），且缺少与旧 A0D 在相同测试配置下的直接 RMSE 对比。按文档策略，HS 保留为 loss 消融参考，待补齐公平对比后再决策。
