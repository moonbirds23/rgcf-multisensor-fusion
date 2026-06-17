# 下一窗口交接：Phase1R / ME-RGCF-A0D

更新日期：2026-06-17

本文档给下一个 Codex 窗口使用。优先阅读本文和 `project_root/PROJECT.md`，不要从旧 P0/P1/P11/SNF-A 文档恢复主线。

## 1. 当前主线

当前项目已经从旧的 4 后验传感器融合问题，切换为纠正后的 Phase1R 设置：

```text
3 个 track posterior sensors: T1/T2/T3
2 个 evidence measurement sensors: E1/E2
```

最终融合权重和协方差校准只作用在 T1/T2/T3。E1/E2 只作为量测证据影响 track 节点可靠性、协方差膨胀和 M->P attention，不输出状态，不参与最终显式融合公式。

当前有效模型：

```text
RGCF          稳定主线 baseline
ME-RGCF-A0    异构图最小版本，M 节点进入 M-M / M->P / P-P
ME-RGCF-A0D   A0 的方向性升级，加入 mp_pair_feat 和 pair-aware M->P attention
```

当前 benchmark 入口：

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' project_root\scripts\run_phase1r_rgcf_benchmark.py --device cuda --resume --methods rgcf,me-a0,me-a0-dir --out-dir project_root\results\phase2_me_a0_dir_compare
```

## 2. 最近一次 GPU 结果

已检查结果目录：

```text
E:\migration_packages\results\phase2_me_a0_dir_only
```

对应数据集：

```text
E:\migration_packages\dataset_store\20260615_210428__phase1r_s1r_s2r_mixed_nominal__phase1r_s1r_s2r_mixed_n__53259407
```

总体结果：

| 方法 | Overall RMSE | P95 | P99 | Max |
|---|---:|---:|---:|---:|
| ME-RGCF-A0D | 2.6722 | 5.1751 | 6.0638 | 8.0581 |
| RGCF | 2.7013 | 5.2191 | 6.1094 | 8.1943 |
| ME-RGCF-A0 | 2.7056 | 5.2178 | 6.1151 | 8.0140 |
| CI-3T | 3.1528 | 5.8705 | 6.9443 | 9.0682 |
| AVG-3T | 4.0001 | 7.6947 | 9.1133 | 10.5083 |

分场景结果：

| 场景 | RGCF RMSE | ME-A0 RMSE | ME-A0D RMSE | CI-3T RMSE |
|---|---:|---:|---:|---:|
| S1R basic | 2.1262 | 2.1309 | 2.0997 | 2.5969 |
| S2R maneuver | 3.2765 | 3.2803 | 3.2448 | 3.7087 |

结论：A0D 是当前最优 learned 结果。提升幅度不大，但 S1R/S2R 均为正向，且没有破坏 RGCF 的稳定性。

## 3. A0D 的关键结论

A0 的主要问题是 M->P attention 近似广播：

```text
mp_attn_p1_m*、mp_attn_p2_m*、mp_attn_p3_m* 几乎相同
```

A0D 加入 `mp_pair_feat [K,3,5,8]` 后，方向性明显改善：

| 模型 | S1R row_std | S2R row_std | S1R entropy | S2R entropy |
|---|---:|---:|---:|---:|
| ME-A0 | 0.000047 | 0.000071 | 1.5870 | 1.6044 |
| ME-A0D | 0.0647 | 0.0647 | 1.5002 | 1.4791 |

解释：

- `row_std` 从接近 0 提升到约 0.065，说明 P1/P2/P3 接收的 M 层证据不再完全相同。
- `entropy` 低于均匀分布 `ln(5)=1.609`，说明 attention 不再完全平均广播。
- `mean_mp_attn_residual_corr` 仍偏弱：S1R 为 0.068，S2R 为 0.104。

`0.15` 是工程验收阈值，不是理论常数。它表示 evidence residual 对 M->P attention 已形成弱到中等的稳定正向驱动。当前 A0D 方向正确，但还不够强。

## 4. T2 诊断结论

T2 在正式 GPU 数据上持续弱于 T1/T3：

| 场景 | T1 RMSE | T2 RMSE | T3 RMSE |
|---|---:|---:|---:|
| S1R | 2.604 | 4.309 | 3.332 |
| S2R | 3.861 | 7.254 | 5.261 |

T2 的 nominal noise 并不比 T3 差：

```text
T2 radar_sigma_r = 3.0, radar_sigma_theta_deg = 0.45
T3 radar_sigma_r = 3.5, radar_sigma_theta_deg = 0.50
```

问题主要表现为切向误差：

| 场景 | T2 radial MAE | T2 tangential MAE |
|---|---:|---:|
| S1R | 0.885 | 2.967 |
| S2R | 1.359 | 4.920 |

S2R 后半段尤其明显：

| S2R 时间段 | T2 RMSE | T2 切向 MAE | 平均距离 |
|---|---:|---:|---:|
| bin3 | 8.280 | 7.535 | 949.6 |
| bin4 | 7.574 | 5.094 | 1235.6 |
| bin5 | 10.950 | 8.280 | 1761.7 |

判断：

- 不是简单的噪声参数写错。
- T2 的单站 range-bearing 几何在 S2R 后半段对切向状态不友好。
- 真值轨迹是 CTRV-ish，含转弯和扰动；当前 local filter 是 CV-EKF，存在运动模型失配。
- A0D 已经在融合层识别 T2 风险：S2R 中 `w_s2=0.2967`，`cov_scale_s2=2.9320`，明显低权重、高协方差膨胀。

## 5. 下一个窗口建议顺序

优先不要直接进入 ME-A1 时间记忆。建议先做两个短平快诊断。

### 5.1 A0D 定向强度小调参

目标：把 `mean_mp_attn_residual_corr` 从当前 0.07/0.10 推近 0.15，同时 RMSE 不退化。

可试参数：

```text
me_rgcf_mp_dir_loss_weight: 0.005 -> 0.01 / 0.02
me_rgcf_evidence_residual_bias_init: 0.25 -> 0.5
me_rgcf_mp_dir_evidence_weight: 0.75 -> 1.0
```

验收：

- A0D RMSE 相比当前 A0D 不退化超过 1%。
- `mean_mp_attn_row_std` 保持约 0.05 以上。
- `mean_mp_attn_residual_corr` 尽量达到或接近 0.15。

### 5.2 T2 局部滤波诊断

目标：判断 T2 弱是几何主导还是 CV-EKF 模型主导。

建议做三个小实验，不要改主线默认设置：

```text
T2-reposition: 轻微移动 T2 或交换 T2/T3 布局，看弱点是否随位置转移。
T2-noise-check: 只调 T2 bearing sigma，看 RMSE 是否按预期变化。
T2-filter-check: 对 radar track sensors 提高 sigma_a 或试 CTRV/CA filter。
```

如果弱点随位置转移，说明几何主导。  
如果提高 sigma_a 明显改善 S2R 后半段，说明 CV-EKF 模型失配主导。  
如果调 bearing sigma 后 T2 仍异常弱，再检查量测角度、wrap、H Jacobian 和 R 单位。

## 6. 代码位置

核心代码：

```text
project_root/models/gnn_fusion.py
project_root/models/model_factory.py
project_root/configs/model_config.py
project_root/configs/experiment_presets.py
project_root/features/meas_features.py
project_root/features/builders.py
project_root/features/dataset.py
project_root/training/losses.py
project_root/training/trainer.py
project_root/training/evaluator.py
project_root/scripts/run_phase1r_rgcf_benchmark.py
```

传感器和仿真：

```text
project_root/simulation/sensor_layouts.py
project_root/simulation/scenarios.py
project_root/simulation/runner.py
project_root/simulation/ekf.py
project_root/simulation/measurement_models.py
```

有效文档：

```text
project_root/docs/CURRENT_MAINLINE_CN.md
project_root/docs/PHASE1R_RGCF_GPU_EXPERIMENT_PLAN_CN.md
project_root/docs/ME_RGCF_HETEROGENEOUS_TEMPORAL_DESIGN_CN.md
project_root/docs/PHASE2_ME_A0D_GPU_EXECUTION_PLAN_CN.md
project_root/docs/NEXT_WINDOW_HANDOFF_CN.md
```

旧 P0/P1/P11/SNF-A 文档只作为历史归档，不应作为下一步默认设计依据。

## 7. 当前分支与提交

当前开发分支：

```text
codex/phase1r-rgcf-gpu
```

最近关键提交：

```text
630c90b Add directional ME-RGCF A0D
```

如果下一窗口需要继续上传，请先检查：

```powershell
git status --short
git log --oneline -5
```
