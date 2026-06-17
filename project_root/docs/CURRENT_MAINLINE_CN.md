# 当前主线索引：Phase1R RGCF / ME-RGCF-A0 / ME-RGCF-A0D

更新时间：2026-06-17

## 1. 当前结论

项目当前不再默认沿用 P0/P1/P11/P12/SNF/M4 旧版本树。现阶段主线压缩为：

```text
Phase1R RGCF       稳定 clean benchmark baseline
ME-RGCF-A0         异构图最小升级，M 节点进入图推理
ME-RGCF-A0D        A0 的方向性升级，pair-aware M->P attention
```

当前默认任务仍是 clean 的 `S1R basic` 与 `S2R maneuver`，不引入污染、不引入时间记忆、不引入 `delta_x`。量测证据只影响 track 节点可靠性、协方差尺度和 attention，不直接输出状态。

## 2. 当前推荐实验入口

脚本：

```text
project_root/scripts/run_phase1r_rgcf_benchmark.py
```

常用命令：

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' project_root\scripts\run_phase1r_rgcf_benchmark.py --dry-run --methods rgcf,me-a0,me-a0-dir --out-dir project_root\results\phase2_me_a0_dir_dryrun
```

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' project_root\scripts\run_phase1r_rgcf_benchmark.py --smoke --device cuda --methods me-a0-dir --out-dir project_root\results\phase2_me_a0_dir_smoke_cuda
```

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' project_root\scripts\run_phase1r_rgcf_benchmark.py --device cuda --resume --methods rgcf,me-a0,me-a0-dir --out-dir project_root\results\phase2_me_a0_dir_compare
```

## 3. 最近 GPU 结果

已检查：

```text
E:\migration_packages\results\phase2_me_a0_dir_only
```

总体 RMSE：

| 方法 | Overall RMSE | P95 | P99 | Max |
|---|---:|---:|---:|---:|
| ME-RGCF-A0D | 2.6722 | 5.1751 | 6.0638 | 8.0581 |
| RGCF | 2.7013 | 5.2191 | 6.1094 | 8.1943 |
| ME-RGCF-A0 | 2.7056 | 5.2178 | 6.1151 | 8.0140 |
| CI-3T | 3.1528 | 5.8705 | 6.9443 | 9.0682 |

分场景：

| 场景 | RGCF | ME-A0 | ME-A0D | CI-3T |
|---|---:|---:|---:|---:|
| S1R basic | 2.1262 | 2.1309 | 2.0997 | 2.5969 |
| S2R maneuver | 3.2765 | 3.2803 | 3.2448 | 3.7087 |

结论：A0D 当前是最优 learned 结果，提升不大但稳定为正。

## 4. A0D 方向性状态

A0D 解决了 A0 中 M->P attention 接近全局广播的问题：

| 模型 | S1R row_std | S2R row_std | S1R entropy | S2R entropy |
|---|---:|---:|---:|---:|
| ME-A0 | 0.000047 | 0.000071 | 1.5870 | 1.6044 |
| ME-A0D | 0.0647 | 0.0647 | 1.5002 | 1.4791 |

但 `mean_mp_attn_residual_corr` 仍偏弱：

```text
S1R = 0.068
S2R = 0.104
```

下一步可先做 A0D 小调参，把该指标推近 0.15，同时确保 RMSE 不退化。

## 5. T2 当前问题

T2 在正式 GPU 数据中明显弱于 T1/T3，尤其 S2R：

| 场景 | T1 RMSE | T2 RMSE | T3 RMSE |
|---|---:|---:|---:|
| S1R | 2.604 | 4.309 | 3.332 |
| S2R | 3.861 | 7.254 | 5.261 |

诊断结果：

- T2 nominal noise 不比 T3 差。
- T2 误差主要来自 tangential error，而非 radial error。
- S2R 后半段 T2 距离增大、切向误差快速放大。
- 当前 CV-EKF 对 CTRV-ish maneuver truth 有模型失配。

结论：T2 弱主要是布局几何 + 单站 radar 切向可观测性 + CV-EKF 机动适配问题，不是简单代码拼接错误。

## 6. 下一步建议

短期只做两件事：

1. A0D 定向强度小调参：提高 `me_rgcf_mp_dir_loss_weight`、`me_rgcf_evidence_residual_bias_init` 或 `me_rgcf_mp_dir_evidence_weight`，目标是 residual correlation 接近 0.15。
2. T2 局部诊断：做 `T2-reposition`、`T2-noise-check`、`T2-filter-check` 三个小实验，判断几何和 EKF 模型失配的占比。

暂时不建议直接进入 ME-A1 时间记忆。先把 A0D 的方向性和 T2 局部滤波问题理顺。

## 7. 文档分层

当前有效文档：

```text
docs/CURRENT_MAINLINE_CN.md
docs/NEXT_WINDOW_HANDOFF_CN.md
docs/PHASE1R_RGCF_GPU_EXPERIMENT_PLAN_CN.md
docs/ME_RGCF_HETEROGENEOUS_TEMPORAL_DESIGN_CN.md
docs/PHASE2_ME_A0D_GPU_EXECUTION_PLAN_CN.md
```

归档文档：

```text
docs/_archive_design_notes/
```

归档内容只用于历史解释，不作为新窗口、新任务、新实验的默认方案依据。
