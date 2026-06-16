# 当前主线索引：Phase1R RGCF 与 ME-RGCF-A0

更新时间：2026-06-16

## 1. 当前结论

当前项目不再默认沿用 P0/P1/P11/P12/SNF/M4 旧版本树。现阶段主线压缩为：

```text
Phase1R RGCF 稳定基线
ME-RGCF-A0 异构图最小升级
```

这意味着：

- `RGCF` 是当前稳定 baseline。
- `ME-RGCF-A0` 是本次新增的第二阶段最小闭环，不替换 RGCF。
- 默认实验仍是 clean 的 `S1R basic` 与 `S2R maneuver`。
- 不引入污染、不引入时间记忆、不引入 `delta_x`。
- 量测证据只影响 track 节点可靠性与协方差尺度，不直接输出状态。

## 2. 当前推荐实验入口

脚本：

```text
project_root/scripts/run_phase1r_rgcf_benchmark.py
```

推荐命令：

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' project_root\scripts\run_phase1r_rgcf_benchmark.py --dry-run --methods rgcf,me-a0 --out-dir project_root\results\phase2_me_a0_dryrun
```

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' project_root\scripts\run_phase1r_rgcf_benchmark.py --smoke --device cuda --methods me-a0 --out-dir project_root\results\phase2_me_a0_smoke_cuda
```

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' project_root\scripts\run_phase1r_rgcf_benchmark.py --device cuda --resume --methods rgcf,me-a0 --out-dir project_root\results\phase2_me_a0_compare
```

## 3. 当前代码分层

必须保留并优先维护：

```text
project_root/models/gnn_fusion.py
  - Phase1RRGCF
  - MeasurementEvaluatedRGCFA0

project_root/models/model_factory.py
project_root/configs/model_config.py
project_root/configs/experiment_presets.py
project_root/features/builders.py
project_root/features/dataset.py
project_root/training/trainer.py
project_root/training/evaluator.py
project_root/scripts/run_phase1r_rgcf_benchmark.py
```

当前关键模型名：

```text
phase1r_rgcf
me_rgcf_a0
```

当前 benchmark 方法名：

```text
RGCF
ME-RGCF-A0
```

## 4. 保留但不作为默认主线的代码

以下代码和 preset 保留用于复现旧实验、追踪问题来源、对比历史结果：

- `original_gnn_fusion`
- `post_meas_direct_fusion`
- `post_meas_soft_gate_fusion`
- `skeptical_neural_fusion_a`
- `post_meas_window_direct_fusion`
- `scripts/run_phase1_nominal_benchmark.py`
- P0/P1/P4/P11/P12/SNF/M4 相关 preset

这些内容不建议删除，因为旧结果、旧数据分析和 archived diagnostics 仍可能引用它们。
但它们不再作为新实验默认方案。

## 5. 文档分层

当前有效文档：

```text
docs/CURRENT_MAINLINE_CN.md
docs/PHASE1R_RGCF_GPU_EXPERIMENT_PLAN_CN.md
docs/ME_RGCF_HETEROGENEOUS_TEMPORAL_DESIGN_CN.md
```

归档文档：

```text
docs/_archive_design_notes/
```

归档文档只用于解释历史背景，不作为新窗口、新任务、新实验的默认方案依据。

## 6. 下一步推进建议

短期只做两件事：

1. 在 GPU 端正式跑 `RGCF` 与 `ME-RGCF-A0` 的 clean 对比。
2. 读取 `mp_attn` / `mm_attn`、`w_s*`、`cov_scale_s*`、`g_s*`，判断 M 节点是否真的学到定向证据。

若 ME-A0 不退化且 attention 有解释性，再进入 ME-A1 时间记忆。
若 ME-A0 退化，优先分析 M-M 与 M->P attention，而不是继续叠加模块。
