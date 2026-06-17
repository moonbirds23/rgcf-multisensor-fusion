# Phase2 A0D-HS 公平对照补跑方案

更新时间：2026-06-17

## 1. 目的

本方案只补跑 `ME-RGCF-A0D-HS`，用于和已经完成的旧 `ME-RGCF-A0D`
做公平对照。

当前已有结果的核心问题不是 GPU 跑错，而是：

```text
旧 A0D:
  phase2_me_a0_dir_only
  每个 scene/model_seed 有 23020 test points
  总计 230200 points

当前 A0D-HS:
  phase2_me_a0_dir_hs_only
  每个 scene/model_seed 只有 201 test points
  总计 2010 points
```

因此当前 HS 可以证明机制方向性增强，但不能直接证明相对旧 A0D
的 RMSE/P95/P99/Max 提升。

本轮补跑目标：

```text
只跑 ME-RGCF-A0D-HS
复用旧 A0D 的 fixed mixed dataset
复用旧 A0D 的 train/val/test seed 协议
复用旧 A0D 的 model seeds / epochs / lr / batch_size / hidden_dim
输出到独立目录，不覆盖旧结果
```

## 2. 必须复用的旧 A0D 配置

旧 A0D 正式结果目录：

```text
E:\migration_packages\results\phase2_me_a0_dir_only
```

旧 A0D 使用的 mixed dataset store：

```text
D:\code\python\project-2\project_root\dataset_store\20260616_201032__phase1r_s1r_s2r_mixed_nominal__phase1r_s1r_s2r_mixed_n__52548661
```

旧 A0D 协议：

```text
methods:          ME-RGCF-A0D
model seeds:      0,1,2,3,4
train seeds:      10-69 per scene
val seeds:        70-89 per scene
test seeds:       90-109
epochs:           80
lr:               0.001
batch size:       64
hidden dim:       64
test points:      23020 per scene/model_seed
```

本轮 A0D-HS 必须让 `num_points` 也变成：

```text
S1R: 23020 per model_seed
S2R: 23020 per model_seed
total: 230200
```

如果补跑后的 `phase1r_run_summary.csv` 里仍然是每个 learned row `num_points=201`，
说明没有复用旧 fixed dataset，这次补跑仍然不能作为公平对照。

## 3. GPU 端运行前检查

进入项目目录：

```powershell
cd D:\code\python\project-2
```

检查旧 A0D dataset store 是否存在：

```powershell
Test-Path 'D:\code\python\project-2\project_root\dataset_store\20260616_201032__phase1r_s1r_s2r_mixed_nominal__phase1r_s1r_s2r_mixed_n__52548661'
```

必须返回：

```text
True
```

如果返回 `False`，不要直接重跑 HS。需要先从旧 A0D 运行环境恢复这个
`dataset_store` 目录；否则脚本会重新生成另一份 dataset，结果仍然不公平。

建议再确认旧 A0D summary 指向的 dataset：

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' -c "import pandas as pd; df=pd.read_csv(r'E:\migration_packages\results\phase2_me_a0_dir_only\phase1r_run_summary.csv'); print(df[df['method'].eq('ME-RGCF-A0D')][['scenario_id','model_seed','dataset_dir','num_points']].head().to_string(index=False))"
```

预期看到 `dataset_dir` 为：

```text
D:\code\python\project-2\project_root\dataset_store\20260616_201032__phase1r_s1r_s2r_mixed_nominal__phase1r_s1r_s2r_mixed_n__52548661
```

并且 `num_points=23020`。

## 4. Dry-run

输出目录建议使用新的独立目录：

```text
E:\migration_packages\results\phase2_me_a0_dir_hs_fair_only
```

dry-run 命令：

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' project_root\scripts\run_phase1r_rgcf_benchmark.py `
  --dry-run `
  --methods me-a0-dir-hs `
  --model-seeds 0,1,2,3,4 `
  --train-seeds 10-69 `
  --val-seeds 70-89 `
  --test-seeds 90-109 `
  --epochs 80 `
  --lr 0.001 `
  --batch-size 64 `
  --hidden-dim 64 `
  --out-dir E:\migration_packages\results\phase2_me_a0_dir_hs_fair_only
```

预期：

```text
methods 中包含 ME-RGCF-A0D-HS
model_init_seeds 为 0,1,2,3,4
runs 为 17
```

说明：当前 benchmark 脚本会固定把 6 个 rule baseline 也写入计划并评估。
这不是补跑重点；本轮唯一新增 learned method 应该只有 `ME-RGCF-A0D-HS`。

## 5. 正式补跑命令

环境变量建议：

```powershell
$env:RGCF_SIM_WORKERS='4'
$env:RGCF_NUM_WORKERS='2'
$env:RGCF_DISABLE_COMPILE='1'
$env:PYTHONIOENCODING='utf-8'
```

正式命令：

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' project_root\scripts\run_phase1r_rgcf_benchmark.py `
  --device cuda `
  --resume `
  --methods me-a0-dir-hs `
  --mixed-dataset-dir 'D:\code\python\project-2\project_root\dataset_store\20260616_201032__phase1r_s1r_s2r_mixed_nominal__phase1r_s1r_s2r_mixed_n__52548661' `
  --model-seeds 0,1,2,3,4 `
  --train-seeds 10-69 `
  --val-seeds 70-89 `
  --test-seeds 90-109 `
  --epochs 80 `
  --lr 0.001 `
  --batch-size 64 `
  --hidden-dim 64 `
  --out-dir E:\migration_packages\results\phase2_me_a0_dir_hs_fair_only
```

不要加：

```text
--smoke
--smoke-duration
--force-regenerate-sims
--no-sim-cache
```

其中 `--mixed-dataset-dir` 是本方案的关键。它确保 HS 评估使用旧 A0D
的同一份 raw sims / dataset split。

## 6. 运行后验收

运行完成后先检查点数：

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' -c "import pandas as pd; df=pd.read_csv(r'E:\migration_packages\results\phase2_me_a0_dir_hs_fair_only\phase1r_run_summary.csv'); learned=df[df['method'].eq('ME-RGCF-A0D-HS')]; print(learned[['scenario_id','model_seed','num_points','rmse','p95','p99','max','dataset_dir']].to_string(index=False)); print('total learned points=', int(learned['num_points'].sum()))"
```

必须满足：

```text
每行 num_points = 23020
total learned points = 230200
dataset_dir = 旧 A0D dataset store
```

再检查 aggregate：

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' -c "import pandas as pd; base=r'E:\migration_packages\results\phase2_me_a0_dir_hs_fair_only'; print(pd.read_csv(base+r'\phase1r_aggregate_overall.csv').to_string(index=False)); print(); print(pd.read_csv(base+r'\phase1r_aggregate_by_scene.csv').to_string(index=False))"
```

## 7. 诊断命令

公平 HS 补跑完成后，重新跑 directionality 诊断：

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' project_root\scripts\diagnose_a0d_directionality.py `
  --a0d-dir E:\migration_packages\results\phase2_me_a0_dir_hs_fair_only `
  --a0-dir E:\migration_packages\results\phase2_me_a0_only `
  --out-dir E:\migration_packages\results\analysis_outputs\a0d_hs_fair_directionality_diagnostics
```

这里 `--a0d-dir` 指向 A0D-like 待诊断目录，因此可以传 A0D-HS。

## 8. 公平对照汇总方式

对比旧 A0D 与公平 HS：

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' -c "import pandas as pd; pairs=[('A0D',r'E:\migration_packages\results\phase2_me_a0_dir_only'),('A0D-HS fair',r'E:\migration_packages\results\phase2_me_a0_dir_hs_fair_only')]; rows=[]; scenes=[]; 
for label,base in pairs:
    o=pd.read_csv(base+r'\phase1r_aggregate_overall.csv')
    o=o[o['method'].astype(str).str.contains('A0D', regex=False)].copy()
    o.insert(0,'label',label); rows.append(o)
    s=pd.read_csv(base+r'\phase1r_aggregate_by_scene.csv')
    s=s[s['method'].astype(str).str.contains('A0D', regex=False)].copy()
    s.insert(0,'label',label); scenes.append(s)
print('OVERALL'); print(pd.concat(rows)[['label','method','n_runs','num_points_total','rmse_mean','p95_mean','p99_mean','max_mean']].to_string(index=False))
print('\nBY_SCENE'); print(pd.concat(scenes)[['label','scenario_id','method','n_runs','num_points_total','rmse_mean','p95_mean','p99_mean','max_mean']].to_string(index=False))"
```

如果两边 `num_points_total` 都是 `230200`，才可以正式判断：

```text
A0D-HS 是否替代 A0D
A0D-HS 是否只是 attention 更好但 RMSE 退化
A0D-HS 是否只改善 P95/P99/tail 而不改善 RMSE
```

## 9. 决策规则

补跑完成后按以下规则判断：

```text
1. 如果 HS 的 RMSE/P95/P99/Max 都不差于旧 A0D，且方向性指标保持更高：
   可以把 HS 作为下一版稳定 A0D 候选。

2. 如果 HS 的 high_spread_corr / hit_rate 更高，但 RMSE 或 P95/P99 退化：
   HS 只保留为 loss 消融，不替换当前 A0D。

3. 如果 HS 的方向性更高，但 delta_w / delta_cov / worst_low_weight_rate 仍弱：
   问题在 downstream fusion head，而不是 pair feature 或 attention 排序。

4. 如果只有某个 seed 明显恶化：
   优先分析初始化敏感性、LR、early stopping，不急着改结构。
```

## 10. 本轮不要做的事

```text
不重跑旧 A0D
不重跑 RGCF / A0
不引入时间窗口
不改模型结构
不改变 HS loss 参数
不重新生成 dataset
```

本轮只补齐一个问题：

```text
ME-RGCF-A0D-HS 在旧 A0D fixed raw dataset 上的公平性能与机制诊断。
```
