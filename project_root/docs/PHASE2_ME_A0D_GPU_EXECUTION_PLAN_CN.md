# Phase2 ME-RGCF-A0D GPU 执行方案

更新时间：2026-06-16

## 1. 实验目的

本轮实验验证 `ME-RGCF-A0D` 是否解决 `ME-RGCF-A0` 中的核心结构问题：

```text
数据里有 evidence -> track 的方向性，
但旧 ME-A0 没有把这个方向性喂给 M->P attention。
```

`ME-RGCF-A0D` 新增 `mp_pair_feat [K,3,5,8]`，把 `evidence_residual_to_track [K,E,3]`
保留为 P-M pair 级别特征，使 M4/M5 能表达“我更反对 P1/P2/P3 中的哪一个”。

本轮仍然是 clean 验证，不引入污染、不引入时间记忆、不引入 `delta_x`。

## 2. 是否需要重新生成数据

不需要重新设计仿真场景，也不需要重新生成轨迹级 raw sim，只要已有 Phase1R raw dataset store 即可。

需要注意：

- 如果 GPU 端已有 Phase1R raw dataset store，且其中 sim 包含 `evidence_residual_to_track [K,E,3]`，可以用 `--mixed-dataset-dir` 复用。
- 如果 GPU 端只有旧 P0/P11/P12/四后验传感器数据，不能复用。
- 如果不确定 dataset store 是否为 Phase1R 新格式，建议不要传 `--mixed-dataset-dir`，让脚本自动重新生成 Phase1R raw sims。
- `ME-RGCF-A0D` 需要重新训练模型。不能复用 `RGCF` 或 `ME-RGCF-A0` 的 checkpoint。

推荐默认做法：

```text
不手动指定 --mixed-dataset-dir，
由 benchmark 自动生成或复用当前 Phase1R S1R/S2R mixed raw dataset。
```

这会保证 `RGCF`、`ME-RGCF-A0`、`ME-RGCF-A0D` 在同一批 S1R/S2R train/val/test seeds 上公平对比。

## 3. 对照组

默认规则 baseline 固定包含：

```text
single-T1
single-T2
single-T3
AVG-3T
WAA-MM-3T
CI-3T
```

本轮 learned 对照组：

```text
RGCF
ME-RGCF-A0
ME-RGCF-A0D
```

方法含义：

| 方法 | 作用 |
| --- | --- |
| `RGCF` | 当前稳定 baseline，evidence pooling/broadcast |
| `ME-RGCF-A0` | 异构 M/P 图，但 M->P 无 pair-wise 方向性 |
| `ME-RGCF-A0D` | 本轮新模型，M->P attention 使用 pair-wise evidence residual |

不进入本轮对比：

- P0/P1/P11/P12 旧版本树
- SNF-A
- M4/RGCF-V5 消融树
- 污染、dropout、bias ramp 等故障场景
- 时间记忆版本 ME-A1

## 4. GPU 执行命令

进入项目根目录：

```powershell
cd D:\code\python\project-2
```

建议环境变量：

```powershell
$env:RGCF_SIM_WORKERS='4'
$env:RGCF_NUM_WORKERS='2'
```

如果 Windows GPU 端 `torch.compile` 不稳定：

```powershell
$env:RGCF_DISABLE_COMPILE='1'
```

### 4.1 Dry-run

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' project_root\scripts\run_phase1r_rgcf_benchmark.py --dry-run --methods rgcf,me-a0,me-a0-dir --out-dir project_root\results\phase2_me_a0_dir_dryrun
```

预期：

- 输出 `phase1r_plan.json/csv`
- methods 中包含 `RGCF`、`ME-RGCF-A0`、`ME-RGCF-A0D`
- formal 默认计划为 6 个规则 baseline + 3 learned methods * 5 model seeds

### 4.2 GPU smoke

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' project_root\scripts\run_phase1r_rgcf_benchmark.py --smoke --device cuda --methods me-a0-dir --out-dir project_root\results\phase2_me_a0_dir_smoke_cuda
```

预期：

- 能生成 Phase1R S1R/S2R clean sims
- 能训练 `ME-RGCF-A0D`
- `eval_details` 中包含 `mp_pair_res_p*_m*`、`mp_pair_rank_p*_m*`
- `phase1r_run_summary.csv` 中包含 attention 诊断字段

### 4.3 正式对比

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' project_root\scripts\run_phase1r_rgcf_benchmark.py --device cuda --resume --methods rgcf,me-a0,me-a0-dir --out-dir project_root\results\phase2_me_a0_dir_compare
```

如果已经确认有可复用 Phase1R raw dataset store，可加：

```powershell
--mixed-dataset-dir D:\code\python\project-2\project_root\dataset_store\<phase1r_s1r_s2r_mixed_dataset_dir>
```

不确定时不要加，避免误用旧数据。

## 5. 输出文件

正式结果目录：

```text
project_root/results/phase2_me_a0_dir_compare
```

重点文件：

```text
phase1r_run_summary.csv
phase1r_aggregate_by_scene.csv
phase1r_aggregate_overall.csv
phase1r_sensor_health_by_scene.csv
phase1r_evidence_residual_report.csv
eval_details/*.csv
```

新增诊断字段：

```text
mean_mp_attn_entropy
mean_mp_attn_row_std
p95_mp_attn_row_std
mean_mp_attn_evidence_mass
mean_mp_attn_own_track_mass
mean_mp_attn_residual_corr
```

`ME-RGCF-A0D` 的 detail 额外包含：

```text
mp_pair_res_p{i}_m{j}
mp_pair_rank_p{i}_m{j}
```

## 6. 验收标准

基础闭环：

- dry-run 成功。
- GPU smoke 成功。
- 正式实验中 `RGCF`、`ME-RGCF-A0`、`ME-RGCF-A0D` 均完成 5 个 model seeds。

性能：

- `ME-RGCF-A0D` 相对 `ME-RGCF-A0` 的 RMSE 退化不超过 1%。
- `ME-RGCF-A0D` 在 S1R/S2R 上应明显优于 `CI-3T` 与 best single track sensor。
- S2R 中 `cov_scale_s2` 应仍明显高于 T1/T3，说明 T2 弱跟踪风险仍被识别。

结构诊断：

- `mean_mp_attn_row_std` 应至少比当前 ME-A0 高 10 倍。
- `mean_mp_attn_residual_corr` 应为正，目标值 `> 0.15`。
- `mp_pair_rank` 高的 evidence pair 应获得更高 M->P attention。
- `E1/E2` 仍不出现在最终 state fusion weights 中。

## 7. 结果判断

如果 `ME-RGCF-A0D` 性能接近 RGCF/ME-A0，且 attention 方向性明显提升：

```text
说明 pair-wise evidence residual 成功进入 M->P attention。
下一步可以考虑 ME-A1 时间记忆。
```

如果 attention 方向性提升但 RMSE 没有提升：

```text
先分析 pair feature 和 directional loss 权重，
不要直接进入时间记忆。
```

如果 attention 仍然塌缩：

```text
优先检查 mp_pair_feat 是否正确落盘，
再检查 me_rgcf_mp_dir_loss_weight 是否启用。
```

