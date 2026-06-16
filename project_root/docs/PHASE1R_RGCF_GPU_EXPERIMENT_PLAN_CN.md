# Phase1R RGCF GPU 实验计划

## 0. 本次上传版执行范围（2026-06-16）

本次 GitHub 上传包含两部分：

1. `Phase1R RGCF` 稳定基线代码与文档入口整理。
2. `ME-RGCF-A0` 最小异构图升级代码。

GPU 端拉取后优先执行以下三步：

```powershell
cd D:\code\python\project-2

& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' project_root\scripts\run_phase1r_rgcf_benchmark.py --dry-run --methods rgcf,me-a0 --out-dir project_root\results\phase2_me_a0_dryrun

& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' project_root\scripts\run_phase1r_rgcf_benchmark.py --smoke --device cuda --methods me-a0 --out-dir project_root\results\phase2_me_a0_smoke_cuda

& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' project_root\scripts\run_phase1r_rgcf_benchmark.py --device cuda --resume --methods rgcf,me-a0 --out-dir project_root\results\phase2_me_a0_compare
```

当前方法范围：

- 默认 learned baseline：`RGCF`
- 本次升级方法：`ME-RGCF-A0`
- 规则 baseline：`single-T1/T2/T3`、`AVG-3T`、`WAA-MM-3T`、`CI-3T`

当前不进入默认对比：

- P0/P1/P11/P12 旧版本树
- SNF-A 旧构想
- M4/RGCF-V5 peer-consistency 消融树
- 含污染或故障场景

关键验收：

- `T1/T2/T3` 单独 EKF 不出现数量级崩溃。
- `E1/E2` 不出现在最终 state fusion weights 中。
- `ME-RGCF-A0` 的 `eval_details` 正常落盘 `mp_attn_p*_m*` 与 `mm_attn_m*_m*`。
- `ME-RGCF-A0` 在 S1R/S2R 上不明显退化于 RGCF。

## 1. 实验目的

本次实验从 Phase1R 重新出发，验证纠正后的 RGCF 信息流是否能够在无污染条件下完成多传感器融合任务。核心纠正点是：不再把无法独立完成 EKF 跟踪的 AOA/UWB 量测源作为后验融合节点，而是将传感器明确拆成 3 个后验跟踪传感器与 2 个量测证据传感器。

实验目标：

- 验证 `T1/T2/T3` 三个 track 传感器各自能够独立 EKF 跟踪。
- 验证 `E1/E2` 只作为 measurement evidence，不参与后验状态权重融合。
- 对比单传感器、经典基础融合算法与完整 `RGCF`。
- 重点观察 RMSE、P95、Max error、传感器健康度、RGCF 权重与协方差校准行为。

## 2. 场景与传感器设计

默认只运行两个 clean 场景：

| 场景 | preset | 目的 |
| --- | --- | --- |
| S1R | `phase1r_basic_3track_2evidence_nominal` | 基础跟踪场景 |
| S2R | `phase1r_maneuver_3track_2evidence_nominal` | 普通机动跟踪场景 |

传感器角色：

| 节点 | 类型 | 角色 | 是否参与后验融合 |
| --- | --- | --- | --- |
| T1 | GPS2D | track posterior sensor | 是 |
| T2 | Radar range-bearing | track posterior sensor | 是 |
| T3 | Radar range-bearing | track posterior sensor | 是 |
| E1 | AOA-only | evidence measurement sensor | 否 |
| E2 | UWB range-only | evidence measurement sensor | 否 |

无污染设定：

- 训练集不引入污染。
- 验证集不引入污染。
- 测试集不引入污染。
- 当前阶段不运行 dropout、pollution、bias ramp 等鲁棒性故障。

## 3. 方法对比

默认方法：

| 方法 | 类型 | 说明 |
| --- | --- | --- |
| `single-T1` | baseline | 只使用 T1 后验 |
| `single-T2` | baseline | 只使用 T2 后验 |
| `single-T3` | baseline | 只使用 T3 后验 |
| `AVG-3T` | baseline | T1/T2/T3 算术平均 |
| `WAA-MM-3T` | baseline | 基于协方差 trace 的 WAA-MM |
| `CI-3T` | baseline | 3-track covariance intersection |
| `RGCF` | learned | evidence-aware reliability and covariance calibrated fusion |

不再默认运行：

- P0
- P1
- P11
- SNF-A
- best-single oracle
- 旧 4 传感器后验融合方案

## 4. RGCF 信息流

仿真输出显式拆分：

- `track_xhat / track_Phat / track_xpred / track_Ppred / track_valid_mask`
- `track_z / track_R / track_innovation / track_nis`
- `evidence_z / evidence_R / evidence_valid_mask`
- `evidence_residual_to_prior / evidence_residual_to_track`

模型输入：

- posterior stream：只编码 T1/T2/T3 的后验状态与协方差。
- track measurement stream：只编码 T1/T2/T3 的 innovation、NIS、R、几何量。
- evidence stream：编码 E1/E2 的原始量测、R、传感器类型、相对 track/prior 的残差。
- cross reliability interaction：用 evidence stream 修正 T1/T2/T3 的可靠性与协方差膨胀系数。

最终融合只在 T1/T2/T3 上执行：

```text
Omega_i = inv(s_i * P_i)
Omega_f = sum_i w_i * Omega_i
eta_f = sum_i w_i * Omega_i * x_i
x_f = inv(Omega_f) * eta_f
P_f = inv(Omega_f)
```

约束：

- `w_i` 只对应 T1/T2/T3。
- E1/E2 不生成 state fusion weight。
- E1/E2 默认不输出 `delta_x` 状态修正。
- evidence 只能影响 reliability logits、covariance scale 与 confidence。

## 5. GPU 运行命令

进入项目目录：

```powershell
cd D:\code\python\project-2\project_root
```

dry-run：

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' scripts\run_phase1r_rgcf_benchmark.py --dry-run --out-dir results\phase1r_rgcf_dryrun
```

GPU smoke：

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' scripts\run_phase1r_rgcf_benchmark.py --smoke --device cuda --out-dir results\phase1r_rgcf_smoke_cuda
```

正式实验：

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' scripts\run_phase1r_rgcf_benchmark.py --device cuda --resume --out-dir results\phase1r_rgcf_compare
```

推荐 GPU 环境变量：

```powershell
$env:RGCF_SIM_WORKERS='4'
$env:RGCF_NUM_WORKERS='2'
```

如果 Windows GPU 端 `torch.compile` 不稳定：

```powershell
$env:RGCF_DISABLE_COMPILE='1'
```

## 6. 输出文件

每次运行输出到 `--out-dir`：

- `phase1r_plan.json`
- `phase1r_plan.csv`
- `phase1r_run_summary.json`
- `phase1r_run_summary.csv`
- `phase1r_aggregate_by_scene.json`
- `phase1r_aggregate_by_scene.csv`
- `phase1r_aggregate_overall.json`
- `phase1r_aggregate_overall.csv`
- `phase1r_sensor_health_by_scene.json`
- `phase1r_sensor_health_by_scene.csv`
- `phase1r_evidence_residual_report.json`
- `phase1r_evidence_residual_report.csv`
- `eval_details/*.csv`

## 7. 验收标准

基础验收：

- dry-run 能生成完整 plan。
- smoke 能完成数据生成、baseline 评估、RGCF 训练和逐场景评估。
- `phase1r_sensor_health_by_scene.csv` 中 T1/T2/T3 不出现数量级崩溃。
- `eval_details` 中只出现 T1/T2/T3 的 `w_s*`，E1/E2 不应成为融合权重节点。

性能验收：

- RGCF 应至少优于 `AVG-3T`。
- RGCF 应接近或优于 `WAA-MM-3T`。
- Max error 与 P95 应比单一弱 track 节点更稳定。

## 8. 当前本地验证

本地已完成：

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' -m py_compile ...
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' project_root\scripts\run_phase1r_rgcf_benchmark.py --dry-run --out-dir project_root\results\phase1r_rgcf_dryrun_2
```

最小 CPU smoke 已跑通：

```powershell
$env:RGCF_SIM_WORKERS='1'
$env:RGCF_NUM_WORKERS='0'
$env:RGCF_DISABLE_COMPILE='1'
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' project_root\scripts\run_phase1r_rgcf_benchmark.py --smoke --device cpu --out-dir project_root\results\phase1r_rgcf_smoke_min_cpu_2 --train-seeds 10 --val-seeds 70 --test-seeds 90 --epochs 1 --model-seeds 0 --no-sim-cache --smoke-duration 20
```

CPU smoke 结果：

| 场景 | AVG-3T RMSE | WAA-MM-3T RMSE | RGCF RMSE |
| --- | ---: | ---: | ---: |
| S1R | 1.2453 | 1.2479 | 1.2021 |
| S2R | 1.5293 | 1.5055 | 1.4028 |

该 smoke 只验证闭环，不作为正式论文或汇报指标。

## 9. 第二阶段 ME-A0 升级状态

当前 GPU 主实验仍以本文档中的 `Phase1R RGCF` 为稳定 baseline。第二阶段的
最小异构图版本 `ME-RGCF-A0` 已经实现为可选方法，但不会替换默认 RGCF。
详见：

`project_root/docs/ME_RGCF_HETEROGENEOUS_TEMPORAL_DESIGN_CN.md`

第二阶段方向：

- 将 measurement 从当前的 context/pooling 形式升级为异构图中的 M 节点。
- 构建 `3P + 5M` typed graph，其中 P 节点为 track 后验，M 节点为 track/evidence 量测。
- 使用 `M-M` 量测一致性、`M→P` cross attention、`P-P` 后验协商三阶段消息传递。
- 在 ME-A1 以后引入时间记忆：M 短窗捕捉量测突变，P 长窗捕捉后验漂移。

第二阶段不改变当前验收标准：

- 默认 benchmark 仍只运行 RGCF。
- 需要显式传入 `--methods me-a0` 或 `--methods rgcf,me-a0` 才运行 ME-RGCF-A0。
- 当前结果表默认比较 `single-T1/T2/T3`、`AVG-3T`、`WAA-MM-3T`、`CI-3T` 和 `RGCF`。
- ME-RGCF-A0 只作为 Phase2 最小异构图消融与创新模型，不引入时间记忆、污染或 `delta_x`。
