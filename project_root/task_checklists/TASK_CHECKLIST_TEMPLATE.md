# GPU 实验任务清单模板

本模板用于当前 Phase1R / ME-RGCF-A0 主线实验交接。

当前默认主线：

```text
Phase1R RGCF stable baseline
ME-RGCF-A0 optional heterogeneous graph upgrade
```

历史 P0/P1/P11/P12/SNF/M4 方案只作为归档诊断材料，不应作为本模板中的默认方法。

## 批次信息

- **任务名称**:
- **创建时间**:
- **执行机器**:
- **Python 环境**:
- **目标阶段**: Phase1R RGCF / ME-A0 / RGCF+ME-A0 compare
- **状态**: `[ ] 待执行` / `[~] 执行中` / `[x] 完成` / `[!] 异常`

## 方法范围

- **默认 baseline**: `RGCF`
- **可选升级**: `ME-RGCF-A0`
- **规则 baseline**:
  - `single-T1`
  - `single-T2`
  - `single-T3`
  - `AVG-3T`
  - `WAA-MM-3T`
  - `CI-3T`
- **禁止默认继承**:
  - P0/P1/P11/P12 旧版本树
  - SNF-A 旧构想
  - M4/RGCF-V5 peer-consistency 消融树
  - AOA/UWB 作为后验融合节点的旧四后验方案

## 数据与场景

- **dataset_store 批次**:
- **场景**:
  - `S1R`: `phase1r_basic_3track_2evidence_nominal`
  - `S2R`: `phase1r_maneuver_3track_2evidence_nominal`
- **污染设置**: clean only
- **传感器角色**:
  - `T1/T2/T3`: track posterior sensors
  - `E1/E2`: evidence measurement sensors

## 运行命令

Dry-run:

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' project_root\scripts\run_phase1r_rgcf_benchmark.py --dry-run --methods rgcf,me-a0 --out-dir project_root\results\phase2_me_a0_dryrun
```

Smoke:

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' project_root\scripts\run_phase1r_rgcf_benchmark.py --smoke --device cuda --methods me-a0 --out-dir project_root\results\phase2_me_a0_smoke_cuda
```

Formal:

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' project_root\scripts\run_phase1r_rgcf_benchmark.py --device cuda --resume --methods rgcf,me-a0 --out-dir project_root\results\phase2_me_a0_compare
```

## 验收指标

- [ ] `T1/T2/T3` 单传感器 EKF 不出现数量级崩溃
- [ ] `E1/E2` 不出现在最终 state fusion weights 中
- [ ] `RGCF` 至少不弱于 `AVG-3T`
- [ ] `ME-RGCF-A0` 相对 `RGCF` 不明显退化
- [ ] `eval_details` 包含 `w_s*`、`cov_scale_s*`、`g_s*`
- [ ] ME-A0 输出 `mp_attn_p*_m*` 与 `mm_attn_m*_m*`
- [ ] S1R/S2R 的 RMSE、P95、Max error 均完成汇总

## 执行记录

- **开始时间**:
- **结束时间**:
- **结果目录**:
- **主要结论**:
- **异常 seed**:
- **下一步建议**:
