# Phase2 ME-RGCF-A0D-HS GPU 执行方案

更新时间：2026-06-17

## 1. 实验目的

`ME-RGCF-A0D-HS` 是当前 `ME-RGCF-A0D` 的稳固型训练变体，不改变图结构，
只把 M->P directional auxiliary loss 从全 M softmax 目标改为 high-spread
aware 的拆分目标：

```text
L_dir = 0.25 * L_id + 1.0 * L_ev_hs
```

其中：

- `L_id`：弱约束 P_i 不完全丢掉自己的 track measurement M_i。
- `L_ev_hs`：只在 evidence residual 对 P1/P2/P3 差异明显的窗口中，加强
  evidence-to-track 方向性监督。
- high-spread gate 使用 batch 内 residual spread 的 `q70 -> q90` 平滑区间：

```text
gate = clip((spread - q70) / (q90 - q70 + eps), 0, 1)
```

本轮目标不是引入时间窗口、污染训练或新结构，而是验证：

```text
A0D-HS 是否能提高 high_spread_corr / hit_rate，
同时保持 A0D 已有 RMSE/P95/P99 优势不退化。
```

## 2. 与已有 A0D 结果的关系

已有 `ME-RGCF-A0D` 结果可以复用，不需要重跑：

```text
E:\migration_packages\results\phase2_me_a0_dir_only
```

本轮只新增训练：

```text
ME-RGCF-A0D-HS
```

对比时使用：

```text
旧结果：RGCF / ME-RGCF-A0 / ME-RGCF-A0D
新结果：ME-RGCF-A0D-HS
```

为了严格公平，GPU 端如果已有 Phase1R S1R/S2R mixed raw dataset store，应在
HS 正式运行时传入同一个 `--mixed-dataset-dir`。如果不确定数据目录是否存在，
可以先不传，由脚本按相同 seed 协议生成 Phase1R clean raw sims；但论文级对比
应优先固定 dataset store。

## 3. 新方法开关

benchmark 新增 learned method：

```text
--methods me-a0-dir-hs
```

等价别名：

```text
--methods me-a0d-hs
--methods a0d-hs
```

显示名：

```text
ME-RGCF-A0D-HS
```

模型 preset suffix：

```text
me_rgcf_a0_dir_hs
```

模型名：

```text
me_rgcf_a0_dir_hs
```

该模型复用 `MeasurementEvaluatedRGCFA0Directional` 网络结构，只改变训练 loss。

## 4. GPU 运行命令

进入项目根目录：

```powershell
cd D:\code\python\project-2
```

建议环境变量：

```powershell
$env:RGCF_SIM_WORKERS='4'
$env:RGCF_NUM_WORKERS='2'
$env:RGCF_DISABLE_COMPILE='1'
```

### 4.1 Dry-run

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' project_root\scripts\run_phase1r_rgcf_benchmark.py --dry-run --methods me-a0-dir-hs --out-dir project_root\results\phase2_me_a0_dir_hs_dryrun
```

预期：

- `phase1r_plan.json/csv` 正常生成。
- methods 中包含 `ME-RGCF-A0D-HS`。
- formal 计划为规则 baseline + `ME-RGCF-A0D-HS * 5 model seeds`。

### 4.2 GPU smoke

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' project_root\scripts\run_phase1r_rgcf_benchmark.py --smoke --device cuda --methods me-a0-dir-hs --out-dir project_root\results\phase2_me_a0_dir_hs_smoke_cuda
```

预期：

- 能训练 `ME-RGCF-A0D-HS`。
- train history 中 `loss_mp_dir_identity`、`loss_mp_dir_evidence`、`mean_mp_hs_gate`
  字段非空。
- `eval_details` 仍包含 `mp_attn_p*_m*`、`mp_pair_res_p*_m*`、`mp_pair_rank_p*_m*`。

### 4.3 正式 HS-only 运行

如果已有可复用 Phase1R mixed dataset：

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' project_root\scripts\run_phase1r_rgcf_benchmark.py --device cuda --resume --methods me-a0-dir-hs --mixed-dataset-dir project_root\dataset_store\<phase1r_s1r_s2r_mixed_dataset_dir> --out-dir project_root\results\phase2_me_a0_dir_hs_only
```

如果不确定 dataset store：

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' project_root\scripts\run_phase1r_rgcf_benchmark.py --device cuda --resume --methods me-a0-dir-hs --out-dir project_root\results\phase2_me_a0_dir_hs_only
```

## 5. 诊断命令

HS 完成后，运行离线 directionality 诊断：

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' project_root\scripts\diagnose_a0d_directionality.py --a0d-dir project_root\results\phase2_me_a0_dir_hs_only --a0-dir project_root\results\phase2_me_a0_only --out-dir project_root\analysis_outputs\a0d_hs_directionality_diagnostics
```

注意：该诊断脚本的 `--a0d-dir` 参数表示“待诊断的 A0D-like 方法目录”，可以指向
A0D-HS 输出目录。

## 6. 验收标准

性能不退化：

```text
ME-RGCF-A0D-HS RMSE 不高于旧 A0D 超过 1%
P95/P99 不高于旧 A0D
Max 不明显恶化，尤其 S2R model_seed4
```

机制增强：

```text
row_std_mean 保持 >= 0.05
high_spread_corr 高于旧 A0D，目标 >= 0.18
worst_track_hit_rate 高于旧 A0D，目标 >= 0.45
worst_low_weight_rate 保持 >= 0.80
worst_high_cov_rate 不低于旧 A0D
```

如果 HS 提高 high-spread directionality 但 RMSE/P95/P99 退化，则不替换当前 A0D；
仅保留为 loss 消融。如果 HS 机制指标和 RMSE 同时改善，才考虑把 HS 作为下一版
稳定 A0D。

## 7. 当前旧 A0D 参考线

旧 A0D 离线诊断结果：

```text
row_std_mean      ~= 0.0647
raw_corr          ~= 0.0845
high_spread_corr  ~= 0.1468
hit_rate          ~= 0.4201
delta_w_mean      ~= -0.2547
delta_cov_mean    ~= +1.5429
```

场景差异：

```text
S1R high_spread_corr ~= 0.0917
S2R high_spread_corr ~= 0.2019
```

HS 的重点是稳固 S1R 和低相关 seed，而不是只提升 S2R。
