# 任务清单: Phase1R RGCF GPU 正式实验

**创建时间**: 2026-06-15
**创建人**: 开发端 Codex
**预计完成**: 2026-06-16
**状态**: [~] 执行中

---

## 0. 协作规则

- **代码来源**: GPU 端必须通过 GitHub `git pull` 获取代码（分支 `codex/phase1r-rgcf-gpu`）。
- **GPU 端职责**: 只执行实验、回填本任务清单、提交小型结果摘要；不修改模型/仿真/训练代码。
- **禁止提交**: `dataset_store/`、`sim_cache/`、`results/` 下的大文件、checkpoint、wheel、zip。
- **异常处理**: 任一步失败时停止后续正式训练，在本文件对应任务中标记 `[!]` 并粘贴关键报错。
- **关键纠正**: Phase1R 不再把 AOA/UWB 作为后验融合节点，改为 3 track (T1/T2/T3) + 2 evidence (E1/E2)。

---

## 任务 01 — 环境预检 & Phase1R 场景验证

- **状态**: [x] 已完成
- **优先级**: P0
- **模型方法**: N/A
- **场景**: S1R (basic) / S2R (maneuver)
- **目标**: 确认 GPU 环境、CUDA、Phase1R 场景预设和传感器角色均正确。

### 执行命令

```powershell
cd D:\code\python\project-2\project_root

# Step 1: CUDA + imports check
D:\envs\nfdkf-gpu\Scripts\python.exe -c "
import torch; print('PyTorch:', torch.__version__);
print('CUDA available:', torch.cuda.is_available());
print('CUDA version:', torch.version.cuda);
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A');
print('GPU count:', torch.cuda.device_count())
"

# Step 2: Phase1R scene presets load check
D:\envs\nfdkf-gpu\Scripts\python.exe -c "
from core.config_loader import load_experiment_bundle
from core.types import RunRequest
from simulation.scenario_factory import build_scenario_from_bundle

for sid, preset in [('S1R', 'phase1r_basic_3track_2evidence_nominal_rgcf'),
                     ('S2R', 'phase1r_maneuver_3track_2evidence_nominal_rgcf')]:
    bundle = load_experiment_bundle(RunRequest(mode='train', preset_name=preset, device='cuda'))
    artifacts = build_scenario_from_bundle(bundle).build()
    layout = artifacts.sensor_layout
    roles = [(n.name, n.sensor_type, getattr(n, 'sensor_role', 'track')) for n in layout.sensors]
    print(f'[{sid}] preset={preset}')
    print(f'[{sid}] sensors={roles}')
    print(f'[{sid}] num_sensors={len(layout.sensors)}')
    tracks = [n.name for n in layout.sensors if getattr(n, 'sensor_role', 'track') == 'track']
    evidences = [n.name for n in layout.sensors if getattr(n, 'sensor_role', 'track') == 'evidence']
    print(f'[{sid}] track_sensors={tracks}')
    print(f'[{sid}] evidence_sensors={evidences}')
    assert len(tracks) == 3, f'{sid}: expected 3 track sensors, got {len(tracks)}'
    assert len(evidences) == 2, f'{sid}: expected 2 evidence sensors, got {len(evidences)}'
print('[phase1r_scenes] ok')
"
```

### 验收标准

- [x] `torch.cuda.is_available()` 为 `True`
- [x] GPU 数量 ≥ 1 (实际: 2x RTX 3090 Ti)
- [x] S1R/S2R 各 5 个传感器 (3 track + 2 evidence)
- [x] track 传感器: T1(GPS2D), T2(Radar RB), T3(Radar RB)
- [x] evidence 传感器: E1(AOA-only), E2(UWB range-only)
- [x] 输出 `[phase1r_scenes] ok`
- [x] fault_mode 均为 clean

### 执行记录（GPU端回填）

- **实际开始**: 2026-06-15
- **实际结束**: 2026-06-15
- **CUDA/torch 信息**: PyTorch 2.7.1+cu126, CUDA 12.6, 2x NVIDIA GeForce RTX 3090 Ti
- **异常备注**: 无

---

## 任务 02 — Phase1R dry-run

- **状态**: [x] 已完成
- **优先级**: P0
- **模型方法**: RGCF
- **场景**: mixed(S1R+S2R)
- **目标**: 确认实验计划正确，方法列表、种子范围、场景映射无误。

### 执行命令

```powershell
cd D:\code\python\project-2\project_root

D:\envs\nfdkf-gpu\Scripts\python.exe -u scripts\run_phase1r_rgcf_benchmark.py `
  --dry-run `
  --device cuda `
  --out-dir results\phase1r_rgcf_dryrun
```

### 验收标准

- [x] `[train_scene_set] S1R+S2R`
- [x] `[methods] single-T1, single-T2, single-T3, AVG-3T, WAA-MM-3T, CI-3T, RGCF`
- [x] `[model_init_seeds] 0, 1, 2, 3, 4`
- [x] `[runs]` = 17 (12 rule baselines + 5 model seeds)
- [x] `phase1r_plan.json` 存在且结构正确

### 执行记录（GPU端回填）

- **实际开始**: 2026-06-15
- **实际结束**: 2026-06-15
- **plan 路径**: `results/phase1r_rgcf_dryrun/phase1r_plan.json`
- **异常备注**: 无。17 runs: 12 rule baselines + 5 RGCF model seeds. All 7 methods verified.

---

- **状态**: [x] 已完成
- **优先级**: P0
- **模型方法**: RGCF
- **场景**: mixed(S1R+S2R)
- **模型种子列表**: 0
- **训练种子范围**: smoke 默认
- **训练参数**: smoke 默认，epochs=2
- **特殊配置**: 无（smoke 自动使用 smoke_duration=20s + 少量种子）

### 执行命令

```powershell
cd D:\code\python\project-2\project_root

$env:RGCF_SIM_WORKERS='4'
$env:RGCF_NUM_WORKERS='2'
$env:RGCF_DISABLE_COMPILE='1'

D:\envs\nfdkf-gpu\Scripts\python.exe -u scripts\run_phase1r_rgcf_benchmark.py `
  --smoke `
  --device cuda `
  --out-dir results\phase1r_rgcf_smoke_cuda
```

### 验收标准

- [ ] `results/phase1r_rgcf_smoke_cuda/phase1r_run_summary.csv` 存在
- [ ] `results/phase1r_rgcf_smoke_cuda/phase1r_aggregate_by_scene.csv` 存在
- [ ] `results/phase1r_rgcf_smoke_cuda/phase1r_aggregate_overall.csv` 存在
- [ ] `results/phase1r_rgcf_smoke_cuda/phase1r_sensor_health_by_scene.csv` 存在
- [ ] `results/phase1r_rgcf_smoke_cuda/phase1r_evidence_residual_report.csv` 存在
- [ ] S1R/S2R 均有 RGCF learned 行
- [ ] S1R/S2R 均有 6 种 rule baseline 行
- [ ] 所有 `rmse/p95/p99/max` 为有限数值，无 NaN/Inf
- [ ] `eval_details` 中只出现 T1/T2/T3 的 `w_s*`，E1/E2 不应成为融合权重节点
- [ ] `phase1r_sensor_health_by_scene.csv` 中 T1/T2/T3 不出现数量级崩溃

### 额外检查命令

```powershell
Import-Csv results\phase1r_rgcf_smoke_cuda\phase1r_aggregate_by_scene.csv |
  Select-Object scenario_id,method,method_category,rmse_mean,p95_mean,p99_mean,max_mean |
  Format-Table -AutoSize
```

### 执行记录（GPU端回填）

- **实际开始**: 2026-06-15 20:50
- **实际结束**: 2026-06-15 20:52
- **结果路径**: `project_root/results/phase1r_rgcf_smoke_cuda/`
- **mixed dataset 路径**: `dataset_store/20260615_205057__phase1r_s1r_s2r_mixed_nominal__...__64640815`
- **关键指标摘要**:
  | S1R RGCF RMSE=1.21 (best), WAA-MM-3T=1.25, AVG-3T=1.25
  | S2R RGCF RMSE=1.41 (best), WAA-MM-3T=1.51, AVG-3T=1.53
  | Overall: RGCF=1.31 > WAA-MM-3T=1.38 > AVG-3T=1.39 > single-T2=1.69
  | RGCF is #1 across all methods in both scenes
  | Weight columns verified: only T1/T2/T3 (w_s1~w_s3), no evidence weights
- **异常备注**: 无。RGCF 是 7 个方法中表现最好的

---

## 任务 04 — Phase1R 正式主实验

- **状态**: [ ] 待执行
- **优先级**: P0
- **模型方法**: RGCF
- **场景**: mixed(S1R+S2R)
- **模型种子列表**: 0,1,2,3,4
- **训练种子范围**: 10-69(train) / 70-89(val) / 90-109(test)
- **训练参数**: epochs=80, lr=1e-3, batch_size=64, hidden_dim=64
- **前置条件**: 任务 03 已完成且无异常

### 执行命令

```powershell
cd D:\code\python\project-2\project_root

$env:RGCF_SIM_WORKERS='4'
$env:RGCF_NUM_WORKERS='2'
$env:RGCF_DISABLE_COMPILE='1'

D:\envs\nfdkf-gpu\Scripts\python.exe -u scripts\run_phase1r_rgcf_benchmark.py `
  --device cuda `
  --model-seeds 0,1,2,3,4 `
  --train-seed-range 10-69 `
  --val-seed-range 70-89 `
  --test-seed-range 90-109 `
  --epochs 80 `
  --lr 1e-3 `
  --batch-size 64 `
  --hidden-dim 64 `
  --out-dir results\phase1r_rgcf_compare
```

### 验收标准

- [ ] RGCF learned 结果为 10 行：5 model seeds × 2 scenes
- [ ] rule baseline 结果为 12 行：6 baselines × 2 scenes
- [ ] `phase1r_run_summary.csv` 存在
- [ ] `phase1r_aggregate_by_scene.csv` 存在
- [ ] `phase1r_aggregate_overall.csv` 存在
- [ ] `phase1r_sensor_health_by_scene.csv` 存在
- [ ] `phase1r_evidence_residual_report.csv` 存在
- [ ] S1R 与 S2R 指标不应完全相同
- [ ] 所有 `rmse/p95/p99/max` 为有限数值，无 NaN/Inf
- [ ] RGCF 应至少优于 AVG-3T
- [ ] RGCF 应接近或优于 WAA-MM-3T
- [ ] S1R RGCF RMSE < 5.0 (参考 CPU smoke: ~1.2)
- [ ] S2R RGCF RMSE < 6.0 (参考 CPU smoke: ~1.4)
- [ ] Max error 与 P95 应比单一弱 track 节点更稳定

### 执行记录（GPU端回填）

- **实际开始**: 2026-06-15 20:55
- **实际结束**: 
- **结果路径**: `project_root/results/phase1r_rgcf_compare/`
- **mixed dataset 路径**: 
- **overall 指标摘要**:
  | (回填)
- **by-scene 指标摘要**:
  | (回填)
- **异常备注**: 

---

## GPU 端完成后提交要求

GPU 端完成后只提交：

- 本文件的执行记录回填
- 可选：小型结果摘要文件，例如 `project_root/results_summary_phase1r_gpu_20260615.md`

不要提交：

- `dataset_store/`
- `sim_cache/`
- `results/`
- `*.pt`
- `*.pkl`
- `*.zip`
- `*.whl`

建议提交命令：

```powershell
git status --short
git add project_root\task_checklists\TASK_PHASE1R_GPU_20260615.md
git commit -m "docs: update phase1r gpu execution checklist"
git push
```
