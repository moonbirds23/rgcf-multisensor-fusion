# 任务清单: Phase1 GPU 正式重跑

**创建时间**: 2026-06-12
**创建人**: 开发端 Codex
**预计完成**: 2026-06-13
**状态**: [~] 执行中

---

## 0. 协作规则

- **代码来源**: GPU 端必须通过 GitHub `git pull` 获取代码。
- **GPU 端职责**: 只执行实验、回填本任务清单、提交小型结果摘要；不修改模型/仿真/训练代码。
- **禁止提交**: `dataset_store/`、`sim_cache/`、`results/` 下的大文件、checkpoint、wheel、zip。
- **异常处理**: 任一步失败时停止后续正式训练，在本文件对应任务中标记 `[!]` 并粘贴关键报错。
- **旧结果状态**: `phase1_nominal_benchmark_p11_main` 和 `phase1_nominal_benchmark_compare` 属于修复前错误场景结果，禁止复用。

---

## 任务 01 — 同步代码与环境预检

- **状态**: [x] 已完成
- **优先级**: P0
- **模型方法**: N/A
- **场景**: S1_balanced / S2_clustered / S3_maneuver
- **目标**: 确认 GPU 环境、CUDA、Phase 1 场景接入均正确。

### 执行命令

```powershell
cd D:\code\python\project-2
git pull

cd project_root
D:\envs\nfdkf-gpu\Scripts\python.exe -u scripts\gpu_preflight_check.py --require-cuda
```

### 验收标准

- [x] `torch.cuda.is_available()` 为 `True`
- [x] `S1 types=['gps2d', 'radar_rb', 'aoa_only', 'uwb_range_only']`
- [x] `S2 pos=[(-850.0, -350.0), (-350.0, -250.0), (-700.0, 450.0), (-150.0, 350.0)]`
- [x] `S3 truth_last_xy` 不同于 S1/S2
- [x] 输出 `[phase1_scenes] ok`
- [x] 输出 `[preflight] ok`

### 执行记录（GPU端回填）

- **实际开始**: 2026-06-12 14:10
- **实际结束**: 2026-06-12 14:11
- **CUDA/torch 信息**: PyTorch 2.7.1+cu126, CUDA 12.6, 2x NVIDIA GeForce RTX 3090 Ti (24 GB each)
- **异常备注**: 无

---

## 任务 02 — Phase 1 dry-run

- **状态**: [x] 已完成
- **优先级**: P0
- **模型方法**: P11
- **场景**: mixed(S1+S2+S3)
- **目标**: 确认实验计划是统一 mixed training，再分别测试 S1/S2/S3。

### 执行命令

```powershell
cd D:\code\python\project-2\project_root

D:\envs\nfdkf-gpu\Scripts\python.exe -u scripts\run_phase1_nominal_benchmark.py `
  --dry-run `
  --methods main `
  --device cuda `
  --out-dir results\phase1_gpu_dryrun
```

### 验收标准

- [x] `[train_scene_set] S1+S2+S3`
- [x] `[test_scenes] S1, S2, S3`
- [x] `[methods] P11_feature_stable_reliability_no_cov`
- [x] `[train_eval_runs] 5`
- [x] `[rule_baseline_rows] 12`

### 执行记录（GPU端回填）

- **实际开始**: 2026-06-12 14:11
- **实际结束**: 2026-06-12 14:12
- **plan 路径**: `results/phase1_gpu_dryrun/phase1_plan.json`
- **异常备注**: 无

---

## 任务 03 — Phase 1 P11 smoke

- **状态**: [x] 已完成
- **优先级**: P0
- **模型方法**: P11
- **场景**: mixed(S1+S2+S3)
- **模型种子列表**: 0
- **训练种子范围**: smoke 默认
- **训练参数**: smoke 默认，epochs=2
- **特殊配置**: 必须 `--force-regenerate-sims`

### 执行命令

```powershell
cd D:\code\python\project-2\project_root

D:\envs\nfdkf-gpu\Scripts\python.exe -u scripts\run_phase1_nominal_benchmark.py `
  --smoke `
  --methods main `
  --device cuda `
  --force-regenerate-sims `
  --out-dir results\phase1_gpu_p11_smoke
```

### 验收标准

- [x] `results/phase1_gpu_p11_smoke/phase1_run_summary.csv` 存在
- [x] `results/phase1_gpu_p11_smoke/phase1_aggregate_by_scene.csv` 存在
- [x] S1/S2/S3 均有 P11 learned 行
- [x] S1/S2/S3 均有 rule baseline 行
- [x] 所有 `rmse/p95/p99/max` 为有限数值，无 NaN/Inf
- [x] `dataset_store/<本次mixed数据集>/meta.json` 存在

### 额外检查命令

```powershell
Import-Csv results\phase1_gpu_p11_smoke\phase1_aggregate_by_scene.csv |
  Select-Object scenario_id,method,method_category,rmse_mean,p95_mean,p99_mean,max_mean |
  Format-Table -AutoSize
```

### 执行记录（GPU端回填）

- **实际开始**: 2026-06-12 15:05
- **实际结束**: 2026-06-12 15:08
- **结果路径**: `project_root/results/phase1_gpu_p11_smoke/`
- **mixed dataset 路径**: `dataset_store/20260612_150503__phase1_s1_s2_s3_mixed_nominal__phase1_s1_s2_s3_mixed_n__21802638`
- **关键指标摘要**:
  | S1 P11 RMSE=3.58 p95=7.07 max=17.26
  | S2 P11 RMSE=2.95 p95=5.61 max=8.03
  | S3 P11 RMSE=5.99 p95=10.79 max=18.55
- **异常备注**: torch.compile 因 Triton 未安装而禁用（Windows 环境），不影响训练结果

---

## 任务 04 — Phase 1 P11 正式主实验

- **状态**: [ ] 待执行
- **优先级**: P0
- **模型方法**: P11
- **场景**: mixed(S1+S2+S3)
- **模型种子列表**: 0,1,2,3,4
- **训练种子范围**: 10-69(train) / 70-89(val) / 90-109(test)
- **训练参数**: epochs=80, lr=1e-3, batch_size=64
- **特殊配置**: 必须 `--force-regenerate-sims`

### 执行命令

```powershell
cd D:\code\python\project-2\project_root

D:\envs\nfdkf-gpu\Scripts\python.exe -u scripts\run_phase1_nominal_benchmark.py `
  --methods main `
  --device cuda `
  --force-regenerate-sims `
  --model-seeds 0,1,2,3,4 `
  --train-seed-range 10-69 `
  --val-seed-range 70-89 `
  --test-seed-range 90-109 `
  --epochs 80 `
  --out-dir results\phase1_gpu_p11_main
```

### 验收标准

- [ ] `phase1_run_summary.csv` 存在
- [ ] `phase1_aggregate_by_scene.csv` 存在
- [ ] `phase1_aggregate_overall.csv` 存在
- [ ] P11 learned 结果为 15 行：5 model seeds × 3 scenes
- [ ] rule baseline 结果为 12 行：4 baselines × 3 scenes
- [ ] S1 与 S2 指标不应完全相同；若完全相同，标记异常并停止
- [ ] 所有 `rmse/p95/p99/max` 为有限数值

### 执行记录（GPU端回填）

- **实际开始**:
- **实际结束**:
- **结果路径**: `project_root/results/phase1_gpu_p11_main/`
- **mixed dataset 路径**:
- **overall 指标摘要**:
- **by-scene 指标摘要**:
- **异常备注**:

---

## 任务 05 — Phase 1 P0/P1/P11 对比实验

- **状态**: [ ] 待执行
- **优先级**: P1
- **模型方法**: P0 / P1 / P11
- **场景**: mixed(S1+S2+S3)
- **模型种子列表**: 0,1,2,3,4
- **训练种子范围**: 10-69(train) / 70-89(val) / 90-109(test)
- **训练参数**: epochs=80, lr=1e-3, batch_size=64
- **前置条件**: 任务 04 已完成且无异常

### 执行命令

```powershell
cd D:\code\python\project-2\project_root

D:\envs\nfdkf-gpu\Scripts\python.exe -u scripts\run_phase1_nominal_benchmark.py `
  --methods compare `
  --device cuda `
  --force-regenerate-sims `
  --model-seeds 0,1,2,3,4 `
  --train-seed-range 10-69 `
  --val-seed-range 70-89 `
  --test-seed-range 90-109 `
  --epochs 80 `
  --out-dir results\phase1_gpu_compare
```

### 验收标准

- [ ] P0 learned 结果为 15 行
- [ ] P1 learned 结果为 15 行
- [ ] P11 learned 结果为 15 行
- [ ] rule baseline 结果为 12 行
- [ ] `phase1_aggregate_overall.csv` 能比较 AVG / CI-multi / WAA-MM / best-single / P0 / P1 / P11
- [ ] 所有 `rmse/p95/p99/max` 为有限数值

### 执行记录（GPU端回填）

- **实际开始**:
- **实际结束**:
- **结果路径**: `project_root/results/phase1_gpu_compare/`
- **mixed dataset 路径**:
- **overall 指标摘要**:
- **P11 相对 P0/P1/CI-multi 的简要结论**:
- **异常备注**:

---

## GPU 端完成后提交要求

GPU 端完成后只提交：

- 本文件的执行记录回填
- 可选：小型结果摘要文件，例如 `project_root/results_summary_phase1_gpu_20260612.md`

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
git add project_root\task_checklists\TASK_PHASE1_GPU_20260612.md
git commit -m "docs: update phase1 gpu execution checklist"
git push
```
