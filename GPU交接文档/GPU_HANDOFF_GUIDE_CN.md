# GPU 实验交接手册

**更新日期**: 2026-06-29
**项目**: RGCF 多传感器融合实验
**GitHub**: https://github.com/moonbirds23/rgcf-multisensor-fusion
**当前分支**: `codex/phase1r-rgcf-gpu`

---

## 1. 代码同步方式：GitHub

### 1.1 远程与本机代码同步

本机（开发端）与 GPU 执行端通过 GitHub 同步代码。开发端修改代码后 push，GPU 端 pull 获取最新代码。

```powershell
# GPU 端拉取代码
cd D:\code\python\project-2
git pull

# 如需切换到指定分支
git checkout codex/phase1r-rgcf-gpu
git pull
```

### 1.2 禁止提交的内容

以下目录和文件不要提交到 GitHub：

```text
project_root/dataset_store/   # 仿真数据集（太大）
project_root/sim_cache/       # 仿真缓存
project_root/results/         # 实验结果（太大）
*.pt                          # 模型权重
*.pkl                         # pickle 文件
*.zip                         # 压缩包
*.whl                         # Python 包
```

### 1.3 Git 提交规范

```powershell
git add <文件列表>
git commit -m "docs: <简短描述>"
git push
```

---

## 2. 实验结果交接方式：硬盘手动拷贝

### 2.1 结果目录结构

所有实验结果在 GPU 端生成后，需要通过移动硬盘/U 盘拷贝回开发端分析。

**GPU 端输出目录**：

| 实验 | 目录 | 说明 |
|------|------|------|
| Phase1R RGCF 正式实验 | `project_root\results\phase1r_rgcf_compare\` | RGCF vs 6 规则 baseline |
| Phase2 ME-RGCF-A0 | `project_root\results\phase2_me_a0_only\` | ME-RGCF-A0 独立实验 |
| Phase2 ME-RGCF-A0D | `project_root\results\phase2_me_a0_dir_only\` | ME-RGCF-A0D (EHGCF) 独立实验 |
| 论文消融实验 | `project_root\results\paper_ehgcf_final_compare\` | 论文最终对比（main + ablation） |
| 论文表格 | `project_root\results\paper_ehgcf_final_compare\paper_tables\` | 收集的论文表格 CSV |

**关键输出文件**（每个实验目录下）：

| 文件 | 说明 |
|------|------|
| `phase1r_aggregate_overall.csv` | 各方法 Overall 对比 |
| `phase1r_aggregate_by_scene.csv` | S1R/S2R 分场景对比 |
| `phase1r_run_summary.csv` | 所有 run 原始结果 |
| `phase1r_run_summary.json` | 所有 run 原始结果（JSON） |
| `phase1r_sensor_health_by_scene.csv` | 传感器健康度 |
| `phase1r_evidence_residual_report.csv` | Evidence 传感器残差报告 |
| `phase1r_plan.json` | 实验计划 |
| `eval_details\*_errors.csv` | 逐 step 评估明细（含权重、attention 等诊断字段） |

### 2.2 共享数据集

正式实验统一使用以下数据集（120 秒轨迹时长）：

```text
D:\code\python\project-2\project_root\dataset_store\20260615_210428__phase1r_s1r_s2r_mixed_nominal__phase1r_s1r_s2r_mixed_n__53259407
```

> 注意：所有 learned 方法对比必须在同一数据集上进行，否则 baseline RMSE 不同会导致无法公平比较。

### 2.3 拷贝操作

```text
GPU 端: D:\code\python\project-2\project_root\results\  -->  移动硬盘
移动硬盘  -->  开发端: D:\code\python\project-2\project_root\results\
```

---

## 3. GPU 硬件配置

### 3.1 本机规格

| 项目 | 规格 |
|------|------|
| GPU | NVIDIA GeForce RTX 3090 Ti (×2) |
| 显存 | 24 GB GDDR6X / 卡 |
| CUDA Compute Capability | 8.6 (Ampere) |
| CPU | Intel Core i9 24C/32T |
| 系统内存 | 64 GB DDR5 |
| Python venv | `D:\envs\nfdkf-gpu\` |
| PyTorch | 2.7.1+cu126 |
| CUDA | 12.6 |

### 3.2 训练参数

| 参数 | 默认值 | 说明 |
|------|:---:|------|
| batch_size | 64 | 24GB 显存充足，可升至 128-256 |
| hidden_dim | 64 | 模型极小 (~50K 参数) |
| num_workers | 2-4 | 过多反而增加 IPC 开销 |
| pin_memory | True | 系统内存充足 |
| torch.compile | 禁用 | Windows 端 Triton 不可用 |
| 混合精度 AMP | 不推荐 | Position error 对精度敏感 |

### 3.3 并行执行策略（双卡）

**本机有 2 张 RTX 3090 Ti，可通过手动分配 GPU 并行跑不同方法：**

```powershell
# GPU 0: 方法 A
$env:CUDA_VISIBLE_DEVICES='0'
python scripts/run_phase1r_rgcf_benchmark.py --methods <方法A> --mixed-dataset-dir <共享数据集路径> ...

# GPU 1: 方法 B（同时启动，另一个终端）
$env:CUDA_VISIBLE_DEVICES='1'
python scripts/run_phase1r_rgcf_benchmark.py --methods <方法B> --mixed-dataset-dir <共享数据集路径> ...
```

**并行条件**：
- 两个进程必须使用**同一份 mixed dataset**（`--mixed-dataset-dir` 指向相同路径）
- 两个进程输出到**不同目录**（`--out-dir` 不同）
- 不要使用 `DataParallel`（模型太小，通信开销 > 计算收益）

**时间对比**：

| 方式 | 10 个训练任务 | 
|------|:---:|
| 串行 | ~10 小时 |
| 双卡并行 | ~5 小时 |

### 3.4 环境变量

```powershell
$env:RGCF_SIM_WORKERS='4'    # 仿真并行进程数
$env:RGCF_NUM_WORKERS='2'    # DataLoader 进程数
$env:RGCF_DISABLE_COMPILE='1' # Windows 端禁用 torch.compile（Triton 不可用）
```

---

## 4. 远程开发端规范

### 4.1 代码编写约束

远端（开发端）编写代码时，必须遵循本机 GPU 配置：

- **batch_size 默认 64**，不硬编码为超出 24GB 显存的值
- **模型 hidden_dim** 保持 64（当前 ~50K 参数，远未触及显存上限）
- **不使用 torch.compile**：Windows 无 Triton，代码中已做 try/except 兼容处理
- **不使用 float64**：3090 Ti FP64 被阉割（FP32 的 1/64）
- **不使用 DataParallel/DDP**：模型太小
- **不依赖 AMP 混合精度**：position error 对精度敏感
- **pin_memory=True**：本机内存 64GB 足够

### 4.2 执行方式

远端编写实验脚本后，通过 GitHub push，GPU 端 pull 后按本手册执行。

---

## 5. 本机并行方案（推荐）

### 5.1 单实验并行

当需要跑多个 learned 方法时（如 RGCF + ME-A0 + ME-A0D），使用双卡并行：

```powershell
# 终端 1（GPU 0）
$env:CUDA_VISIBLE_DEVICES='0'
python scripts/run_phase1r_rgcf_benchmark.py --methods posterior-only --mixed-dataset-dir "D:\...\53259407" --out-dir "results\posterior_only" ...

# 终端 2（GPU 1）
$env:CUDA_VISIBLE_DEVICES='1'
python scripts/run_phase1r_rgcf_benchmark.py --methods ehgcf-no-calib --mixed-dataset-dir "D:\...\53259407" --out-dir "results\ehgcf_no_calib" ...
```

### 5.2 论文统一脚本

推荐使用论文实验统一脚本 `run_paper_ehgcf_experiments.py`，一条命令运行完整的 main + ablation：

```powershell
python scripts/run_paper_ehgcf_experiments.py --mode all --device cuda --resume --out-root "results\paper_ehgcf_final_compare" --mixed-dataset-dir "D:\...\53259407"
```

该脚本会自动：
- 复用共享数据集
- `--resume` 跳过已完成的方法
- 分阶段输出到 `main\` 和 `ablation\` 子目录

### 5.3 结果收集

```powershell
python scripts/collect_paper_ehgcf_results.py
```

生成论文表格到 `paper_ehgcf_final_compare\paper_tables\`。

---

## 6. 本文件同步

本文件位于项目仓库内，需要随代码一同提交到 GitHub：

```powershell
cd D:\code\python\project-2
git add GPU交接文档\GPU_HANDOFF_GUIDE_CN.md
git commit -m "docs: add GPU handoff guide"
git push
```

---

## 附录：常用命令速查

### 环境检查

```powershell
cd D:\code\python\project-2\project_root
D:\envs\nfdkf-gpu\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.__version__)"
```

### 实验 dry-run

```powershell
python scripts/run_phase1r_rgcf_benchmark.py --dry-run --methods rgcf --device cuda
```

### 实验 smoke test

```powershell
$env:RGCF_DISABLE_COMPILE='1'
python scripts/run_phase1r_rgcf_benchmark.py --smoke --device cuda --methods <method> --epochs 2 --model-seeds 0
```

### 正式实验

```powershell
$env:RGCF_SIM_WORKERS='4'
$env:RGCF_NUM_WORKERS='2'
$env:RGCF_DISABLE_COMPILE='1'
python scripts/run_phase1r_rgcf_benchmark.py --device cuda --methods <method> --model-seeds 0,1,2,3,4 --train-seed-range 10-69 --val-seed-range 70-89 --test-seed-range 90-109 --epochs 80 --mixed-dataset-dir "D:\...\53259407" --out-dir "results\<output_dir>"
```

### 论文完整实验

```powershell
python scripts/run_paper_ehgcf_experiments.py --mode all --device cuda --resume --out-root "results\paper_ehgcf_final_compare" --mixed-dataset-dir "D:\...\53259407"
```
