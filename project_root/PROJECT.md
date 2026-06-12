# RGCF 实验项目 — 运行与维护规范

更新日期：2026-06-12

## 项目定位

本项目为 **RGCF（Robust Generalized Cross-modal Fusion）** 实验代码的**执行环境**。
项目的核心任务是**运行实验**，不在此进行任何代码的功能性修改或开发。

### 核心原则

- **只执行，不开发。** 所有代码变更（算法、模型、配置逻辑等）应在上游仓库完成，
  通过 `git pull` 同步到本机，不在本机直接修改任何 `.py` 源文件。
- **允许的例外（维护类操作）：**
  - 修正因环境差异导致的路径/导入/依赖兼容性问题
  - 更新 `.md` 文档、操作手册、runbook
  - 调整本地配置文件（`configs/` 中的参数值，而非配置逻辑）
  - 修复因 Python/依赖版本差异引起的语法或 API 兼容性问题
- **禁止的操作：**
  - 修改 `core/`、`models/`、`simulation/`、`training/`、`features/` 中的
    算法逻辑或模型结构
  - 在 `experiments/` 中新增实验场景（除非已有上游对应代码）
  - 修改 `configs/` 中的配置加载逻辑（`base_config.py`、`config_loader.py` 等）

## 项目结构

```
project_root/
├── configs/          # 实验配置（参数值、场景定义、预设）
├── core/             # 核心基础设施（配置加载、IO、注册表、结果管理）
├── data/             # 数据存储与缓存
├── experiments/      # 实验入口脚本
├── features/         # 特征工程
├── models/           # 模型定义（GNN 融合、融合基类）
├── rgcf_figures/     # 论文图表绘制
├── results/          # 实验结果输出
├── scripts/          # 批量运行、诊断、绘图脚本
├── simulation/       # 仿真引擎（故障注入、传感器、EKF、轨迹）
├── training/         # 训练器、评估器、损失函数
├── requirements_gpu_base.txt
├── PROJECT.md        # 本文件
├── EXPERIMENT_REDESIGN_CURRENT_CN.md  # 实验设计重构方案（最新）
├── GPU_MIGRATION_RUNBOOK_CN.md        # GPU 环境迁移手册
└── USB_TRANSFER_GUIDE_CN.md           # USB 数据传输指南
```

## Git 远程仓库与代理配置

> **代理已配置 ✓ | 远程仓库待配置**

### 当前代理配置

```bash
# Git 代理（已生效）
http.proxy=http://127.0.0.1:33210
https.proxy=http://127.0.0.1:33210

# Shell 环境变量（每次启动终端需手动设置，或写入 ~/.bashrc）
export http_proxy=http://127.0.0.1:33210
export https_proxy=http://127.0.0.1:33210
export all_proxy=socks5://127.0.0.1:33211
```

### 配置远程仓库

```bash
# 设置上游仓库地址（替换为实际 URL）
git remote add origin <YOUR_GITHUB_REPO_URL>

# 确认远程配置
git remote -v

# 首次拉取
git pull origin master
```

### 备选方案：gitclone.com 镜像加速

如果代理不可用，可以用 gitclone.com 作为拉取镜像：

```bash
# 方式一：直接通过镜像 clone（首次）
git clone https://gitclone.com/github.com/<USER>/<REPO>.git

# 方式二：已有本地仓库，通过镜像 pull
git remote set-url origin https://gitclone.com/github.com/<USER>/<REPO>.git

# 注意：镜像只适合 pull，不适合 push。如需 push 仍需代理访问 GitHub。
```

## 每次运行实验前的标准流程

**必须按以下顺序执行，不可跳过步骤：**

### 第一步：同步最新代码

```bash
# 1. 进入项目目录
cd D:\code\python\project-2\project_root

# 2. 拉取最新代码
git pull origin main   # 或 master，取决于默认分支名

# 3. 确认无本地未提交的意外修改
git status

# 4. 如果 git status 显示有意外的本地修改：
#    - 维护类修改：确认是否已在上游同步，如不需要保留则 git stash / git restore
#    - 非维护类修改：立即 git stash 或 git restore，不在本机保留
```

### 第二步：检查依赖

```bash
# 确认依赖完整
pip install -r requirements_gpu_base.txt

# 可选：运行 GPU 预检脚本
python scripts/gpu_preflight_check.py
```

### 第三步：确认当前实验目标

查阅 `EXPERIMENT_REDESIGN_CURRENT_CN.md`（当前实验设计重构方案），确认：
- 当前 Phase（Phase 1 / 2 / 3）
- 当前主模型版本（默认 P11）
- 当前对比池（P0 / P1 / P11）

### 第四步：运行实验

根据实验目标选择对应的入口脚本，例如：

```bash
# Phase 1 基准测试
python scripts/run_phase1_nominal_benchmark.py

# RGCF 消融实验
python scripts/run_rgcf_v5_ablation.py

# 论文图表生成
python scripts/run_rgcf_figures.py
```

### 第五步：记录结果

- 实验结果默认输出到 `results/` 目录
- 图表输出到 `rgcf_figures/` 目录
- 在 `results/` 中按日期/实验名建立子目录，记录运行参数与输出摘要

## 实验常用命令速查

| 脚本 | 用途 |
|------|------|
| `scripts/run_phase1_nominal_benchmark.py` | Phase 1 名义基准测试 |
| `scripts/run_rgcf_v5_ablation.py` | RGCF V5 消融实验 |
| `scripts/run_rgcf_figures.py` | 论文图表批量生成 |
| `scripts/gpu_preflight_check.py` | GPU 环境预检 |
| `scripts/warmup_sim_cache.py` | 仿真缓存预热 |
| `scripts/plot_rgcf_paper_figures.py` | RGCF 论文图表绘制 |
| `scripts/collect_rgcf_main_ablation.py` | 消融结果汇总 |
| `scripts/diagnose_rgcf_v5_calibration.py` | V5 校准诊断 |

## 当前实验状态

- **当前日期：** 2026-06-12
- **活跃 Phase：** Phase 1（S1/S2/S3 mixed training set）
- **主模型版本：** P11（`feature_stable_reliability_no_cov`）
- **对比池：** P0 / P1 / P11
- **详细方案：** 参见 [EXPERIMENT_REDESIGN_CURRENT_CN.md](EXPERIMENT_REDESIGN_CURRENT_CN.md)

## 环境要求

- Python 3.x（参见 `requirements_gpu_base.txt`）
- CUDA GPU（参见 [GPU_MIGRATION_RUNBOOK_CN.md](GPU_MIGRATION_RUNBOOK_CN.md)）
- 操作系统：Windows（详见迁移手册中的环境配置说明）

## 注意事项

1. **不要在本机直接提交代码。** 如需修改代码，在上游仓库完成并通过 PR 合并后，
   再在本机 `git pull`。
2. **每次运行前必须执行 `git pull` + `git status` 检查。** 在有未拉取的远端
   更新时运行实验，等于在旧代码上做无用功。
3. **实验结果（`results/`、`rgcf_figures/` 输出）建议加入 `.gitignore`**，
   避免把大体积输出文件推送到仓库。
4. **windows 环境上的 `configs/` 路径配置可能需要与上游 Linux 配置分开维护。**
   如果上游更新了配置逻辑（而非参数值），优先保留本机兼容性修改，并在 git 中
   使用 `git stash` 管理本地差异。
