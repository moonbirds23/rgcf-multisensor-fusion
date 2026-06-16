# GPU 电脑迁移与实验一执行手册

更新日期：2026-06-11

本文档用于在没有本地 AI 辅助的 GPU 电脑上稳定复现实验一。当前原则是：

1. 先在本机修复并提交代码。
2. GPU 电脑只拉取代码，不复用旧 `dataset_store/`、`sim_cache/`、错误结果目录。
3. 在 GPU 电脑上重新生成 mixed dataset，重新训练，重新评估。

## 1. 迁移方式

推荐使用 GitHub。

原因：

- 代码、文档、脚本能保持版本一致。
- GPU 电脑没有本地 AI 时，最怕手工复制漏文件；GitHub 可以避免漏改。
- 后续实验结果可以按目录回传，代码可以继续用 commit 对齐。

不推荐只用 U 盘复制整个工程目录，因为当前本机有大量旧结果、旧缓存和错误数据：

```text
project_root/dataset_store/
project_root/sim_cache/
project_root/results/
```

这些目录不要作为有效实验迁移。

## 2. 本机迁移前检查

在本机完成代码修改后，运行：

```powershell
cd D:\code\python\project-2

& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' -m compileall -q project_root

cd project_root

& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' -u scripts\gpu_preflight_check.py

& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' -u scripts\run_phase1_nominal_benchmark.py `
  --dry-run `
  --methods main `
  --device cuda
```

预期：

```text
[phase1_scenes] ok
[preflight] ok
[train_scene_set] S1+S2+S3
[test_scenes] S1, S2, S3
[methods] P11_feature_stable_reliability_no_cov
```

## 3. GPU 电脑 Python 环境

推荐新建干净环境，不要把 Anaconda base 或旧 AI 环境混在一起。

推荐版本：

```text
Python 3.10 x64
PyTorch CUDA wheel
```

如果 GPU 很新，优先使用 PyTorch 官网当前 selector 给出的 CUDA wheel。当前 PyTorch 官方 previous versions 页面也说明旧版本可用，但更推荐安装最新版本；当前页面列出的新版本 wheel 支持 CUDA 12.6、12.8、13.0 等 index-url。

### 3.1 创建 venv

```powershell
py -3.10 -m venv D:\envs\nfdkf-gpu
D:\envs\nfdkf-gpu\Scripts\activate
python -m pip install --upgrade pip
```

### 3.2 安装 PyTorch

优先去 PyTorch 官网选择与你显卡驱动匹配的 CUDA 版本。

常用示例，二选一，不要同时执行：

```powershell
# CUDA 12.8 wheel 示例
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

```powershell
# CUDA 12.6 wheel 示例
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

如果这两条都不适合你的显卡/驱动，以官网 selector 为准。

### 3.3 安装项目基础依赖

```powershell
pip install -r requirements_gpu_base.txt
```

不要安装本机 `pip freeze` 里的 TensorFlow 包；它们不是当前 Phase 1 主实验需要的依赖。

### 3.4 CUDA 检查

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

必须看到：

```text
True
```

## 4. GPU 电脑拉取项目

推荐：

```powershell
cd D:\code\python
git clone <你的 GitHub 仓库地址> project-2
cd D:\code\python\project-2\project_root
```

如果仓库很大，不要上传这些目录：

```text
project_root/dataset_store/
project_root/sim_cache/
project_root/results/
```

如果必须用压缩包迁移，只打包代码和文档目录，不打包旧数据：

```text
configs/
core/
data/
experiments/
features/
models/
simulation/
training/
scripts/
rgcf_figures/
requirements_gpu_base.txt
GPU_MIGRATION_RUNBOOK_CN.md
EXPERIMENT_REDESIGN_CURRENT_CN.md
README.md
```

## 5. GPU 电脑预检

```powershell
cd D:\code\python\project-2\project_root
D:\envs\nfdkf-gpu\Scripts\activate

python -u scripts\gpu_preflight_check.py --require-cuda
```

必须确认：

```text
S1 types=['gps2d', 'radar_rb', 'aoa_only', 'uwb_range_only']
S2 pos=[(-850.0, -350.0), (-350.0, -250.0), (-700.0, 450.0), (-150.0, 350.0)]
S3 motion init_v=20.0
[preflight] ok
```

## 6. 实验一执行顺序

### 6.1 dry-run

```powershell
python -u scripts\run_phase1_nominal_benchmark.py `
  --dry-run `
  --methods main `
  --device cuda
```

### 6.2 P11 smoke

必须强制重新生成仿真：

```powershell
python -u scripts\run_phase1_nominal_benchmark.py `
  --smoke `
  --methods main `
  --device cuda `
  --force-regenerate-sims `
  --out-dir results\phase1_gpu_p11_smoke
```

### 6.3 P11 正式主实验

```powershell
python -u scripts\run_phase1_nominal_benchmark.py `
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

### 6.4 P0/P1/P11 对比

P11 主实验确认无误后再跑：

```powershell
python -u scripts\run_phase1_nominal_benchmark.py `
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

## 7. 没有本地 AI 时的执行策略

严格按以下顺序，不要跳步：

1. `git pull`
2. 激活环境
3. `python -u scripts\gpu_preflight_check.py --require-cuda`
4. `--dry-run`
5. `--smoke --methods main --force-regenerate-sims`
6. 检查 `phase1_run_summary.csv`
7. 跑 P11 formal
8. 跑 compare formal
9. 把 `results/phase1_gpu_*` 和对应 `dataset_store/<本次mixed数据集>` 复制回本机分析

如果任何一步报错，停止，不要继续跑正式实验。

## 8. 明确不要做

不要使用：

```text
--resume
--mixed-dataset-dir 旧 dataset
旧 sim_cache
旧 phase1_nominal_benchmark_p11_main
旧 phase1_nominal_benchmark_compare
```

当前旧结果已经被判定为 invalid，只能作为排查记录。
