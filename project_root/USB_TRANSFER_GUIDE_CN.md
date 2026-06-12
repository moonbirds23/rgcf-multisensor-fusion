# 移动硬盘迁移指南

本指南用于把项目带到 GPU 电脑执行实验一。

## 核心原则

不要直接复制本机 venv。

Windows venv 内部包含绝对路径，例如：

```text
D:\code\python\env\env-NDKF - torch\Scripts\python.exe
```

复制到另一台电脑后经常因为路径、Python DLL、CUDA、torch wheel 不一致而失效。

更稳的方法是：

```text
移动硬盘 = 代码包 + Python安装器 + requirements + 一键脚本
GPU电脑 = 新建干净 venv + 安装 CUDA torch + 重新生成数据
```

## 移动硬盘中建议放置

```text
GPU_USB_KIT/
  project_root_gpu_migration_*.zip
  GPU_MIGRATION_RUNBOOK_CN.md
  USB_TRANSFER_GUIDE_CN.md
  requirements_gpu_base.txt
  scripts/
    install_gpu_env_online.ps1
    install_gpu_env_offline.ps1
    download_gpu_offline_wheels.ps1
    gpu_preflight_check.py
    run_gpu_phase1_smoke.ps1
    run_gpu_phase1_main.ps1
```

可选：

```text
python-3.10.x-amd64.exe
```

Python 安装器建议从 Python 官网下载 64-bit Windows installer。不要使用 32-bit Python。

如果 GPU 电脑联网慢，可以在本机提前下载 wheelhouse：

```powershell
cd D:\code\python\project-2\project_root

powershell -ExecutionPolicy Bypass -File scripts\download_gpu_offline_wheels.ps1 `
  -TorchIndexUrl "https://download.pytorch.org/whl/cu126"
```

或者使用 CUDA 12.8 wheel：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\download_gpu_offline_wheels.ps1 `
  -TorchIndexUrl "https://download.pytorch.org/whl/cu128"
```

下载完成后，移动硬盘包会多出：

```text
GPU_USB_KIT/
  installers/
  wheelhouse/
```

## GPU 电脑操作顺序

### 1. 安装 Python

安装 Python 3.10 x64。

安装时勾选：

```text
Add python.exe to PATH
pip
```

### 2. 解压项目

把：

```text
project_root_gpu_migration_*.zip
```

解压到：

```text
D:\code\python\project-2\project_root
```

或你自己的目录。后续命令都在 `project_root` 下执行。

### 3. 创建 GPU 环境，在线方式

```powershell
cd D:\code\python\project-2\project_root
powershell -ExecutionPolicy Bypass -File scripts\install_gpu_env_online.ps1
```

如果默认 CUDA 12.8 wheel 不适合你的显卡/驱动，可以指定 CUDA 12.6：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_gpu_env_online.ps1 `
  -TorchIndexUrl "https://download.pytorch.org/whl/cu126"
```

### 3b. 创建 GPU 环境，离线方式

如果移动硬盘里已经有 `wheelhouse/`：

```powershell
cd D:\code\python\project-2\project_root
powershell -ExecutionPolicy Bypass -File scripts\install_gpu_env_offline.ps1 `
  -Wheelhouse "E:\GPU_USB_KIT\wheelhouse"
```

如果你把 `wheelhouse` 复制到了项目根目录：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_gpu_env_offline.ps1 `
  -Wheelhouse "wheelhouse"
```

### 4. 预检

```powershell
D:\envs\nfdkf-gpu\Scripts\python.exe -u scripts\gpu_preflight_check.py --require-cuda
```

必须看到：

```text
[torch.cuda] True
[phase1_scenes] ok
[preflight] ok
```

### 5. 先跑 smoke

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_gpu_phase1_smoke.ps1
```

### 6. 再跑正式 P11

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_gpu_phase1_main.ps1
```

### 7. P0/P1/P11 对比

P11 正式主实验没问题后，再跑：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_gpu_phase1_main.ps1 `
  -Mode compare `
  -OutDir "results\phase1_gpu_compare"
```

## 不要复制/复用

不要从本机复制这些旧目录作为有效实验：

```text
dataset_store/
sim_cache/
results/phase1_nominal_benchmark_p11_main/
results/phase1_nominal_benchmark_compare/
```

它们属于修复前旧数据，已经判定 invalid。

## 如果 GPU 电脑没有网络

需要提前在有网络的电脑下载：

```text
Python 3.10 x64 installer
PyTorch CUDA wheel
requirements_gpu_base.txt 对应的 wheelhouse
```

使用：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\download_gpu_offline_wheels.ps1
```

PyTorch CUDA wheel 很大，而且必须匹配 GPU 电脑驱动支持的 CUDA wheel。如果不确定，优先准备 `cu126`；如果 GPU 很新或驱动很新，再准备 `cu128`。
