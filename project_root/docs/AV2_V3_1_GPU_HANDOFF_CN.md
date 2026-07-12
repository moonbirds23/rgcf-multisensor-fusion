# AV2 V3.1 GPU 交接说明

## 范围

本交接包只用于 V3.1 的 Level C CUDA 冒烟阶段：30--50 个 nominal-only 场景、measurement seed 100、model seed 0、1--3 epoch。它不执行 1000 场景正式实验，也不生成任何 bias、dropout、delay 或其他 fault 数据。

## 代码同步

先在 GPU 端更新 Git 工作区。若当前分支没有本次 AV2 V3.1 文件，将压缩包中的 `SOURCE_SNAPSHOT` 覆盖到 `project_root` 同名目录；不要覆盖 GPU 端的原始数据目录、缓存、checkpoint 或结果目录。

## 环境

优先复用现有 GPU Python 环境。Python 3.10 x64 是推荐版本。必须满足：

- CUDA 可用的 PyTorch；`torch.cuda.is_available()` 必须为 `True`；
- `av2==0.2.1`、`pyarrow==17.0.0`；
- `numpy`、`scipy`、`matplotlib`、`pandas`、`scikit-learn`、`opencv-python`。

如现有环境缺少非 PyTorch 包，执行：

```powershell
$py = 'D:\envs\pefnet-av2-gpu\Scripts\python.exe'
& $py -m pip install -r requirements_av2_level_c_gpu.txt
```

如 CUDA PyTorch 缺失或不可用，先由 GPU 端本地 AI 根据 `nvidia-smi` 和当前驱动选择匹配的 PyTorch CUDA wheel；不得安装 CPU wheel，也不得让脚本回退到 CPU。

## 执行

确认移动硬盘挂载后，从 `project_root` 执行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_av2_nominal_level_c.ps1 `
  -PythonExe 'D:\envs\pefnet-av2-gpu\Scripts\python.exe' `
  -DataRoot 'E:\migration_packages\PEFNet_AV2\small_scene_feasibility_v1' `
  -SceneCount 30 -Epochs 1 -BatchSize 128
```

脚本顺序固定：GPU 预检 -> 嵌套 30 场景 nominal cache -> CUDA smoke。任何一步失败均停止，不允许跳过。

## 验收产物

均位于 `DataRoot`：

- `metadata/av2_level_c_nominal_prepare.json`
- `logs/av2_gpu_smoke.log`
- `checkpoints/av2_smoke_seed0.pt`
- `reports/AV2_LEVEL_C_GPU_SMOKE_REPORT.md`

只有报告为 `GO`，才能开始 1000 场景 F1--F11 实现与运行。
