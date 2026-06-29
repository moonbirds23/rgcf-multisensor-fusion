# EHGCF GPU Experiment Runbook

本文档用于把本机已经整理好的 EHGCF 论文补充实验发送到 GPU 端运行。最终论文方法名统一为 `EHGCF`，代码内部最终实现对应 `ME-RGCF-A0D`。

## 1. 实验目标

主实验比较：

- `EHGCF`
- `CI-3T`
- `WAA-MM-3T`
- `AVG-3T`
- `single-T1`
- `single-T2`
- `single-T3`

精简消融比较：

- `Posterior-only`
- `RGCF`
- `ME-RGCF-A0`
- `EHGCF w/o calibrated fusion`
- `EHGCF`

注意：`RGCF`、`ME-RGCF-A0`、`ME-RGCF-A0D-HS` 只能作为过程版本、对照组或消融组，不作为论文创新路线叙述。

## 2. 环境准备

在 GPU 端进入项目上级目录，例如：

```powershell
cd D:\code\python\project-2
```

建议使用项目原有 Python 环境：

```powershell
& "D:\code\python\env\env-NDKF - torch\Scripts\python.exe" -c "import torch; print(torch.cuda.is_available()); print(torch.__version__)"
```

若输出 `True`，再继续正式实验。

## 3. 先检查命令

只打印命令，不启动训练：

```powershell
& "D:\code\python\env\env-NDKF - torch\Scripts\python.exe" project_root\scripts\run_paper_ehgcf_experiments.py --dry-run --mode all --device cuda
```

默认输出根目录：

```text
E:\migration_packages\results\paper_ehgcf_final_compare
```

## 4. 正式运行

推荐一次性运行主实验和消融实验：

```powershell
& "D:\code\python\env\env-NDKF - torch\Scripts\python.exe" project_root\scripts\run_paper_ehgcf_experiments.py --mode all --device cuda --resume
```

若 GPU 端已有固定 mixed dataset，可显式传入：

```powershell
& "D:\code\python\env\env-NDKF - torch\Scripts\python.exe" project_root\scripts\run_paper_ehgcf_experiments.py --mode all --device cuda --resume --mixed-dataset-dir "D:\code\python\project-2\project_root\dataset_store\<DATASET_DIR_NAME>"
```

如果没有传 `--mixed-dataset-dir`，脚本会先按固定 seeds 生成 mixed dataset；`--mode all` 的消融阶段会复用主实验阶段生成的数据集。

## 5. 快速 smoke 检查

正式跑之前可以用短流程确认脚本、CUDA、数据生成和模型构造都正常：

```powershell
& "D:\code\python\env\env-NDKF - torch\Scripts\python.exe" project_root\scripts\run_paper_ehgcf_experiments.py --mode all --device cuda --smoke --epochs 2 --model-seeds 0 --resume --out-root "E:\migration_packages\results\paper_ehgcf_smoke_check"
```

Smoke 结果不能写入论文，只用于检查流程。

## 6. 结果整理

训练完成后运行：

```powershell
& "D:\code\python\env\env-NDKF - torch\Scripts\python.exe" project_root\scripts\collect_paper_ehgcf_results.py
```

默认生成：

```text
E:\migration_packages\results\paper_ehgcf_final_compare\paper_tables\table_main_overall.csv
E:\migration_packages\results\paper_ehgcf_final_compare\paper_tables\table_by_scene.csv
E:\migration_packages\results\paper_ehgcf_final_compare\paper_tables\table_ablation.csv
E:\migration_packages\results\paper_ehgcf_final_compare\paper_tables\mechanism_plot_pool.csv
E:\migration_packages\results\paper_ehgcf_final_compare\paper_tables\result_sources_manifest.csv
```

`mechanism_plot_pool.csv` 会保留误差序列、权重、gate、covariance scale、M-P attention、M-M attention、pair residual/rank 等可用列，用于后续实验图像生成。

## 7. 需要回传的文件

请回传整个目录：

```text
E:\migration_packages\results\paper_ehgcf_final_compare
```

至少需要包含：

- `main\phase1r_aggregate_overall.csv`
- `main\phase1r_aggregate_by_scene.csv`
- `main\phase1r_run_summary.csv`
- `ablation\phase1r_aggregate_overall.csv`
- `ablation\phase1r_aggregate_by_scene.csv`
- `ablation\phase1r_run_summary.csv`
- `paper_tables\*.csv`
- `main\eval_details\*_errors.csv`
- `ablation\eval_details\*_errors.csv`

## 8. 常见问题

- 如果提示未知方法名，确认代码已包含 `posterior-only`、`ehgcf`、`ehgcf-no-calib`。
- 如果显存不足，先降低 `--batch-size 32`，保持 seeds 和 epochs 不变。
- 如果中断，继续使用 `--resume`。
- 如果 mixed dataset 路径不存在，去掉 `--mixed-dataset-dir`，让脚本按固定 seeds 重建。
