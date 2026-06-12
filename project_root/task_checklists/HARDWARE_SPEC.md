# 本机硬件规格 — GPU 实验基准

**记录日期**: 2026-06-12
**机器角色**: 开发端 / GPU 执行端（双用途）

---

## GPU

| 项目 | 规格 |
|------|------|
| GPU 型号 | **NVIDIA GeForce RTX 3090 Ti** (×2) |
| 显存 / 卡 | **24 GB GDDR6X** |
| CUDA Compute Capability | **8.6** (Ampere) |
| 驱动版本 | 32.0.15.9186 (R591) |
| Tensor Cores | 第 3 代 (Ampere) |
| FP32 理论算力 | ~40 TFLOPS / 卡 |
| FP16 理论算力 | ~80 TFLOPS / 卡 (with Tensor Cores) |
| 推荐 CUDA Toolkit | 11.x / 12.x |
| 推荐 PyTorch CUDA | cu118 / cu121 / cu124 / cu126 / cu128 |

### GPU 环境路径

```text
Python venv:  D:\envs\nfdkf-gpu\
Python 版本:  3.10 x64
PyTorch wheel: 按驱动匹配（推荐 CUDA 12.6/12.8）
```

---

## CPU

| 项目 | 规格 |
|------|------|
| CPU 型号 | Intel Core i9 (13th/14th Gen) with UHD Graphics 770 |
| 物理核心 / 逻辑核心 | 24 Cores / 32 Threads |
| 基频 | ~3.0 GHz (max boost ~5.8 GHz) |

---

## 内存

| 项目 | 规格 |
|------|------|
| 系统内存 | **64 GB** DDR5 |

---

## 存储

| 项目 | 路径 |
|------|------|
| 项目根目录 | `D:\code\python\project-2\` |
| Python 虚拟环境 | `D:\envs\nfdkf-gpu\` |
| 数据集存储 | `project_root\dataset_store\` |
| 仿真缓存 | `project_root\sim_cache\` |
| 实验结果 | `project_root\results\` |

---

## GPU 编程约束（供代码优化参考）

### 必须遵守的配置

- **batch_size**: 默认 64，显存充足时可提升至 128 或 256
- **模型 hidden_dim**: 默认 64，可安全升至 128（当前模型极小，远未触及 24 GB 显存上限）
- **num_workers**: 推荐 2-4（数据量较小，过多 worker 反而增加 IPC 开销）
- **pin_memory**: 必须 `True`（系统内存充足）
- **torch.compile**: 推荐 `mode="reduce-overhead"`（适合小 batch 高频调用的 GNN）
- **混合精度**: 当前模型精度要求高，暂不推荐 AMP（Position error 对精度敏感）

### 双卡使用策略

- 当前 Phase 1 单次训练模型极小（~50K 参数），无需模型并行
- 如需并行跑多个 model seed，可手动分配 `CUDA_VISIBLE_DEVICES=0` / `=1`
- 不建议用 `DataParallel`（小模型通信开销 > 计算收益）
- 如需全量 grid search，建议用 shell 脚本分别在不同 GPU 上启动独立进程

### 避免

- 不要在 3090 Ti 上使用 `torch.backends.cudnn.benchmark = True`（当前代码已设为 `False`，小模型 + 固定 input shape 无收益）
- 不要使用 `float64` 训练（3090 Ti 的 FP64 被严重阉割，仅为 FP32 的 1/64）
- 不要使用 CUDA Graphs（图节点数 4，graph capture overhead > 收益）

---

## 其他电脑适配说明

如果后续在其他 GPU 电脑上运行本项目，请以此文件为模板，记录对应机器的硬件规格。
代码中的 GPU 优化参数（batch_size, num_workers, pin_memory）应依据实际硬件调整：

| 硬件条件 | batch_size | num_workers | 备注 |
|----------|:----------:|:-----------:|------|
| RTX 3090/4090 (24 GB) | 64-256 | 2-4 | 当前基准配置 |
| RTX 3060/4060 (8-12 GB) | 64-128 | 2 | 显存足够，计算稍慢 |
| RTX 2060/3060 Mobile (6 GB) | 32-64 | 1-2 | 降低 batch_size |
| GTX 1060/1660 (6 GB) | 32 | 1 | 无 Tensor Core，训练慢约 3-5x |
| 纯 CPU (无 GPU) | 16-32 | 0 | 预计训练时间 10-20x |
