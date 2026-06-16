# 历史方案归档说明

本目录保存已经退出当前默认主线的历史方案、实验改造计划、旧 GPU 交接材料和旧架构示意。

这些文件的用途是：

- 追溯问题来源；
- 复现实验背景；
- 对比旧设计为什么失败或被替换；
- 为论文讨论提供历史诊断证据。

这些文件不再作为当前实现、训练或 GPU 实验的默认依据。

## 当前主线

请优先阅读：

```text
../CURRENT_MAINLINE_CN.md
../PHASE1R_RGCF_GPU_EXPERIMENT_PLAN_CN.md
../ME_RGCF_HETEROGENEOUS_TEMPORAL_DESIGN_CN.md
../../PROJECT.md
../../../README.md
```

当前主线一句话：

```text
Phase1R RGCF 是稳定 clean baseline；ME-RGCF-A0 是可选的异构图最小升级。
```

## 已归档方向

以下方向只保留为历史记录：

- P0 restart / DS-RCAA 方案；
- P1 dual-stream direct 方案；
- P11/P12 gate/covariance 迭代方案；
- M4 / RGCF V5 peer-consistency 消融方案；
- SNF-A / Skeptical Neural Fusion 早期构想；
- 旧 GPU 迁移手册、旧 USB 传输手册、旧上传清单；
- 旧任务模板与旧 Phase1 GPU 执行记录。

## 使用规则

- 不要把本目录文档中的模块名直接当作当前模型名。
- 不要把旧文档中的默认命令复制到新实验中。
- 如果需要复用旧结论，必须标注为“历史诊断证据”。
- 新实验默认入口是 `scripts/run_phase1r_rgcf_benchmark.py`。
- 新实验默认 learned 方法是 `RGCF`；`ME-RGCF-A0` 需要显式通过 `--methods` 启用。

## 代码留存规则

旧代码不做物理删除，除非确认没有任何复现实验、preset 或结果分析依赖它。

当前对旧代码的处理方式是：

```text
保留代码能力
归档旧文档思想
在当前 README/PROJECT 中明确默认路径
```

这样可以避免后续窗口被旧方案牵引，同时保留必要的可复现性。
