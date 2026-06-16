# M4 当前版本记录归档

更新日期：2026-06-09

本文件原本用于说明 `M4_V5_peer_consistency` 是当时的 RGCF V5 当前主版本。
该判断现在已经归档，不再作为项目默认结论。

新的当前实验设计见：

```text
project_root/EXPERIMENT_REDESIGN_CURRENT_CN.md
```

## 归档原因

后续 P0-P12 多轮实验和 full-tail diagnostics 表明，M4 虽然相对 M3 降低了
部分 `bias_ramp` 灾难性融合错误，但它并没有稳定解决窗口内慢漂移和窗口后
残留风险。继续把 M4 当作当前主版本，会让论文主线被单一版本树和单一故障
类型牵制。

## M4 仍然保留的价值

- M4 是发现 `bias_ramp` tail-risk 问题的重要诊断版本。
- M4 证明 peer consistency 能改善一部分慢漂移故障下的错误高权重问题。
- M4 的失败点说明 NIS、covariance 和单时刻 gate 不足以稳定覆盖慢漂移隐性故障。
- M4 的 max-error diagnosis 可作为第三层 Hidden Drift & Recovery Stress Test 的动机材料。

## 不再沿用的判断

- 不再认为 M4 是已商定的当前主版本。
- 不再把 `bias_ramp` 和 dropout/pollution 平铺为同级主 benchmark 任务。
- 不再围绕 M4A 系列无限扩展作为论文主线。
- 不再把 covariance-aware fusion 视为默认主方法。
- 不再只用 M4 相对 M3 的提升来支撑最终论文故事。

## 后续引用方式

如果需要引用 M4，请把它称为：

> early peer-consistency diagnostic variant

而不是：

> current main version

后续新增结果应优先纳入三层实验框架：

1. Nominal Fusion Benchmark
2. Observable Degradation Benchmark
3. Hidden Drift & Recovery Stress Test
