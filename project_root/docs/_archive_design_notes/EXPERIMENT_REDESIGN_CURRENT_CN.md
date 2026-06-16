# 当前实验设计重构方案：分层融合评估框架

更新日期：2026-06-09

本文档替代此前以 `M4_V5_peer_consistency`、`P0-P12` 或单一 `bias_ramp`
修补为中心的实验叙事。旧实验结果仍然保留为诊断证据，但不再作为论文主线
的默认约束。

## 0. 当前统一口径（2026-06-09 更新）

- Phase 1 改为一个 `S1/S2/S3 mixed training set`：同一方法、同一 model seed
  只训练一次，再分别测试 S1、S2、S3。
- 当前主版本优先选择 `P11_feature_stable_reliability_no_cov`。原因是它保留
  post-stream、meas-stream 与 reliability gate 的中等复杂度，同时移除了此前
  容易放大 tail risk 的 covariance path 和 measurement-representation shortcut。
- Phase 1 默认对比池收敛为 P0/P1/P11：P0 是 post-only learned baseline，
  P1 是 dual-stream direct baseline，P11 是当前主模型。
- `P12_recovery_aware_full_gate_no_cov` 暂不作为 Phase 1 默认主版本；它更适合
  放在 Phase 3 的窗口恢复、hidden drift 与 recovery-aware stress test 中验证。
- P4/P12 仍保留为扩展消融和诊断证据，但不再约束当前主实验叙事。

## 1. 当前核心判断

已有多轮实验表明，当前模型并不是整体融合能力不足。去掉慢漂移类污染后，
模型在 clean、dropout、pollution 等可观测退化场景中已经能稳定完成融合。

真正反复造成灾难性 tail error 的，是一类更特殊的故障：

> 慢漂移、局部滤波器自洽、跨传感器不一致、窗口后可能残留的隐性 drift
> 风险。

因此，后续论文故事不再定义为“调好四种手工污染”，而是定义为：

> 面向异构多传感器跟踪的可靠性引导融合，并通过分层 benchmark 评估其在
> 正常融合、可观测退化、窗口恢复与隐性漂移压力测试中的行为。

## 2. 保留与抛弃

### 2.1 明确保留

- 保留 `post-stream + meas-stream + reliability gate` 作为中等复杂度主模型族。
- 保留 P0/P1/P4/P11/P12 等已有 variant 作为消融候选和历史证据。
- 保留 full-tail diagnostics，包括 p95、p99、max、fault sensor weight、
  fault top-rate、per-mode top max。
- 保留 `bias_ramp` 相关结果，但它不再和 dropout/pollution 平铺为同级主任务。
- 保留窗口恢复问题，将其升级为第三层压力测试主题。

### 2.2 明确抛弃

- 抛弃“当前主版本已经是 M4”的假设。
- 抛弃“必须在同一个主 benchmark 中同时解决 clean/dropout/pollution/bias_ramp”
  的叙事。
- 抛弃继续围绕协方差融合路径作为主方法的想法；covariance-aware fusion 只作为
  诊断分支，除非后续能证明其 tail 稳定。
- 抛弃只用 `quick_rmse_pos` 或单条 quick sim 判断方法优劣的做法。
- 抛弃为每一层测试单独训练一个专用模型作为主论文结果的做法。
- 抛弃把极端 drift 混入第二层常规退化 benchmark 的做法。

## 3. 三层实验架构

### 3.1 第一层：Nominal Fusion Benchmark

目的：证明模型在无注入故障时具备基础异构融合能力。

建议只保留少量但有表达力的 nominal scene：

| 场景 | 设计 | 回答的问题 |
|---|---|---|
| S1-balanced | 四个异构传感器覆盖均衡 | 标准异构融合能力 |
| S2-clustered | 传感器集中在一侧，几何冗余下降 | 几何布局困难时是否稳定 |
| S3-maneuver | 布局保持 S1，目标轨迹含转弯/加速度变化 | 动态变化下是否稳定 |

第一层不强调污染。主要报告：

- RMSE / p95 / p99 / max；
- 与 AVG、CI、WAA、best-single、post-only GNN 的对比；
- per-scene 稳定性。

### 3.2 第二层：Observable Degradation Benchmark

目的：评估常规可观测退化下的可靠性融合能力。

第二层不包含 mild drift / hard drift。推荐退化族：

| 退化族 | 含义 | 可观测性 |
|---|---|---|
| Missing / Dropout | 传感器缺测或失效 | valid/mask 明显变化 |
| Noise Inflation | 测量噪声变大但仍输出 | innovation/NIS/噪声统计变化 |
| Spike / Outlier | 短时异常跳点 | 短时残差和 peer 差异明显 |
| Shock / Offset | 突然持续偏置 | 比 slow drift 更容易暴露 |

建议编排方式：

- 每条 rollout 只注入一种 primary degradation，避免归因混乱。
- 退化目标传感器在 S1-S4 间均衡采样。
- 退化窗口随机但受控，例如 `t0 in [25, 45]`，持续 `20-35s`。
- 每个退化族设置 low / medium / high 三档强度。
- train/val/test 均保持 family 和 severity 的基本均衡。

第二层主要报告：

- per-family p95 / p99 / max；
- per-severity robustness curve；
- fault sensor mean weight；
- fault sensor top-rate；
- reliability separation。

### 3.3 第三层：Hidden Drift & Recovery Stress Test

目的：专门评估窗口型故障、窗口后恢复、隐性慢漂移和方法边界。

第三层不是“只测 `bias_ramp`”，而是测一组恢复压力场景：

| 压力测试 | 测试重点 |
|---|---|
| Mild Drift | 中等慢漂移，检验模型是否能处理轻度隐性退化 |
| Hard Drift / bias_ramp | 慢漂移、局部自洽、跨传感器不一致的最难场景 |
| Shock Recovery | 突然偏置结束后，局部状态是否残留 |
| Dropout Reacquisition | 缺测恢复后，传感器是否能重新被信任 |

推荐固定阶段：

```text
pre-window:      0-30s
fault-window:    30-70s
recovery-window: 70-90s
post-stable:     90-120s
```

第三层重点指标：

- fault-window p95 / max；
- recovery-window RMSE；
- post-window max；
- recovery time；
- post-fault error AUC；
- fault sensor weight AUC；
- time-to-downweight；
- time-to-recover；
- false recovery rate：故障刚结束但状态尚未恢复时，过早给高权重的比例。

## 4. 训练协议

主模型只训练一次，三层实验主要作为测试协议。不要为每一层单独训练一个主模型。

推荐统一训练集：

```text
40% nominal
60% observable degradation
0% drift
```

也可以根据后续数据量调整为：

```text
35% nominal
65% observable degradation
0% drift
```

关键原则：

- 训练集中不放 mild drift / hard drift，使第三层成为严格 OOD stress test。
- 每条 rollout 只含一种主退化。
- 第二层退化族和强度要均衡。
- 同一个训练协议用于所有主模型和消融模型。

验证集建议拆成两套：

| 验证集 | 用途 |
|---|---|
| `val_in_distribution` | 与训练同分布，用于普通收敛监控 |
| `val_robust_small` | 小规模固定鲁棒验证集，用于 checkpoint selection |

当前只按验证 MSE 选 best checkpoint 会掩盖 tail risk。后续建议增加鲁棒选择分数：

```text
score = nominal_rmse
      + lambda1 * degradation_p95
      + lambda2 * fault_sensor_top_rate
      + lambda3 * recovery_window_p95
```

`recovery_window_p95` 可以只用于选择扩展实验 checkpoint；主训练集仍不包含 drift。

## 5. 消融协议

所有消融模型使用同一训练集、同一验证协议、同一三层测试协议。

建议候选：

- post-only baseline；
- dual-stream direct；
- full reliability gate；
- no gate feature path；
- no gate weight path；
- no measurement representation；
- no peer feature；
- recovery-aware 或 risk-target 扩展分支；
- covariance diagnostic branch。

主论文不要把 covariance-aware fusion 作为默认主方法，除非它在 full-tail 指标上稳定。

## 6. 论文叙事建议

推荐论文故事：

1. 我们提出可靠性引导的异构多传感器融合模型。
2. 第一层证明它在 nominal 多场景下具备基础融合能力。
3. 第二层证明它在可观测传感器退化下具有鲁棒性。
4. 第三层展示窗口恢复和隐性慢漂移是更高难度问题，并系统分析模型边界。
5. 如有轻量恢复机制或 risk-aware checkpoint selection，可作为第三层改进点报告。

不推荐论文故事：

- “我们解决了 clean/dropout/pollution/bias_ramp 四类污染。”
- “M4 是最终主版本。”
- “协方差校准是核心贡献。”
- “只要 quick RMSE 好就证明模型有效。”

## 7. 旧结果如何使用

旧结果不删除，但角色调整如下：

| 旧结果 | 新角色 |
|---|---|
| M3/M4 peer-consistency | 早期 tail-risk 诊断证据 |
| P0-P9 main ablation | 模块消融候选池和 cov 分支失败证据 |
| P11/P12 | 新主模型候选，不自动视为最终主版本 |
| bias_ramp max-error diagnosis | 第三层 hidden drift stress test 的动机 |

后续新增实验结果应优先落到三层框架中，而不是继续扩展旧 M4/Px 版本树。

## 8. 当前主实验执行记录（2026-06-09）

当前主线已经切换为三阶段实验架构：

1. Phase 1：Nominal Fusion Benchmark，先做 clean nominal benchmark。
2. Phase 2：Observable Degradation Benchmark，只放可观测退化，不放 drift。
3. Phase 3：Hidden Drift & Recovery Stress Test，专门处理 hidden drift、窗口恢复和边界压力。

这三阶段是当前主实验，不再继续把旧 `M4` 或 `P0-P12` 版本树作为主线。旧结果保留为诊断证据、候选模型证据和消融材料。

### 8.1 Phase 1 执行入口

新增脚本：

```powershell
py -u scripts/run_phase1_nominal_benchmark.py
```

默认等待 Agent 1 提供/确认三个 nominal preset：

| 场景 | 默认 preset 占位 |
|---|---|
| S1-balanced | `phase1_s1_balanced_hetero_nominal` |
| S2-clustered | `phase1_s2_clustered_hetero_nominal` |
| S3-maneuver | `phase1_s3_maneuver_hetero_nominal` |

如果实际 preset 名称不同，通过命令行覆盖：

```powershell
py -u scripts/run_phase1_nominal_benchmark.py --dry-run `
  --scenario-presets "S1=<preset_s1>,S2=<preset_s2>,S3=<preset_s3>"
```

dry-run：

```powershell
py -u scripts/run_phase1_nominal_benchmark.py --dry-run
```

smoke-run：

```powershell
py -u scripts/run_phase1_nominal_benchmark.py --smoke `
  --scenario-presets "S1=phase1_s1_balanced_hetero_nominal,S2=phase1_s2_clustered_hetero_nominal,S3=phase1_s3_maneuver_hetero_nominal" `
  --out-dir results/phase1_nominal_benchmark_smoke
```

正式 run：

```powershell
py -u scripts/run_phase1_nominal_benchmark.py `
  --scenario-presets "S1=phase1_s1_balanced_hetero_nominal,S2=phase1_s2_clustered_hetero_nominal,S3=phase1_s3_maneuver_hetero_nominal" `
  --model-seeds 0,1,2,3,4 `
  --train-seed-range 10-69 `
  --val-seed-range 70-89 `
  --test-seed-range 90-109 `
  --epochs 80 `
  --out-dir results/phase1_nominal_benchmark
```

详细计划见 `configs/PHASE1_NOMINAL_BENCHMARK_PLAN.md`。

### 8.2 Phase 1 当前比较对象

Phase 1 runner 当前默认比较池训练/评估：

- `P0_post_only_single_stream`
- `P1_dual_stream_direct`
- `P11_feature_stable_reliability_no_cov`

其中 `--methods main` 只训练 P11；`--methods compare` / 默认配置训练
P0/P1/P11；`--methods extended` 才纳入 P4/P12 作为扩展诊断。

训练口径是单一 `S1/S2/S3 mixed training set`，测试时分别输出 S1、S2、S3
指标，不再为每个场景单独训练一个主模型。

并在每个 test split 上聚合 rule baselines：

- `AVG`
- `CI-multi`
- `WAA-MM`
- `best-single`

输出聚合指标：

- RMSE
- p95
- p99
- max

聚合文件：

- `phase1_run_summary.json` / `phase1_run_summary.csv`
- `phase1_aggregate_by_scene.json` / `phase1_aggregate_by_scene.csv`
- `phase1_aggregate_overall.json` / `phase1_aggregate_overall.csv`

### 8.3 明确废弃概念

- 不为每一层训练一个专用主模型作为主论文结果。
- 不把 drift 放进 Phase 2 常规 observable degradation benchmark。
- 第三层才做 hidden drift / recovery stress。
- `P0-P12` 是候选与证据池，不再是主实验树本身。
