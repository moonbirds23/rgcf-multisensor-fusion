# Codex 论文撰写规范与自我约束

本文档用于约束 Codex 在 `workshop_paper` 目录下协助撰写论文、摘要、实验分析、图表说明和答辩材料时的行为。任何论文内容生成前，应优先遵守本文档。

## 1. 总体原则

1. 论文只推出一个确定的最终方法，不把项目内部探索过程写成论文主线。
2. 内部模型迭代、参数调试、失败尝试、阶段命名只能作为实验记录、消融实验或边界分析，不能作为创新点本身。
3. 所有创新点必须从问题定义、方法结构和可验证贡献出发，而不是从代码版本演进出发。
4. 不夸大实验结论。没有公平对照、没有同一数据集、没有同一 seed 协议的结果，不能写成正式性能优势。
5. 不把 smoke test、dry-run、小规模验证写成论文主结果。它们只能用于说明流程跑通或机制初步有效。
6. 论文表述必须区分“已证明”“实验显示”“初步观察”“推测原因”。不能把推测写成事实。

## 1.1 最终论文版本与论文名称

最终论文版本已经确定为项目内部实现 `ME-RGCF-A0D`。

论文中不直接使用 `ME-RGCF-A0D` 作为主方法名称。该内部名只可在代码实现说明、
实验复现实验表或附录中作为 implementation name 出现。

论文展示名称确定为：

```text
EHGCF: Evidence-aware Heterogeneous Graph Calibrated Fusion
```

中文名称：

```text
证据感知异构图校准融合方法
```

名称含义：

| 组成 | 对应论文创新点 |
|---|---|
| Evidence-aware | 测量证据与后验信息的共同引入 |
| Heterogeneous Graph | 面向 P/M 节点的 GNN 异构网络 |
| Calibrated Fusion | 可靠性门控与协方差尺度校准的信息融合公式 |

后续论文正文、摘要、标题、图表和实验主结果中，统一使用 `EHGCF` 或
“证据感知异构图校准融合方法”指代最终方法。

## 2. 创新点固定边界

当前论文创新点限定为三项：

### 2.1 测量证据与后验信息的共同引入

可写内容：

- 同时利用后验状态、跟踪量测残差和外部测量证据。
- 区分 posterior track sensor 与 evidence-only measurement sensor。
- evidence-only sensor 不直接输出状态，不参与最终状态求和，只评价并调节后验轨迹节点。

不可写内容：

- 不写“Phase1R 修正传感器角色是一个版本创新”。
- 不写“先前版本错误，因此当前版本创新”。
- 不把简单特征堆叠夸大为理论突破。

推荐表述：

```text
本文构建了后验状态、跟踪量测残差与外部测量证据的联合表征机制，
在保持 evidence-only 传感器不直接参与状态输出的前提下，
使外部测量证据能够评价并调节各后验轨迹节点的可靠性。
```

### 2.2 GNN 异构网络

可写内容：

- 将 posterior track 建模为 P 节点。
- 将 track measurement 与 external evidence 建模为 M 节点。
- 通过 M-M、M->P、P-P 三类交互建模异构信息流。
- 若最终方法使用 pair-level evidence-to-track residual，可写为定向证据交互机制。

不可写内容：

- 不把 `RGCF -> ME-RGCF-A0 -> ME-RGCF-A0D` 写成论文贡献。
- 不写“A0D 是 A0 的版本升级，因此是创新点”。
- 不在主方法介绍中反复出现内部版本名。

推荐表述：

```text
本文设计了面向异构传感器角色的 P/M 双层图神经网络，
通过测量层一致性建模、测量到后验的定向证据交互以及后验层协商，
刻画 evidence-only 量测对不同后验轨迹的差异化影响。
```

### 2.3 融合公式的改良处理

可写内容：

- 网络只对 posterior track nodes 输出最终融合权重。
- 可靠性门控、协方差尺度和基础权重 logits 共同决定融合权重。
- 最终采用带协方差尺度校准的信息形式融合，而不是简单状态平均。

不可写内容：

- 不写成“只是 softmax 加权平均”。
- 不把 evidence-only nodes 写入最终状态求和。
- 不声称信息融合公式本身完全原创；应强调与学习到的可靠性和协方差校准结合。

推荐表述：

```text
本文提出可靠性-协方差联合校准的信息形式融合策略，
将网络预测的可靠性门控和协方差尺度纳入融合权重计算，
并在最终状态估计中使用校准后的协方差信息进行加权。
```

## 3. 内部版本的论文角色

内部版本名可以出现在实验表格或消融实验中，但正文解释必须转写为模块含义。

| 内部名称 | 论文角色 |
|---|---|
| RGCF | 过程版本；learned baseline 或 evidence pooling/broadcast 消融 |
| ME-RGCF-A0 | 过程版本；异构 P/M 图但无 pair-level 定向证据的消融 |
| ME-RGCF-A0D | 最终实现版本；论文中命名为 `EHGCF` |
| ME-RGCF-A0D-HS | 过程版本；loss 变体、机制分析或补充消融 |

写作约束：

```text
RGCF、ME-RGCF-A0、ME-RGCF-A0D-HS 均为过程版本。
它们不能作为论文最终方法，也不能作为创新点名称。
```

```text
ME-RGCF-A0D 是最终代码实现名。
论文中应将其表述为 EHGCF，而不是作为版本名进行叙述。
```

禁止将内部版本写成：

```text
本文依次提出 RGCF、ME-RGCF-A0、ME-RGCF-A0D 和 ME-RGCF-A0D-HS。
```

应写成：

```text
为验证各模块贡献，本文设置若干消融模型，分别移除异构图结构、
pair-level 定向证据交互或可靠性-协方差校准项。
```

## 4. 实验写作约束

1. 主结果必须来自正式实验，不使用 smoke/dry-run 指标。
2. 对比实验必须尽量保持同一数据集、同一 train/val/test split、同一 model seeds 和同一训练协议。
3. 如果测试点数不同，必须显式说明，不能直接比较性能优劣。
4. 只要提升幅度较小，必须谨慎表述，优先写“稳定改善”或“在该设置下取得较低误差”，不要写“显著优越”。
5. 机制诊断指标只能证明机制行为，不能自动证明最终任务性能提升。
6. 消融实验必须对应三个创新点：
   - 去掉测量证据共同建模；
   - 去掉异构图或定向 M->P 交互；
   - 去掉可靠性门控或协方差尺度校准。
7. 若缺少统计显著性检验，不能写“statistically significant”或“显著提升”。

## 5. 术语约束

优先使用以下论文术语：

- posterior track sensor
- evidence-only measurement sensor
- posterior node / P node
- measurement/evidence node / M node
- heterogeneous graph fusion
- evidence-to-track interaction
- reliability gate
- covariance scale calibration
- calibrated information fusion

慎用或避免：

- 版本升级
- 当前阶段
- A0/A0D/HS 作为正文主叙事
- 修 bug 后效果更好
- 终于解决
- 完全证明

## 6. 图表约束

推荐主图：

```text
posterior state stream
track measurement residual stream
external evidence measurement stream
        -> heterogeneous P/M GNN
        -> reliability gate + covariance scale + weight logits
        -> calibrated information fusion
```

不推荐主图：

```text
RGCF -> ME-RGCF-A0 -> ME-RGCF-A0D -> ME-RGCF-A0D-HS
```

图中必须体现：

- evidence-only nodes 不直接输出状态；
- 最终融合只在 posterior track nodes 上进行；
- measurement/evidence 通过图交互、可靠性和协方差校准间接影响融合。

## 7. 写作前自检清单

生成任何论文段落前，Codex 应自检：

1. 这段是否把内部版本迭代写成了创新点？
2. 这段是否只推出一个最终方法？
3. 这段是否把中间模型放在 baseline/消融/诊断位置？
4. 这段是否夸大了尚未公平验证的实验结果？
5. 这段是否清楚区分了 posterior sensor 与 evidence-only sensor？
6. 这段是否避免让 evidence-only nodes 直接参与状态融合？
7. 这段是否对应三个固定创新点之一？

若任一问题答案为“否”，应先重写再输出。
