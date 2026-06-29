# 论文成果表述约束与创新点定义

更新日期：2026-06-29

本文用于约束后续将项目总结为论文成果时的叙事方式。核心原则是：

```text
论文只推出一个确定的最终方法。
项目探索过程中的中间版本不能作为创新点立意。
中间版本最多作为消融实验、对照模型或失败/边界分析材料。
```

## 0. 最终论文版本与命名

最终论文版本确定为项目内部实现：

```text
ME-RGCF-A0D
```

该名称是项目内部实现名，不作为论文主方法名称。论文展示名称确定为：

```text
EHGCF: Evidence-aware Heterogeneous Graph Calibrated Fusion
```

中文名称：

```text
证据感知异构图校准融合方法
```

命名对应关系：

| 论文名组成 | 对应内容 |
|---|---|
| Evidence-aware | 测量证据与后验信息共同引入 |
| Heterogeneous Graph | 面向 posterior nodes 与 measurement/evidence nodes 的异构 GNN |
| Calibrated Fusion | 可靠性门控与协方差尺度校准的信息融合公式 |

论文中统一使用 `EHGCF` 指代最终方法。只有在复现实验、代码索引或附录中需要
说明实现对应关系时，才写：

```text
EHGCF is implemented by the internal method `ME-RGCF-A0D`.
```

## 1. 总体写作约束

论文中不要把 `RGCF -> ME-RGCF-A0 -> ME-RGCF-A0D -> A0D-HS`
写成公共领域已知问题的自然演进，也不要把这些内部版本迭代本身作为创新点。

这些名字和路径属于本项目的探索过程，不是读者天然知道的研究脉络。
论文应从问题定义、方法设计和实验验证出发，直接描述最终提出的方法。

推荐写法：

```text
本文提出一种面向异构多传感器融合的证据感知图融合方法。
该方法联合利用后验状态信息、跟踪传感器量测残差信息和外部量测证据信息，
通过异构图网络建模后验节点与量测证据节点之间的关系，并在最终融合阶段
引入可靠性权重与协方差尺度校准的信息形式融合公式。
```

不推荐写法：

```text
本文先提出 RGCF，再升级到 ME-RGCF-A0，再升级到 ME-RGCF-A0D。
```

```text
A0D 相比 A0 的版本升级是本文创新点。
```

```text
A0D-HS 是下一版，因此构成第四个创新点。
```

在论文中，中间模型只能这样使用：

| 项目内部模型 | 论文中角色 |
|---|---|
| RGCF | 过程版本；baseline 或 evidence pooling/broadcast 消融 |
| ME-RGCF-A0 | 过程版本；异构图但无 pair-level 定向证据交互的消融 |
| ME-RGCF-A0D | 最终实现版本；论文中命名为 `EHGCF` |
| ME-RGCF-A0D-HS | 过程版本；loss 消融或机制分析，不作为主方法 |

## 2. 论文最终方法的三个创新点

### 创新点一：测量证据与后验信息的共同引入

最终方法不是只融合多个传感器的后验轨迹，也不是只使用单帧原始量测。
它同时引入三类互补信息：

```text
1. 后验状态信息：各 track sensor 的 EKF 后验状态与协方差。
2. 跟踪量测信息：track sensor 的 innovation、NIS、量测噪声与几何特征。
3. 外部证据信息：evidence-only sensor 的原始量测、噪声、相对 prior/track 的残差。
```

其中，后验信息描述“每条轨迹当前估计在哪里”，测量残差信息描述
“该轨迹自身量测是否一致”，外部证据信息描述“独立证据如何评价各条轨迹”。

该创新点的重点不是简单堆叠特征，而是建立受约束的信息流：

```text
evidence-only sensor 不被伪装成 posterior state node；
它不直接输出融合状态，也不参与最终状态加权；
它只通过可靠性、协方差尺度和图交互影响 track posterior node。
```

论文中可表述为：

```text
本文构建了后验状态、跟踪量测残差与外部测量证据的联合表征机制，
在保持 evidence-only 传感器不直接参与状态输出的前提下，
使外部测量证据能够评价并调节各后验轨迹节点的可靠性。
```

### 创新点二：面向异构传感器角色的 GNN 异构网络

最终方法将传感器节点区分为 posterior track nodes 与 measurement/evidence nodes，
并使用异构图网络描述二者之间的交互。

推荐抽象为：

```text
P nodes: 后验轨迹节点，携带状态估计、协方差和有效性。
M nodes: 测量/证据节点，携带量测、残差、噪声和传感器角色。
```

图网络的核心不是“版本升级”，而是三类关系建模：

```text
M-M: 测量证据之间的一致性与互相校验。
M->P: 测量证据对后验轨迹节点的定向评价。
P-P: 后验轨迹节点之间的一致性协商。
```

如果最终方法使用 pair-level evidence-to-track 特征，则应归入本创新点：

```text
M->P 交互不是全局广播，而是基于每个 measurement/evidence node
与每个 posterior node 的 pair feature 进行定向注意力建模。
```

论文中可表述为：

```text
本文设计了面向异构传感器角色的 P/M 双层图神经网络，
通过测量层一致性建模、测量到后验的定向证据交互以及后验层协商，
刻画 evidence-only 量测对不同后验轨迹的差异化影响。
```

### 创新点三：融合公式的改良处理

最终融合阶段不应写成普通 softmax 加权平均。项目中的核心处理是：

```text
1. 网络只为 posterior track nodes 输出融合权重。
2. 网络同时输出 reliability gate 和 covariance scale。
3. 最终使用带协方差尺度校准的信息形式融合，而不是简单状态平均。
```

抽象公式可以写为：

```math
\ell_i =
\frac{a_i}{T}
+ \alpha \log(g_i+\epsilon)
- \beta \log(s_i+\epsilon)
```

```math
w_i = \operatorname{softmax}_i(\ell_i)
```

其中 \(a_i\) 是基础权重 logit，\(g_i\) 是可靠性估计，\(s_i\) 是协方差尺度。

最终融合使用：

```math
\tilde{P}_i = s_i P_i
```

```math
\hat{x}
=
\frac{\sum_i w_i \tilde{P}_i^{-1}\hat{x}_i}
       {\sum_i w_i \tilde{P}_i^{-1}}
```

该创新点的重点是：权重、可靠性和协方差校准共同作用于融合，而 evidence-only
节点不直接进入最终求和。

论文中可表述为：

```text
本文提出可靠性-协方差联合校准的信息形式融合策略，
将网络预测的可靠性门控和协方差尺度纳入融合权重计算，
并在最终状态估计中使用校准后的协方差信息进行加权，
从而降低低可靠后验轨迹对融合结果的影响。
```

## 3. 论文创新点的推荐最终写法

可以在论文贡献段落中写成：

```text
本文的主要贡献如下：

1. 提出一种测量证据与后验状态共同建模的异构多传感器融合框架。
   该框架区分 posterior track sensor 与 evidence-only measurement sensor，
   在不将 evidence-only 传感器伪装为状态后验节点的前提下，
   联合利用后验状态、跟踪量测残差和外部量测证据。

2. 设计一种面向传感器角色的异构图神经网络。
   该网络将后验轨迹建模为 P 节点，将跟踪量测和外部证据建模为 M 节点，
   通过 M-M、M->P 和 P-P 交互刻画测量证据对不同后验轨迹的差异化影响。

3. 构造一种可靠性-协方差联合校准的信息形式融合公式。
   网络只对 posterior track nodes 输出融合权重，并结合可靠性门控与协方差尺度
   对最终信息加权融合进行调节，从而提升多传感器融合的稳定性与鲁棒性。
```

## 4. 实验叙事约束

实验对比应服务于最终方法，而不是服务于版本故事。

主实验推荐包含：

```text
最终方法 vs 经典融合方法：
single sensor / average / WAA-MM / CI

最终方法 vs learned baseline：
去掉测量证据共同建模、去掉异构图、去掉 pair-level M->P 交互、
去掉可靠性或协方差校准等。
```

消融实验中可以出现内部版本名，但正文解释应转写为模块名：

| 内部名 | 论文解释 |
|---|---|
| RGCF | 过程版本；evidence pooling/broadcast baseline |
| ME-RGCF-A0 | 过程版本；heterogeneous P/M graph without pair-level M->P evidence |
| ME-RGCF-A0D | 最终实现版本；论文中命名为 `EHGCF` |
| ME-RGCF-A0D-HS | 过程版本；directional loss variant / loss ablation |

论文图示也应画最终方法结构，而不是画项目版本演进路线。

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
RGCF -> A0 -> A0D -> HS
```
