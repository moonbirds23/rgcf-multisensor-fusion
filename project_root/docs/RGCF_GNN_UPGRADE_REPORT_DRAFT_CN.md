# RGCF 图神经网络升级汇报草稿

更新时间：2026-06-22

## 1. 论文成果表述约束

本文档保留项目内部的模型关系和实验对比，但论文写作时不能把内部版本迭代
当作创新点立意。论文只应推出一个确定的最终方法；其他模型仅作为 baseline、
消融实验或机制诊断材料。

论文贡献应固定为三点：

```text
1. 测量证据与后验信息的共同引入。
2. 面向后验节点与测量/证据节点的 GNN 异构网络。
3. 结合可靠性门控与协方差尺度校准的改良信息融合公式。
```

论文中不要写成：

```text
RGCF -> ME-RGCF-A0 -> ME-RGCF-A0D -> ME-RGCF-A0D-HS
```

这种路径属于项目探索过程，不是公共已知常识，也不是论文贡献本身。

内部模型在论文中的角色应降级为：

| 内部模型 | 论文角色 |
|---|---|
| RGCF | learned baseline 或去除异构图/定向证据交互的消融 |
| ME-RGCF-A0 | 异构图但无 pair-level M->P 证据的消融 |
| ME-RGCF-A0D | 当前主方法候选或最终推出方法 |
| ME-RGCF-A0D-HS | loss 变体/机制诊断，公平补跑前不作为主方法 |

目前可作为主结论的正式同规模对比是：

```text
RGCF / ME-RGCF-A0 / ME-RGCF-A0D
```

`ME-RGCF-A0D-HS` 的现有结果证明了方向性机制有效，但测试集规模与旧 A0D
不同，因此不能直接作为替代 A0D 的性能结论。

详细论文表述约束见：

```text
docs/PAPER_FRAMING_CONSTRAINTS_CN.md
```

## 2. 问题定义

每个时刻有三条后验 track 节点：

```text
P = {P1, P2, P3}
```

还有五个 measurement/evidence 节点：

```text
M = {M1, M2, M3, M4, M5}
```

其中：

```text
M1-M3: 对应三条 track 的自身量测/创新特征
M4-M5: evidence-only 节点，只评价 track，不直接参与状态输出
```

最终融合只输出三条 track 的状态加权结果，evidence 节点不直接输出状态。

## 3. RGCF：全局 evidence pooling

RGCF 的核心设计是：先把 evidence 节点编码后池化成一个全局 evidence 上下文，
再把该上下文广播到每个 track 节点。

设 posterior track 特征为 \(x_i^P\)，track measurement 特征为 \(x_i^M\)，
evidence 特征为 \(x_e^E\)。

编码：

```math
h_i^P = f_P(x_i^P)
```

```math
h_i^M = f_M(x_i^M)
```

```math
h_e^E = f_E(x_e^E)
```

evidence pooling：

```math
\alpha_e^E =
\operatorname{softmax}_e \left( \phi_E(h_e^E) \right)
```

```math
\bar{h}^E = \sum_e \alpha_e^E h_e^E
```

track 节点初始表示：

```math
z_i =
f_{\text{cross}}\left(
  h_i^P,\ h_i^M,\ \bar{h}^E,\ |h_i^P-h_i^M|,\ h_i^P \odot h_i^M
\right)
```

然后做 P-P 图传播：

```math
\alpha_{i\ell}^{PP}
= \operatorname{softmax}_{\ell}
\left(\phi_{PP}([z_i,z_\ell])\right)
```

```math
\tilde{z}_i =
u_{PP}\left([z_i,\sum_{\ell \ne i}\alpha_{i\ell}^{PP} z_\ell]\right)
```

RGCF 的优点是稳定：evidence 能影响每个 track 的可靠性、协方差尺度和最终权重。
但它的限制也很明确：\(\bar{h}^E\) 是全局上下文，对所有 track 广播。
因此 evidence 可以表达“当前证据整体可靠/不可靠”，但很难表达：

```text
这个 evidence 更反对 P1，而不是 P2/P3。
```

## 4. ME-RGCF-A0：异构 P/M 图

A0 的升级是把 measurement/evidence 作为显式 M 节点进入图，而不是先池化成一个
全局 evidence 向量。

节点编码：

```math
h_i^P = f_P(x_i^P) + e_P
```

```math
h_j^M = f_M(x_j^M) + e_{\text{role}(j)}
```

其中 role embedding 区分：

```text
track measurement 节点: M1-M3
evidence-only 节点:    M4-M5
```

A0 包含三段交互：

1. M-M self graph：measurement/evidence 节点先互相交互。

```math
\alpha_{j\ell}^{MM}
= \operatorname{softmax}_{\ell}
\left(\phi_{MM}([h_j^M,h_\ell^M])\right)
```

```math
\tilde{h}_j^M
= u_{MM}\left([h_j^M,\sum_\ell \alpha_{j\ell}^{MM}h_\ell^M]\right)
```

2. M->P cross graph：M 节点把证据信息写入 P 节点。

```math
e_{ij}^{MP} = \phi_{MP}([h_i^P,\tilde{h}_j^M])
```

```math
\alpha_{ij}^{MP}
= \operatorname{softmax}_{j}(e_{ij}^{MP})
```

```math
\hat{h}_i^P
= u_{MP}\left([h_i^P,\sum_j \alpha_{ij}^{MP}\tilde{h}_j^M]\right)
```

3. P-P graph：track 节点之间再做一致性传播。

```math
\alpha_{i\ell}^{PP}
= \operatorname{softmax}_{\ell}
\left(\phi_{PP}([\hat{h}_i^P,\hat{h}_\ell^P])\right)
```

```math
\tilde{h}_i^P
= u_{PP}\left([\hat{h}_i^P,\sum_{\ell\ne i}\alpha_{i\ell}^{PP}\hat{h}_\ell^P]\right)
```

A0 的结构比 RGCF 更丰富，因为 evidence 不再只是被池化后广播，而是作为 M4/M5
进入图中参与 M-M、M->P 交互。

但 A0 有一个关键短板：M->P attention 的打分只看 \([h_i^P,h_j^M]\)，没有显式的
pair-wise residual。也就是说，M4/M5 虽然进入了图，但它不知道自己对 P1/P2/P3
分别有多大残差，因此容易退化成近似广播。

这一点在诊断数据中非常明显：

```text
ME-RGCF-A0 row_std_mean = 0.000059
```

M->P attention 的不同 P 行几乎一样，说明 evidence 对不同 track 的区分很弱。

## 5. ME-RGCF-A0D：pair-aware M->P attention

A0D 是本次图神经网络升级的关键版本。它不只是把 M 节点放进图里，还在每个
P-M pair 上加入方向性特征：

```math
q_{ij} \in \mathbb{R}^{8}
```

代码中的 pair feature 含义为：

| 维度 | 含义 |
|---:|---|
| 0 | `own_track`: \(M_j\) 是否为 \(P_i\) 自己的 track measurement |
| 1 | `other_track`: \(M_j\) 是否为其他 track 的 measurement |
| 2 | `evidence`: \(M_j\) 是否为 evidence-only 节点 |
| 3 | `log_residual`: evidence 到该 \(P_i\) 的 \(\log(1+r)\) |
| 4 | `centered_residual`: 相对同一 evidence 下不同 P 的中心化 residual |
| 5 | `rank`: 同一 evidence 下 residual 排名，越大表示越差 |
| 6 | `valid_p`: 该 P 节点是否有效 |
| 7 | `valid_m`: 该 M 节点是否有效 |

对于 M4/M5，若 evidence 对 P1/P2/P3 的 residual 为：

```math
r_{e,i}
```

则：

```math
q_{i,e}^{(3)} = \log(1+r_{e,i})
```

```math
q_{i,e}^{(4)}
= \operatorname{clip}
\left(
  \frac{\log(1+r_{e,i}) -
  \frac{1}{3}\sum_{k=1}^{3}\log(1+r_{e,k})}{3},
  -1,1
\right)
```

```math
q_{i,e}^{(5)} =
\operatorname{rank}_{k\in\{1,2,3\}}\left(\log(1+r_{e,k})\right)
```

A0D 的 M->P attention 打分变为：

```math
g_{ij} = f_q(q_{ij})
```

```math
e_{ij}^{MP}
= \phi_{MP}([h_i^P,\tilde{h}_j^M,g_{ij}])
 + b_{\text{id}} q_{ij}^{(0)}
 + b_{\text{res}} q_{ij}^{(2)}q_{ij}^{(5)}
```

```math
\alpha_{ij}^{MP}
= \operatorname{softmax}_{j}(e_{ij}^{MP})
```

消息也从“只传 M 节点”升级为 pair-aware 消息：

```math
m_{ij}=f_{\text{msg}}([\tilde{h}_j^M,g_{ij}])
```

```math
\hat{h}_i^P
= u_{MP}\left([h_i^P,\sum_j\alpha_{ij}^{MP}m_{ij}]\right)
```

这一步的含义很重要：

```text
同一个 evidence M4/M5 面对 P1/P2/P3 时，不再是同一个消息。
它会携带 pair residual / rank 信息，从而表达“我更反对哪条轨迹”。
```

因此 A0D 相较于 RGCF/A0 的核心优越性是：

```text
RGCF:  evidence 是全局广播上下文
A0:    evidence 是图节点，但方向性仍弱
A0D:   evidence 是图节点，并且对每个 P-M pair 有显式方向性
```

## 6. 输出融合公式

最终融合仍然只在 P 节点上完成。对每个 track 节点输出：

```math
g_i = \sigma(f_g(\tilde{h}_i^P)) \cdot \text{valid}_i
```

```math
s_i = 1 + \operatorname{softplus}(f_s(\tilde{h}_i^P))
```

其中 \(g_i\) 是可靠性 gate，\(s_i\) 是协方差尺度。

权重 logit 为：

```math
\ell_i =
\frac{f_w(\tilde{h}_i^P)}{T}
+ \alpha_g \log(g_i+\epsilon)
- \beta_s \log(s_i+\epsilon)
```

```math
w_i = \operatorname{softmax}_i(\ell_i)
```

在 `info_diag` 融合模式下，若 track 状态为 \(\hat{x}_i\)，对角协方差为
\(P_i\)，则先用 \(s_i\) 放大协方差：

```math
\tilde{P}_i = s_i P_i
```

融合输出为信息加权平均：

```math
\hat{x}
=
\frac{\sum_i w_i \tilde{P}_i^{-1}\hat{x}_i}
       {\sum_i w_i \tilde{P}_i^{-1}}
```

因此 A0D 的 pair-aware evidence 不是直接输出状态，而是通过：

```text
M->P attention -> P node embedding -> gate/cov_scale/weight -> information fusion
```

间接影响最终轨迹融合。

## 7. A0D-HS：高 residual spread 的方向性 loss

A0D-HS 不改变上述图结构，只改变 auxiliary directional loss。

原 A0D 使用 full softmax 方向性约束：

```math
L_{\text{dir}}
=
D_{KL}
\left(
  \operatorname{softmax}
  (w_{\text{id}}q^{(0)} + w_{\text{ev}}q^{(2)}q^{(5)})
  \ \Vert\ 
  \alpha^{MP}
\right)
```

HS 版本把方向性拆成两部分：

```math
L_{\text{dir}}^{HS}
=
0.25L_{\text{id}} + 1.0L_{\text{ev-hs}}
```

own-track anchor：

```math
L_{\text{id}}
=
-\frac{1}{|P|}
\sum_i \log \alpha_{i,i}^{MP}
```

evidence-to-track high-spread 约束：

对每个 evidence \(e\)，计算 residual spread：

```math
\operatorname{spread}_e
=
\max_i q_{i,e}^{(3)} - \min_i q_{i,e}^{(3)}
```

用 batch 内分位数 \(Q_{0.70}\) 与 \(Q_{0.90}\) 构造 gate：

```math
\gamma_e =
\operatorname{clip}
\left(
\frac{\operatorname{spread}_e-Q_{0.70}}
     {Q_{0.90}-Q_{0.70}+\epsilon},
0,1
\right)
```

然后只在 residual 差异明显的窗口强化 evidence 对 P 的排序：

```math
\pi_{e,i}
=
\operatorname{softmax}_i
\left(
\frac{w_{\text{ev}}}{\tau}q_{i,e}^{(5)}
\right)
```

```math
L_{\text{ev-hs}}
=
\frac{\sum_e \gamma_e
D_{KL}(\pi_e\Vert \bar{\alpha}_{e})}
{\sum_e \gamma_e}
```

其中 \(\bar{\alpha}_e\) 是 M4/M5 在 P 维度内归一化后的 attention。

HS 的意义是：不要让 residual 差异不明显的窗口干扰方向性训练；只在 evidence
确实能区分 P1/P2/P3 的窗口里强化排序。

## 8. 正式同规模性能对比

以下结果来自同一类正式 Phase1R S1R/S2R clean benchmark。
learned 方法均为 5 个 model seed、2 个 scene，总计 230,200 个 learned test points。

| 方法 | 图结构含义 | n_runs | points | RMSE | P95 | P99 | Max |
|---|---|---:|---:|---:|---:|---:|---:|
| RGCF | evidence pooling / broadcast | 10 | 230200 | 2.7013 | 5.2191 | 6.1094 | 8.1943 |
| ME-RGCF-A0 | 异构 P/M 图，无 pair residual | 10 | 230200 | 2.7056 | 5.2178 | 6.1151 | 8.0140 |
| ME-RGCF-A0D | pair-aware M->P attention | 10 | 230200 | 2.6722 | 5.1751 | 6.0638 | 8.0581 |
| CI-3T | 规则基线 | 2 | 46040 | 3.1528 | 5.8705 | 6.9443 | 9.0682 |
| WAA-MM-3T | 规则基线 | 2 | 46040 | 3.6040 | 6.8086 | 7.7426 | 9.1351 |
| AVG-3T | 规则基线 | 2 | 46040 | 4.0001 | 7.6947 | 9.1133 | 10.5083 |

A0D 相对 RGCF：

| 指标 | RGCF | A0D | 相对变化 |
|---|---:|---:|---:|
| RMSE | 2.7013 | 2.6722 | -1.08% |
| P95 | 5.2191 | 5.1751 | -0.84% |
| P99 | 6.1094 | 6.0638 | -0.75% |
| Max | 8.1943 | 8.0581 | -1.66% |

A0D 相对 A0：

| 指标 | A0 | A0D | 相对变化 |
|---|---:|---:|---:|
| RMSE | 2.7056 | 2.6722 | -1.23% |
| P95 | 5.2178 | 5.1751 | -0.82% |
| P99 | 6.1151 | 6.0638 | -0.84% |
| Max | 8.0140 | 8.0581 | +0.55% |

解释：

```text
A0D 的平均误差、P95、P99 相对 RGCF/A0 均稳定改善；
Max 相对 A0 略高，但仍低于 RGCF，且整体 tail 指标 P95/P99 改善。
```

## 9. 分场景结果

| 场景 | 方法 | n_runs | points | RMSE | P95 | P99 | Max |
|---|---|---:|---:|---:|---:|---:|---:|
| S1R | RGCF | 5 | 115100 | 2.1262 | 4.1506 | 4.9907 | 7.1588 |
| S1R | ME-RGCF-A0 | 5 | 115100 | 2.1309 | 4.1504 | 5.0043 | 7.0209 |
| S1R | ME-RGCF-A0D | 5 | 115100 | 2.0997 | 4.1091 | 4.9675 | 6.9039 |
| S2R | RGCF | 5 | 115100 | 3.2765 | 6.2877 | 7.2280 | 9.2299 |
| S2R | ME-RGCF-A0 | 5 | 115100 | 3.2803 | 6.2851 | 7.2259 | 9.0070 |
| S2R | ME-RGCF-A0D | 5 | 115100 | 3.2448 | 6.2411 | 7.1602 | 9.2124 |

分场景 RMSE 改善：

```text
S1R: A0D vs RGCF  -1.25%
S1R: A0D vs A0    -1.46%
S2R: A0D vs RGCF  -0.97%
S2R: A0D vs A0    -1.08%
```

说明 A0D 的收益不是只来自某一个场景；basic 与 maneuver 场景都有正向收益。

## 10. 结构诊断：A0D 是否真的改变了图交互

A0 和 A0D 的关键差别不只体现在 RMSE，而是体现在 M->P attention 是否摆脱广播。

| 方法 | n_runs | entropy | row_std | evidence_mass | own_track_mass | residual_corr | high_spread_corr | hit_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ME-RGCF-A0 | 10 | 1.5957 | 0.000059 | 0.3969 | 0.2010 | N/A | N/A | N/A |
| ME-RGCF-A0D | 10 | 1.4896 | 0.0647 | 0.4176 | 0.3192 | 0.0845 | 0.1468 | 0.4201 |

解读：

```text
A0 row_std 接近 0，说明不同 P 行看到的 M->P attention 几乎一样；
A0D row_std 提升到 0.0647，说明 P1/P2/P3 对 M1-M5 的关注开始分化；
A0D residual_corr 为正，说明 attention 与 evidence residual/rank 有弱方向性；
但 residual_corr 仍不够强，因此后续才设计了 HS loss 做机制增强。
```

这组诊断是 A0D 相对 A0 的关键证据：A0 引入了异构图，但 A0D 才让 evidence
具备了面向不同 track 的方向表达能力。

## 11. A0D-HS 结果如何使用

现有 A0D-HS 结果：

```text
E:\migration_packages\results\phase2_me_a0_dir_hs_only
```

该结果测试规模较小：

```text
A0D-HS: 2010 learned test points
旧 A0D: 230200 learned test points
```

因此 A0D-HS 不能直接参与上面的正式性能排序。但它可以作为方向性机制验证。

| 方法 | n_runs | entropy | row_std | raw_corr | high_spread_corr | evidence_norm_corr | hit_rate | worst_low_weight_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0D-HS small | 10 | 1.2598 | 0.1001 | 0.8949 | 0.9092 | 0.6311 | 1.0000 | 0.5554 |

与旧 A0D 诊断参考相比：

```text
row_std:            0.0647 -> 0.1001
high_spread_corr:   0.1468 -> 0.9092
hit_rate:           0.4201 -> 1.0000
```

这说明 HS loss 确实能强力提升 M4/M5 对 P1/P2/P3 的方向排序。

但同时：

```text
worst_low_weight_rate = 0.5554 < 0.80
delta_w_mean          = -0.0234
delta_cov_mean        = +0.0589
```

这表示 attention 已经会排序，但下游 \(w/g/cov\_scale\) 的响应还不够强。
所以 HS 当前更适合作为 loss 消融与机制证明，不建议在公平补跑完成前替代 A0D。

## 12. 汇报结论建议

建议在汇报中使用以下表述：

```text
本文提出一种面向异构多传感器融合的证据感知图融合方法。
该方法共同引入后验状态、跟踪量测残差和外部测量证据，
在区分 posterior track sensor 与 evidence-only measurement sensor 的前提下，
使用异构图网络建模测量证据对不同后验轨迹的差异化影响。

方法层面，本文构建 posterior nodes 与 measurement/evidence nodes 组成的异构图，
通过测量层一致性、测量到后验的定向证据交互以及后验层协商，
将外部 evidence 对 track 的评价传递到可靠性、协方差尺度和融合权重中。

融合层面，本文不采用简单状态平均，而是使用可靠性-协方差联合校准的信息形式融合公式。
网络只对 posterior track nodes 输出融合权重，并结合可靠性门控和协方差尺度
调节最终信息加权状态估计，避免 evidence-only 节点直接参与状态求和。
```

同时保留边界：

```text
内部中间模型只作为 baseline、消融实验或机制诊断材料。
论文正文不要把内部版本演进写成创新点。
训练损失变体若未完成公平补跑，只能作为 loss 消融或机制分析。
```

## 13. 后续模块图建议

后续生成图像时，建议做三张图：

1. 最终方法总览图：

```text
posterior state stream
track measurement residual stream
external evidence measurement stream
        -> heterogeneous P/M GNN
        -> reliability gate + covariance scale + weight logits
        -> calibrated information fusion
```

2. 异构图模块图：

```text
P nodes: P1/P2/P3
M nodes: M1/M2/M3/M4/M5
M-M attention
pair feature q_ij
M->P pair-aware attention
P-P attention
gate / cov_scale / weight
information fusion
```

3. 消融与机制诊断图：

```text
without heterogeneous graph
without pair-level M->P evidence
without reliability/covariance calibration
full method
```

图中重点不要画成“evidence 直接参与状态融合”，而要画成：

```text
evidence -> attention/reliability/covariance -> track-only state fusion
```
