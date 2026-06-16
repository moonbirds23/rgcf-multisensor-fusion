# P0 重启主线方案：双流可靠性校准 AA-MM 融合网络

更新日期：2026-06-13

## 1. 背景与动机

当前 P11 系列实验暴露出一个关键问题：模型在部分 S3 maneuver 场景中会把
`valid=1` 但局部 EKF 已经严重漂移的 AOA-only / UWB-range-only 节点赋予极高
融合权重，导致少数时刻出现数百米甚至上千米的 tail error。

已复现的典型现象包括：

- S3 seed 106，约 40.5s-41.7s，UWB 节点局部位置误差约 930m-970m，
  P11 某些 checkpoint 仍给该节点约 0.96 的融合权重。
- S3 seed 96，约 86s，AOA 节点局部位置误差约 530m，
  P11 某 checkpoint 给该节点约 0.96 的融合权重。
- P1 direct dual-stream 也会出现类似问题，说明问题不只是 P11 gate 公式，
  而是量测流参与权重学习后，模型可能学到危险的 sensor shortcut。
- P0 post-only learned baseline 反而整体更稳定，尤其在 S3 tail 上明显优于 P11。

因此建议暂时停止沿 P11/P12 继续叠加模块，回到 P0 的完整稳定数据流，在
P0 的后验主干上重新引入：

1. 节点参与融合的可靠性估计；
2. 当前时刻融合结果的跟踪可信度估计；
3. 单一、可解释、可微分的经典融合公式层；
4. 双流信息交互，但避免直接黑盒输出融合状态。

本方案暂命名为：

```text
Dual-Stream Reliability-Calibrated AA-MM Fusion
双流可靠性校准 AA-MM 融合网络
```

简称：

```text
DS-RCAA
```

其中 AA-MM 表示 Arithmetic Average with Moment Matching。

## 2. 总体设计原则

### 2.1 回到 P0，但不放弃双流思想

P0 的优点是数据流稳定，后验端已经包含局部 EKF 的状态估计与协方差信息。
但 P0 缺少显式可靠性建模，也没有利用量测端关于当前时刻传感器状态的证据。

新方案保留 P0 作为主干，同时恢复 P11 系列的合理初衷：

```text
post-stream 负责判断局部后验状态是否可信；
meas-stream 负责判断当前量测证据是否支持该后验；
cross-modal interaction 负责判断二者是否一致；
fusion layer 使用显式公式生成最终融合结果。
```

### 2.2 只采用一种融合策略

上一版讨论中过多候选融合策略会导致论文主线发散。因此本方案只采用一种主融合策略：

```text
Reliability-Calibrated AA Moment Matching
可靠性校准的 AA-MM 融合
```

网络不在 AA / CI / robust center 之间选择，而是学习 AA-MM 所需的可靠性参数：

- 节点可靠性 `r_i`
- 协方差膨胀系数 `u_i`
- 节点异常概率 `bad_i`
- 当前时刻全局跟踪风险 `risk_t`

最终状态和协方差由显式 AA-MM 公式输出。

### 2.3 网络不直接回归融合状态

不建议直接使用：

```text
x_fused = MLP(all node features)
```

原因：

- 可解释性弱，难以说明为何信任某节点；
- 泛化风险高，容易记忆传感器编号或场景偏好；
- tail 风险不可控；
- 不利于与经典融合理论结合。

推荐方式：

```text
网络输出可靠性参数；
显式 AA-MM 融合层输出 x_fused 和 P_fused。
```

这样既保留深度学习能力，又保留融合理论结构。

## 3. 总体数据流

整体数据流如下：

```text
Local EKF outputs at time t
  x_i, P_i, valid_i
  z_i, R_i, innovation_i, NIS_i, geometry_i
        |
        +-----------------------------+
        |                             |
        v                             v
Post Feature Builder            Meas Feature Builder
        |                             |
        v                             v
Post Graph Encoder              Meas Graph Encoder
        |                             |
        +-------------+---------------+
                      v
        Cross-Modal Reliability Interaction
                      |
                      v
      q_post_i, q_meas_i, r_i, u_i, bad_i
                      |
                      v
        Reliability-Calibrated AA-MM Layer
                      |
                      v
      fused state x_f, fused covariance P_f,
      tracking confidence / risk
```

## 4. 输入定义

设当前时刻有 `N` 个节点，每个节点对应一个局部传感器/EKF。

### 4.1 后验端输入

每个节点的后验端原始输入：

```text
x_i in R^4
P_i in R^{4x4}
valid_i in {0,1}
```

其中：

```text
x_i = [px_i, py_i, vx_i, vy_i]
P_i = local EKF posterior covariance
```

推荐构造后验流特征：

```text
f_post_i = [
    x_i / state_scale,
    logdiag(P_i),
    valid_i,
    log trace(P_i),
    log det(P_i),
    covariance anisotropy,
    condition number proxy,
    peer position deviation,
    peer velocity deviation,
    signed delta to peer center,
    sensor type embedding
]
```

其中 peer center 建议使用 robust center，例如 median 或 Huber center，而不是普通均值。

后验流回答的问题是：

```text
从局部后验状态和节点间一致性看，该节点可信吗？
```

### 4.2 量测端输入

每个节点的量测端原始输入：

```text
z_i
R_i
innovation_i
NIS_i
sensor_type_i
geometry_i
measurement_available_i
```

推荐构造量测流特征：

```text
f_meas_i = [
    whitened innovation_i,
    abs whitened innovation norm,
    log(1 + NIS_i),
    rolling NIS,
    rolling abs whitened innovation,
    logdiag(R_i),
    sensor type embedding,
    geometry features,
    measurement availability,
    innovation jump score
]
```

量测流回答的问题是：

```text
当前量测证据是否支持该节点的局部后验？
当前传感器量测是否异常？
当前几何条件是否容易导致不可观或弱观测？
```

## 5. 双流图编码器

### 5.1 Post Graph Encoder

后验图编码器输入全部节点的后验特征：

```text
F_post = {f_post_i}_{i=1..N}
```

输出：

```text
H_post = {h_post_i}_{i=1..N}
```

建议实现：

```text
h_post_i = PostGraphEncoder(f_post_i, {f_post_j})
```

可选结构：

- MLP + Graph Attention
- Graph Transformer
- message passing + residual MLP
- 加 temporal GRU/TCN 的窗口版本

后验图编码器不直接输出融合权重，只输出节点上下文表示。

### 5.2 Meas Graph Encoder

量测图编码器输入全部节点的量测特征：

```text
F_meas = {f_meas_i}_{i=1..N}
```

输出：

```text
H_meas = {h_meas_i}_{i=1..N}
```

建议实现：

```text
h_meas_i = MeasGraphEncoder(f_meas_i, {f_meas_j})
```

量测图编码器用于建模不同传感器量测证据之间的相对异常程度，而不是直接修正状态。

## 6. Cross-Modal Reliability Interaction

### 6.1 输入是什么

交互模块输入不是原始 `x_i` 或 `z_i`，而是两条图网络编码后的节点表示：

```text
h_post_i
h_meas_i
```

同时加入显式诊断特征：

```text
d_i = [
    peer deviation,
    Mahalanobis-to-peer proxy,
    log trace(P_i),
    covariance anisotropy,
    NIS_i,
    rolling NIS_i,
    post-meas consistency score,
    valid_i,
    sensor type embedding
]
```

每个节点的交互输入可以写成：

```text
g_i = [
    h_post_i,
    h_meas_i,
    h_post_i - h_meas_i,
    h_post_i * h_meas_i,
    d_i
]
```

### 6.2 交互模块的作用

它需要回答：

```text
后验端认为该节点可信吗？
量测端是否支持该节点？
二者是否冲突？
如果冲突，应降低节点可靠性，还是只膨胀协方差？
```

推荐实现方式：

```text
post queries measurement:
a_pm_i = CrossAttention(Q=h_post_i, K=H_meas, V=H_meas)

measurement queries post:
a_mp_i = CrossAttention(Q=h_meas_i, K=H_post, V=H_post)

c_i = MLP([
    h_post_i,
    h_meas_i,
    a_pm_i,
    a_mp_i,
    h_post_i - h_meas_i,
    h_post_i * h_meas_i,
    d_i
])
```

如果初期想降低实现复杂度，可以先用：

```text
c_i = MLP(g_i)
```

然后再升级为 cross-attention。

### 6.3 输出

交互模块后接多个 head：

```text
q_post_i = sigmoid(Head_post(c_i))
q_meas_i = sigmoid(Head_meas(c_i))
r_i      = sigmoid(Head_reliability(c_i))
u_i      = 1 + softplus(Head_inflation(c_i))
bad_i    = sigmoid(Head_bad(c_i))
```

含义：

- `q_post_i`：后验端可信度；
- `q_meas_i`：量测端可信度；
- `r_i`：节点最终参与融合的可靠性；
- `u_i`：局部协方差膨胀系数；
- `bad_i`：节点异常概率。

其中 `u_i >= 1`，表示只允许放大不确定性，不建议缩小局部 EKF covariance。

## 7. 为什么需要协方差膨胀系数 u_i

当前项目中已经观察到：AOA-only / UWB-range-only 的局部 EKF 可能发生数百米甚至上千米漂移，但其 covariance 某些方向仍可能表现得过度自信。

如果后续融合直接使用原始 `P_i`，则会产生两个风险：

1. 融合状态被坏节点拉偏；
2. 融合协方差低估真实风险，tracking confidence 虚高。

因此需要网络输出 `u_i` 来修正局部 covariance：

```text
P_i_cal = u_i P_i
```

更稳妥地写：

```text
P_i_cal = clamp(u_i P_i, P_floor, P_cap)
```

`r_i` 和 `u_i` 的作用不同：

```text
r_i 控制该节点参与融合多少；
u_i 控制该节点一旦参与，其局部 covariance 应该被放大多少。
```

例如：

```text
节点状态没有完全坏，但 covariance 明显过度自信：
    r_i 可以中等；
    u_i 应该较大。

节点状态明显偏离 peers 且量测也异常：
    r_i 应该低；
    u_i 也可以高；
    bad_i 应该高。
```

## 8. 唯一融合策略：Reliability-Calibrated AA-MM

### 8.1 节点权重

由网络输出的 `r_i` 生成融合权重：

```text
s_i = valid_i * r_i
w_i = s_i / sum_j s_j
```

为了数值稳定：

```text
w_i = s_i / max(sum_j s_j, eps)
```

如果所有节点可靠性都接近 0，可以回退到 P0/PAA 默认策略，例如 valid 节点均匀权重或 best-safe node。

### 8.2 状态融合

AA 状态融合：

```text
x_f = sum_i w_i x_i
```

这一步保持简单、可解释、可微分。

### 8.3 协方差校准

先对每个节点 covariance 进行可靠性校准：

```text
P_i_cal = u_i P_i
```

可选增强：

```text
P_i_cal = D_i P_i D_i
D_i = diag(sqrt(u_i))
```

其中 `u_i` 可以是 scalar，也可以是每个状态维度一个值：

```text
u_i = [u_px, u_py, u_vx, u_vy]
```

初期建议先用 scalar `u_i`，减少训练不稳定性。

### 8.4 Moment Matching 协方差融合

AA-MM 的融合 covariance：

```text
P_f = sum_i w_i [
    P_i_cal + (x_i - x_f)(x_i - x_f)^T
]
```

其中第二项：

```text
(x_i - x_f)(x_i - x_f)^T
```

显式刻画节点之间的分歧。

这点非常重要：当多个节点状态分歧很大时，即使某些节点 covariance 很小，
`P_f` 也会因为节点间 disagreement 而变大，反映当前融合结果的风险。

### 8.5 当前时刻跟踪可信度

基于融合结果与网络上下文输出 tracking risk：

```text
risk_t = RiskHead([
    global_context,
    trace(P_f),
    logdet(P_f),
    weight_entropy,
    max bad_i,
    mean r_i,
    max node disagreement,
    post-meas conflict score
])
```

可以定义：

```text
confidence_t = exp(-risk_t)
```

或者输出归一化置信度：

```text
confidence_t = sigmoid(Head_conf(...))
```

## 9. 模型 forward 输出建议

建议模型前向输出结构如下：

```python
{
    "pred": x_f,                    # [B, 4]
    "pfused": P_f,                  # [B, 4, 4]
    "weights": w,                   # [B, N]
    "reliability": r,               # [B, N]
    "post_conf": q_post,            # [B, N]
    "meas_conf": q_meas,            # [B, N]
    "cov_inflation": u,             # [B, N] or [B, N, 4]
    "bad_score": bad,               # [B, N]
    "risk": risk_t,                 # [B]
    "confidence": confidence_t,     # [B]
    "aux": {
        "post_emb": H_post,
        "meas_emb": H_meas,
        "cross_emb": C,
        "peer_deviation": peer_dev,
        "weight_entropy": entropy,
    }
}
```

这些输出都可以用于诊断图表和论文分析。

## 10. 损失函数设计

当前 P0/P11 主要使用 MSE 或 position/velocity MSE。新方案建议改为多目标损失：

```text
L = L_state
  + lambda_rank  * L_reliability_rank
  + lambda_calib * L_cov_calibration
  + lambda_tail  * L_tail
  + lambda_cons  * L_post_meas_consistency
  + lambda_temp  * L_temporal_smooth
  + lambda_ent   * L_weight_safety
```

### 10.1 状态损失 L_state

建议使用 Huber/SmoothL1：

```text
L_state = Huber(x_f[0:2] - x_true[0:2])
        + gamma_v * Huber(x_f[2:4] - x_true[2:4])
```

相比 MSE，Huber 对异常点更稳，但仍能优化主误差。

### 10.2 可靠性排序损失 L_reliability_rank

训练时可以利用真值计算每个局部节点误差：

```text
e_i = ||x_i_pos - x_true_pos||
```

希望误差小的节点可靠性更高：

```text
if e_i + margin < e_j:
    r_i should be > r_j
```

可写成 pairwise ranking loss：

```text
L_rank = mean_{i,j} relu(margin - (r_i - r_j)) 
         for pairs where e_i < e_j
```

也可以使用 soft target：

```text
r_target_i = exp(-e_i / tau)
```

然后：

```text
L_rel = BCE(r_i, r_target_i)
```

注意：真值只在训练时用于监督 reliability，推理时不需要。

### 10.3 covariance calibration 损失 L_cov_calibration

希望融合 covariance 与实际误差匹配。

可用 NLL 形式：

```text
e_f = x_f - x_true
L_nll = 0.5 * e_f^T P_f^{-1} e_f + 0.5 * logdet(P_f)
```

或者先只对 position 做：

```text
e_pos = x_f_pos - x_true_pos
P_pos = P_f[0:2, 0:2]
L_nll_pos = 0.5 * e_pos^T P_pos^{-1} e_pos + 0.5 * logdet(P_pos)
```

这会逼迫 `u_i` 和 `P_f` 不要过度自信。

### 10.4 tail risk 损失 L_tail

为了压 p95/p99/max，可以引入 batch 内 top-k 或 CVaR：

```text
err_b = ||x_f_pos - x_true_pos||
L_tail = mean(top_k(err_b))
```

或者：

```text
L_tail = CVaR_alpha(err_b)
```

初期可以简单用 batch top 10%。

### 10.5 post-meas consistency 损失

如果 `q_post_i` 和 `q_meas_i` 长期强烈冲突，需要模型给出解释：

- 后验可信但量测异常：可以中等 reliability + 高 inflation；
- 后验不可信但量测正常：需要降低 reliability；
- 两者都不可信：低 reliability + 高 bad score。

可先用弱正则：

```text
L_cons = mean(|r_i - q_post_i * q_meas_i|)
```

或者：

```text
r_prior_i = q_post_i * q_meas_i
L_cons = BCE(r_i, stopgrad(r_prior_i))
```

### 10.6 temporal smoothness

可靠性不应无理由剧烈跳变：

```text
L_temp = mean((r_i,t - r_i,t-1)^2)
       + mean((u_i,t - u_i,t-1)^2)
```

但遇到 innovation spike 时允许跳变，可以用量测变化作为门控。

### 10.7 weight safety

训练时用局部误差构造高风险节点 mask：

```text
bad_local_i = 1 if e_i > threshold
```

惩罚高误差节点拿高权重：

```text
L_weight_safety = mean(w_i^2 for bad_local_i)
```

这直接针对 P11 的失败模式。

## 11. 评价指标设计

除了 RMSE，还应报告 tail 和可靠性行为。

### 11.1 轨迹误差

- RMSE position
- p95 error
- p99 error
- max error
- per-seed max
- per-window max

### 11.2 节点权重与可靠性

- mean weight per sensor
- top-1 sensor rate
- high-local-error top-1 rate
- high-local-error mean weight
- reliability separation
- weight entropy
- reliability temporal jump rate

### 11.3 covariance / confidence calibration

- NLL
- calibration curve
- empirical error vs predicted confidence
- high-risk recall
- low-confidence coverage

### 11.4 S3 专项诊断

必须专门报告：

- AOA/UWB local posterior error distribution
- AOA/UWB top weight rate
- S3 seed 96/106/107 max-window behavior
- confidence 是否能在 tail 爆点前升高 risk

## 12. 分阶段版本设计

建议不要一次性实现完整模型，而是分阶段推进。

### P0R0：原始 P0

用途：

- 保留当前最稳定 baseline；
- 作为所有新版本的对照。

### P0R1：P0 + peer/cov shape features

改动：

- 不引入 meas-stream；
- 只增强 post features；
- 保持简单 learned AA 或 P0 decoder。

目标：

- 验证 peer/cov shape 是否能降低 S3 tail。

### P0R2：Post Reliability AA-MM

改动：

- 引入 `r_i` 和 `u_i`；
- 使用 AA-MM 显式融合层；
- 暂不引入 meas-stream。

目标：

- 证明可靠性校准 AA-MM 在 P0 后验流上比原 P0 更稳。

### P0R3：Dual-stream Reliability

改动：

- 引入 meas-stream；
- 后验流和量测流分别编码；
- Cross-Modal Reliability Interaction 输出 `q_post_i, q_meas_i, r_i, u_i, bad_i`；
- 仍使用唯一 AA-MM 融合层。

目标：

- 验证量测端是否能提升可靠性判断，而不是放大 tail。

### P0R4：Temporal Dual-stream Reliability

改动：

- 加短窗口 GRU/TCN/Transformer；
- reliability 和 confidence 使用历史上下文。

目标：

- 处理慢漂移、恢复窗口、短时异常。

### P0R5：Risk-Calibrated DS-RCAA

改动：

- 完整 confidence/risk head；
- 加 calibration loss；
- 用风险感知 checkpoint selection。

目标：

- 输出可解释的 tracking confidence；
- 支撑论文中的可靠性评估主线。

## 13. 与 P11 的关键区别

| 维度 | P11 | 新 DS-RCAA |
|---|---|---|
| 主干 | post-only representation + meas gate bias | post-stream + meas-stream 双流交互 |
| 融合权重 | softmax learned logits + gate bias | reliability-normalized AA 权重 |
| 融合状态 | 信息融合公式，容易受局部小 P 影响 | AA-MM，显式包含节点分歧 |
| covariance | P11 no cov calibration，但仍用原始 Pdiag | 网络输出 `u_i` 校准局部 P |
| reliability | clean nominal 下全节点 target=0.8 | 训练中用 local error/ranking/calibration 监督 |
| tail 控制 | 无显式 tail loss | top-k/CVaR + weight safety |
| confidence | 弱 | 输出 tracking risk/confidence |
| 可解释性 | gate/weight 难解释 | q_post/q_meas/r/u/bad 可拆解分析 |

## 14. 实现建议

### 14.1 新模型文件

建议新增：

```text
models/dual_stream_rcaa.py
```

包含：

- `PostGraphEncoder`
- `MeasGraphEncoder`
- `CrossModalReliabilityBlock`
- `ReliabilityHeads`
- `RCAAMomentMatchingFusion`
- `DualStreamRCAAFusion`

### 14.2 新特征

在 `features/post_features.py` 中扩展：

- peer deviation
- cov shape
- sensor type embedding

在 `features/meas_features.py` 中扩展：

- innovation jump
- geometry / observability proxy
- rolling statistics

### 14.3 新 loss

建议新增：

```text
training/rcaa_losses.py
```

包含：

- `state_huber_loss`
- `reliability_ranking_loss`
- `aa_mm_nll_loss`
- `tail_cvar_loss`
- `weight_safety_loss`
- `temporal_smooth_loss`

### 14.4 新 evaluator 指标

扩展 evaluator 输出：

- local node error stats
- high-error top-rate
- reliability vs local error correlation
- risk calibration metrics
- S3 seed 96/106/107 detailed windows

## 15. 关键风险与防护

### 风险 1：可靠性仍然学成 sensor-id shortcut

防护：

- 加 sensor type dropout；
- 训练时随机 sensor order；
- 报告 per-type reliability；
- 使用 local-error ranking loss。

### 风险 2：`u_i` 无限制变大，模型逃避误差

防护：

```text
u_i = clamp(1 + softplus(a_i), 1, u_max)
```

并加入：

```text
L_u_prior = mean((log u_i)^2)
```

### 风险 3：AA-MM 过于保守

防护：

- `r_i` 仍可学习，让好节点获得高权重；
- `u_i` 只校准 covariance，不直接抹平状态；
- 用 RMSE 与 p95/p99/max 同时报导，不只看 tail。

### 风险 4：量测流再次放大 tail

防护：

- meas-stream 不直接输出融合权重；
- meas-stream 只作为 reliability evidence；
- 加 post-meas consistency loss；
- 加 high-error weight penalty。

## 16. 推荐论文叙事

可以形成以下论文故事：

1. 传统 learned fusion 直接学习节点权重，在弱观测异构传感器场景中容易放大 tail risk。
2. 仅依赖 `valid` 或局部 covariance 不足以判断节点是否可靠。
3. 提出双流可靠性校准框架：
   - 后验流评估局部后验可信度；
   - 量测流评估当前量测证据可信度；
   - 跨模态交互判断二者是否一致；
   - 网络输出 reliability 与 covariance inflation；
   - 显式 AA-MM 层输出融合状态与融合 covariance。
4. 在 S1/S2/S3 nominal、observable degradation 和 hidden drift/recovery stress 中验证：
   - 保持低 RMSE；
   - 显著降低 p99/max；
   - 提供可校准 tracking confidence。

## 17. 一句话总结

新方案不是让网络直接“选传感器”或“回归融合状态”，而是：

```text
用双流图网络学习后验端和量测端的可靠性证据，
用跨模态交互判断二者是否一致，
输出节点可靠性 r_i 和协方差膨胀 u_i，
再通过显式 Reliability-Calibrated AA-MM 公式完成融合，
同时输出当前跟踪可信度。
```

这样能够保留 P0 的稳定数据流，继承 P11 的双流动机，并避免 P11 中坏节点被高权重采信导致的 tail error 爆炸。
