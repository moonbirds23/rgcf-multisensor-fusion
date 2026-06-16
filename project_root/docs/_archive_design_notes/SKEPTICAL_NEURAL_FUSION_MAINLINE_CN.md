# Skeptical Neural Fusion 主线方案

更新日期：2026-06-13

## 0. 当前结论

项目主线从 P11/P12 和保守 DS-RCAA 方案切换为：

```text
Skeptical Neural Fusion
怀疑式神经融合 / 证据审查式神经融合
```

核心判断：

```text
局部 EKF 后验不是传感器真理，而是一个带偏见的专家证词。
融合器不应只是学习权重，而应审查每个局部滤波器在当前时刻是否有资格发言。
```

这条路线仍以 P0 的稳定后验数据流为基础，但不把量测流降级为弱辅助。量测流的角色改为
“检察官”：它不直接生成融合状态，而是审查后验流的可信度、发现证据冲突，并参与节点
信任状态的更新。

## 1. 为什么换主线

P11/P1 暴露出的失败不是简单的 gate 设计问题，而是更根本的问题：

```text
valid = 1 的局部 EKF 节点可能已经严重漂移；
局部 covariance 可能无法真实反映漂移风险；
学习型融合模型可能把坏节点当成高可信专家；
一旦坏节点拿到极高权重，tail error 会直接爆炸。
```

因此新主线不再追求：

```text
更复杂的 soft gate
更多的 attention 堆叠
单纯的双流拼接
黑盒 MLP/GNN 直接回归融合状态
```

而追求：

```text
证据审查
信任生命周期
反例训练
稳健显式融合
风险自知
```

## 2. 目标不要过慢

本方案不采用“先做很多保守小版本，最后才尝试完整模型”的慢路线。建议用三步推进：

### Stage A：直接实现最小完整闭环

最小闭环不是 post-only P0R2，而是：

```text
post witness encoder
measurement evidence prosecutor
cross-examination module
trust state update
robust fusion layer
risk-aware training outputs
```

要求第一版就具备：

- 后验端证词建模；
- 量测端证据审查；
- 节点 trust / quarantine / recovery 状态；
- 显式稳健融合层；
- overconfidence 风险损失；
- valid-but-drift 反例增强。

### Stage B：快速跑关键失败窗口

优先验证：

- S3 seed 96；
- S3 seed 106；
- S3 seed 107；
- AOA/UWB high-error high-weight rate；
- p99 / max；
- risk 是否能在爆点时升高。

### Stage C：扩展到完整 benchmark

确认关键窗口不炸后，再扩展到：

- S1/S2/S3 nominal；
- observable degradation；
- hidden drift/recovery；
- sensor layout perturbation；
- stronger maneuver；
- swapped sensor order。

## 3. 总体结构

```text
Local EKF outputs + measurements
  x_i, P_i, valid_i, z_i, R_i, innovation_i, NIS_i, geometry_i
        |
        +-----------------------------+
        |                             |
        v                             v
Post Witness Encoder          Measurement Evidence Prosecutor
        |                             |
        v                             v
post testimony_i              evidence challenge_i
        |                             |
        +-------------+---------------+
                      v
             Cross-Examination Module
                      |
                      v
          Trust State Memory / Lifecycle
                      |
                      v
      r_i, u_i, cap_i, quarantine_i, risk_i
                      |
                      v
          Robust Explicit Fusion Layer
                      |
                      v
        x_f, P_f, confidence_t, risk_t
```

## 4. 模块定义

### 4.1 Post Witness Encoder

后验端不再叫 post-stream encoder，而叫 witness encoder。每个局部 EKF 是一个 witness。

输入：

```text
x_i
P_i
valid_i
sensor type
peer deviation
covariance shape
posterior temporal drift
```

输出：

```text
t_post_i      后验证词表示
claim_i       后验端声称的可信度
post_risk_i   后验自身风险
```

它回答：

```text
这个局部滤波器从后验状态看，像不像一个可靠证人？
```

### 4.2 Measurement Evidence Prosecutor

量测端不是弱 adapter，而是 prosecutor。它审查后验端证词。

输入：

```text
innovation_i
whitened innovation_i
NIS_i
rolling NIS_i
R_i
measurement jump
geometry / observability proxy
sensor type
```

输出：

```text
evidence_i      当前量测证据强度
challenge_i     对后验证词的质疑强度
conflict_i      post-meas 冲突程度
meas_risk_i     量测端风险
```

它回答：

```text
当前量测是否支持该局部后验？
如果不支持，是噪声、几何不可观，还是局部滤波器已经漂了？
```

### 4.3 Cross-Examination Module

这是新方案的核心交互模块。它不是简单 concat，也不是让 meas 流直接抢融合权重。

输入：

```text
t_post_i
claim_i
post_risk_i
evidence_i
challenge_i
conflict_i
peer diagnostics
```

建议第一版用 residual cross-MLP + graph context：

```text
z_i = [
    t_post_i,
    evidence_i,
    t_post_i * evidence_i,
    t_post_i - evidence_i,
    challenge_i,
    conflict_i,
    peer diagnostics
]

c_i = CrossExamGNN(z_i, {z_j})
```

后续可升级为 cross-attention，但第一版不要让 attention 直接生成融合权重。

输出：

```text
accept_i       接受该证词的倾向
discount_i     降权倾向
inflate_i      协方差膨胀倾向
isolate_i      隔离倾向
recover_i      恢复倾向
```

## 5. Trust State Memory / Lifecycle

这是相比保守 DS-RCAA 最大的升级。

每个节点维护一个信任状态：

```text
trust_i,t
mode_i,t in {normal, suspect, quarantined, recovering}
```

### 5.1 信任状态更新

```text
trust_i,t = GRU(
    trust_i,t-1,
    c_i,t,
    innovation statistics,
    peer disagreement,
    previous mode_i,t-1
)
```

或者第一版用轻量更新：

```text
trust_i,t = alpha * trust_i,t-1 + (1-alpha) * trust_obs_i,t
```

### 5.2 隔离机制

如果节点满足：

```text
high conflict
high peer deviation
low measurement support
or high posterior drift
```

则进入：

```text
suspect -> quarantined
```

在 quarantined 状态：

```text
该节点不能立刻恢复高权重；
需要连续 K 帧通过 consistency test 才能进入 recovering；
recovering 再逐步恢复 trust。
```

这比单帧 reliability 更适合 hidden drift 和 recovery 主题。

## 6. 网络输出

最终每个节点输出：

```text
r_i        节点可靠性，控制参与融合程度
u_i        协方差膨胀，修正过度自信
cap_i      evidence-conditioned weight cap
q_i        quarantine probability
trust_i    节点信任状态
```

其中：

```text
u_i = clamp(1 + softplus(a_i), 1, u_max)
cap_i = cap_min + (cap_max - cap_min) * evidence_debt_i
```

`cap_i` 取代固定 weight cap。高权重不是禁止，而是需要证据债务：

```text
如果 post 与 meas 都支持，peer 也一致，cap_i 可以高；
如果证据冲突，cap_i 必须低。
```

## 7. 融合层：不要只限于 AA-MM

AA-MM 可以作为第一版显式层，但最终建议升级为：

```text
Reliability-Calibrated Robust Barycentric Fusion
```

### 7.1 AA-MM 基础版

```text
s_i = valid_i * r_i
w_i = normalize_with_cap(s_i, cap_i)

x_f = sum_i w_i x_i

P_i_cal = u_i P_i

P_f = sum_i w_i [
    P_i_cal + (x_i - x_f)(x_i - x_f)^T
]
```

### 7.2 Robust Barycenter 增强版

状态不再是简单线性加权，而是：

```text
x_f = argmin_x sum_i r_i * rho(
    ||x - x_i||_{P_i_cal^{-1}}
)
```

其中 `rho` 可选：

```text
Huber
Geman-McClure
Tukey biweight
```

这让坏节点即使没有完全被识别，也不会线性拉爆结果。

实现上可以用少量 IRLS 迭代，保持可微：

```text
initialize x_f with AA-MM
for m = 1..M:
    robust_weight_i = psi(residual_i) / residual_i
    update x_f with reliability * robust_weight
```

第一版可先实现 AA-MM，第二版立即上 robust barycenter，不建议长期停留在线性 AA。

## 8. 反例训练：Valid-but-Drift Augmentation

不能只等仿真自然产生坏点。训练时应主动制造反例。

随机选择一个节点：

```text
valid_i = 1
x_i <- x_i + structured drift
P_i <- falsely confident covariance
innovation statistics partially plausible
```

漂移类型：

```text
constant offset
slow ramp
one-axis confident drift
range-only ambiguity drift
AOA-like angular drift
post-fault residual drift
```

目标：

```text
逼模型学会拒绝 valid 但漂移的证词。
```

这应该作为训练协议的一部分，而不是可选 trick。

## 9. 风险参与训练

`risk_t` 不只是输出展示，而要参与训练。

### 9.1 Overconfidence loss

如果模型预测低风险但实际误差大，重罚：

```text
L_overconf = 1[risk_t low] * error_t
```

更平滑地：

```text
L_overconf = exp(-risk_t) * error_t
```

### 9.2 Underconfidence loss

如果误差小却一直高风险，也要轻罚：

```text
L_underconf = risk_t * 1[error_t small]
```

核心目标：

```text
模型不仅要融合准，还要知道自己什么时候不准。
```

## 10. 传感器身份去中心化

为避免 sensor-id shortcut，网络应尽量 permutation-invariant。

建议：

```text
随机打乱节点顺序训练；
使用 sensor type embedding，但做 type dropout；
图/集合编码器使用 permutation equivariant 结构；
评估 swapped sensor order。
```

模型应学习：

```text
证据质量和几何条件
```

而不是：

```text
第几个传感器通常靠谱
```

## 11. 损失函数

第一版完整闭环建议：

```text
L = L_state
  + lambda_rank       * L_reliability_rank
  + lambda_safety     * L_bad_high_weight
  + lambda_inflation  * L_cov_inflation_prior
  + lambda_nll        * L_cov_calibration
  + lambda_overconf   * L_overconfidence
  + lambda_quarantine * L_trust_lifecycle
  + lambda_aug        * L_valid_but_drift_aug
```

### 11.1 状态损失

```text
L_state = Huber(x_f - x_true)
```

### 11.2 可靠性排序

训练时用局部真值误差：

```text
e_i = ||x_i - x_true||
e_i < e_j => r_i > r_j
```

### 11.3 坏节点高权重惩罚

```text
bad_i = 1 if local_error_i > threshold
L_bad_high_weight = mean(w_i^2 for bad_i)
```

### 11.4 covariance calibration

```text
L_nll = 0.5 * e_f^T P_f^{-1} e_f + 0.5 * logdet(P_f)
```

### 11.5 trust lifecycle loss

对于增强生成或已知故障窗口，可构造弱标签：

```text
normal / suspect / quarantined / recovering
```

监督 `mode_i,t`。

## 12. 评价指标

必须报告：

```text
RMSE
p95
p99
max
per-seed max
high-local-error top1 rate
bad node weight > 0.5 rate
AOA/UWB high-weight rate
quarantine precision / recall
risk AUROC
overconfidence rate
low-confidence coverage
swapped-order generalization
```

关键诊断窗口：

```text
S3 seed 96
S3 seed 106
S3 seed 107
```

## 13. 快速开发路线

### Milestone 1：诊断固化与反例生成

产物：

- 固化 S3 失败窗口脚本；
- valid-but-drift augmentation；
- bad-node high-weight 指标。

### Milestone 2：Skeptical Fusion v1

产物：

- Post Witness Encoder；
- Measurement Evidence Prosecutor；
- Cross-Examination Module；
- evidence-conditioned cap；
- AA-MM fusion；
- overconfidence loss。

### Milestone 3：Trust Lifecycle v2

产物：

- trust memory；
- quarantine / recovering；
- S3 recovery 分析。

### Milestone 4：Robust Barycenter v3

产物：

- 替换线性 AA 为 robust M-estimator barycenter；
- 对比 AA-MM 与 robust barycenter。

### Milestone 5：Full Benchmark

产物：

- S1/S2/S3；
- observable degradation；
- hidden drift/recovery；
- swapped sensor order；
- stronger maneuver。

## 14. 与旧方案的区别

| 维度 | 旧 DS-RCAA / P0R2 | 新 Skeptical Neural Fusion |
|---|---|---|
| 问题定义 | reliability-calibrated fusion | local filters as biased witnesses |
| 量测流 | 辅助 adapter | evidence prosecutor |
| 权重限制 | 固定 weight cap | evidence-conditioned cap |
| 时间状态 | 可选 smoothness | trust lifecycle / quarantine |
| 融合层 | AA-MM | AA-MM 起步，robust barycenter 升级 |
| 训练数据 | 自然仿真为主 | valid-but-drift 反例增强 |
| risk | 输出校准 | 参与 overconfidence 训练 |
| 叙事 | 可靠性校准 | 证据审查与怀疑式融合 |

## 15. 论文叙事

推荐问题定义：

```text
In heterogeneous multi-sensor tracking, local filters should be treated as
fallible witnesses rather than reliable observations. Under weak observability
and maneuvering, a local posterior can remain valid while becoming severely
drifted. We propose a skeptical neural fusion framework that cross-examines
posterior testimony with measurement evidence, maintains sensor trust states,
and performs robust explicit fusion with calibrated confidence.
```

中文表达：

```text
在异构多传感器跟踪中，局部滤波器不是绝对可靠的观测源，而是可能带偏见的证词。
弱观测和机动条件下，局部后验可能在 valid=1 的情况下严重漂移。本文提出一种
怀疑式神经融合框架，通过量测证据审查后验证词，维护传感器信任生命周期，并结合
稳健显式融合层与风险校准，抑制 valid-but-drift 节点导致的 tail error。
```

## 16. 一句话版本

```text
不要把融合器做成权重分配器；
把它做成审判系统：
后验是证词，量测是证据，网络负责质询和维护信任状态，
显式稳健融合层负责裁决最终状态，risk head 负责承认自己何时可能错。
```
