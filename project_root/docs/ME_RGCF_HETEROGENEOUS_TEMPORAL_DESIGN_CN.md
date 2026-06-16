# ME-RGCF：第二阶段异构图 + 时间记忆融合架构设计

> 2026-06-16 更新：`ME-RGCF-A0` 最小异构图版本已经实现为可选方法
> `me_rgcf_a0` / `ME-RGCF-A0`。当前实现只包含 3 个 P 节点 + 5 个 M
> 节点的 M-M、M->P、P-P 图推理，不包含时间记忆、不包含污染场景、不包含
> `delta_x`。默认主线仍是 `Phase1R RGCF`，ME-A0 需要在 benchmark 中显式
> 使用 `--methods me-a0` 或 `--methods rgcf,me-a0` 启用。

## 0. 阶段定位

本文档描述的是 **Phase2 设计方向**，不是当前 GPU 主实验的默认实现。

当前主线仍然是 `Phase1R RGCF`：

- 先使用 `phase1r_basic_3track_2evidence_nominal` 和 `phase1r_maneuver_3track_2evidence_nominal` 两个 clean 场景。
- 先验证 `3 个 track posterior sensors + 2 个 evidence measurement sensors` 的纠正信息流。
- 默认对比 `single-T1/T2/T3`、`AVG-3T`、`WAA-MM-3T`、`CI-3T` 和 `RGCF`。
- 当前 `Phase1RRGCF` 保持为主实验模型，用于建立稳定 baseline。

`ME-RGCF` 是在 Phase1R 结果稳定后推进的第二阶段创新方向，目标是把当前的 evidence pooling 升级为 **measurement nodes as graph nodes**，并进一步引入时间记忆来刻画量测信号与后验信号的时序错位。

阶段边界：

| 阶段 | 默认模型 | 目标 | 是否当前 GPU 主实验 |
| --- | --- | --- | --- |
| Phase1R | `Phase1RRGCF` | 纠正传感器角色，验证无污染 clean 闭环 | 是 |
| Phase2 / ME-A0 | `ME-RGCF` without temporal memory | 验证异构图中 M 节点独立身份是否带来收益 | 否 |
| Phase2 / ME-A1+ | `ME-RGCF` with temporal memory | 验证量测短窗和后验长窗的时序互补价值 | 否 |
| Phase3 | ME-RGCF under fault scenarios | 引入污染/故障后验证鲁棒性 | 否 |

推进原则：

1. 不用 ME-RGCF 替代当前 Phase1R 主线。
2. 不在 Phase1R 正式 GPU 实验中默认运行 ME-RGCF。
3. 先完成 Phase1R RGCF 的正式 clean benchmark，再把 ME-RGCF 作为第二阶段消融扩展。
4. ME-RGCF 的 M 节点只参与消息传递，不输出融合权重，不直接输出状态修正。

## 1. 设计动机

### 1.1 当前 RGCF 的两个结构局限

当前 `Phase1RRGCF` 的 GNN 是一个**同构图自注意力层**——3 个 track 节点全连接，evidence 被池化广播为全局上下文：

```text
evidence ─→ pool ─→ broadcast ─→ [same context for every P node]
                                              │
  P1 ←──→ P2 ←──→ P3                          │
   │       │       │                           │
  P1 sees evidence the same way P2 and P3 do.  │
```

两个局限：

**(1) 量测和后验被"预混"**：`post_enc` 和 `meas_enc` 的输出在进入 GNN 之前就通过 `cross(concat(...))` 混成一个 64 维向量。量测没有独立的图身份——它只是后验节点的一个属性。M 之间无法互相比对，M 无法针对特定 P 发送定向信号。

**(2) 无时间记忆**：每个时刻的 GNN 推理是独立的。量测尖峰（innov 剧烈变化）和后验滞后（xhat 缓慢漂移）之间的时间错位无法被利用。EKF 在深度污染阶段会"适应"偏置（NIS 回落到正常范围），逐帧快照无法区分"正常"和"被污染后自洽"。

### 1.2 核心洞察：量测信号和后验信号的时序错位

```
污染注入 (t=50) 时：
  量测 (innov/NIS)：尖峰，立即报警
  后验 (xhat error)：几乎未变，K 稀释了单步修正量

污染稳态 (t=60) 时：
  量测 (innov/NIS)：回到正常范围（EKF 已"适应"偏置）
  后验 (xhat error)：深度偏离，但量测已经沉默

两种信号的退化曲线是错位的：
  - 量测信号是微分信号（变化检测），快但短
  - 后验信号是积分信号（累积偏离），慢但稳
  - 单靠任何一种都不够——需要联合时序对比
```

### 1.3 解决方向

| 局限 | 解决方向 |
|---|---|
| 量测没有独立图身份 | 异构图：M 节点和 P 节点作为两类节点共存于图中 |
| M 无法定向给 P 发信号 | M→P cross attention：每个 P 单独去 M 层查询 |
| 无时间记忆 | 双窗口时间编码：M 短窗（5步）捕捉突变，P 长窗（20步）感知漂移 |
| 逐帧无法识别 EKF 适应 | 跨时间对比特征：量测短期波动 vs 后验中期漂移的比值 |

---

## 2. 整体架构

### 2.1 图结构

每个时刻构建一张异构图，包含 **3 个 P 节点** + **5 个 M 节点**：

```text
  M1 ─── M2 ─── M3 ─── M4 ─── M5       ← Measurement layer (5 nodes)
   │  \   │  \   │  \   │  \   │
   │   \  │   \  │   \  │   \  │        ← M→P cross attention
   v    v v    v v    v v    v v
  P1 ─── P2 ─── P3                       ← Posterior layer (3 nodes)
   │      │      │
   w1     w2     w3                      ← Fusion weights (P-only output)
```

| 节点类型 | 数量 | 来源 | 携带信息 | 是否有输出头 |
|---|---|---|---|---|
| P (Posterior) | 3 | T1/T2/T3 EKF 后验 | `xhat, logPdiag, valid, sensor_type` | 是：`w_i, s_i` |
| M (Measurement) | 5 | M1-M3 来自 track 量测<br>M4-M5 来自 evidence 量测 | `z, R, innov/NIS(仅M1-M3), geometry, residual` | 否 |

三类边：

| 边类型 | 方向 | 参数 | 语义 |
|---|---|---|---|
| M→M | 量测节点之间 | `W_MM` | 量测一致性比对——"两个量测对同一目标，故事一致吗？" |
| M→P | 量测→后验 | CrossAttn(`Q=P, K/V=M`) | 量测证据校准后验可靠性——"M 层告诉我，我的后验有问题" |
| P→P | 后验节点之间 | `W_PP` | 后验协商融合权重——"我俩一致，他偏了，给他降权" |

不做 P→M 边——避免后验信息回流污染量测节点的纯净性。

### 2.2 数据流全景

```text
                    特征提取阶段                    模型阶段
                    ────────────                    ────────

  T1 EKF ──→ post_1, meas_1 ──→ P1 node feat ──→ PosteriorEncoder ──→ h_P_1
  T2 EKF ──→ post_2, meas_2 ──→ P2 node feat ──→ PosteriorEncoder ──→ h_P_2
  T3 EKF ──→ post_3, meas_3 ──→ P3 node feat ──→ PosteriorEncoder ──→ h_P_3

  M1(track meas) ──────────────→ M1 node feat ──→ MeasurementEncoder ──→ h_M_1
  M2(track meas) ──────────────→ M2 node feat ──→ MeasurementEncoder ──→ h_M_2
  M3(track meas) ──────────────→ M3 node feat ──→ MeasurementEncoder ──→ h_M_3
  M4(evidence AOA) ────────────→ M4 node feat ──→ MeasurementEncoder ──→ h_M_4
  M5(evidence UWB) ────────────→ M5 node feat ──→ MeasurementEncoder ──→ h_M_5

  所有 h_P_i, h_M_j ∈ R^64

  + TypeEmbedding(P) or TypeEmbedding(M) added to each node

                    ┌─────── Temporal Memory ───────┐
                    │  M nodes: short GRU (L=5)      │
                    │  P nodes: long GRU  (L=20)     │
                    │  Cross-time contrast features  │
                    └────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        Stage 1: M-M GNN    Stage 2: M→P     Stage 3: P-P GNN
        (量测互相比对)       (量测校准后验)    (后验协商融合)
              │               │               │
              ▼               ▼               ▼
         h_M' [5,64]    h_P^0 [3,64]    h_P' [3,64]
                              │               │
                              └───────────────┘
                                      │
                              ┌───────┼───────┐
                              ▼       ▼       ▼
                          w_1,s_1  w_2,s_2  w_3,s_3
                              │       │       │
                              └───────┴───────┘
                                      │
                              x_f = Σ α_i · xhat_i
```

---

## 3. 节点编码

### 3.1 节点特征定义

**P 节点输入** `post_feat_i ∈ R^9`（与当前 Phase1RRGCF 相同）：

| 索引 | 内容 | 说明 |
|---|---|---|
| 0:4 | `xhat / scale` | 归一化 EKF 后验 (px, py, vx, vy) |
| 4:8 | `log(P_diag)` | 协方差对角元取 log |
| 8 | `valid` | 传感器当前有效标志 |

**M 节点输入** `meas_feat_j ∈ R^me`：

对 track 量测节点 (M1-M3)，与当前 `meas_feat` 的 18 维一致：

| 索引 | 内容 |
|---|---|
| 0:2 | `tanh(white_innov / 5)` — 白化 innovation |
| 2 | `log1p(NIS).clamp(0,5)` |
| 3:5 | `log1p(R_diag).clamp(0,8)` |
| 5:8 | pred-sensor 几何 (dx, dy, range) |
| 8:12 | sensor_type one-hot |
| 12 | valid |
| 13 | rolling_mean(log_nis, 30) |
| 14 | rolling_mean(abs_innov, 30) |
| 15:17 | padding |

对 evidence 量测节点 (M4-M5)，与当前 `evidence_feat` 的 16 维一致：

| 索引 | 内容 |
|---|---|
| 0:2 | `tanh(z_white / 10)` — 白化量测值 |
| 2:4 | `log1p(R_diag).clamp(0,8)` |
| 4:8 | sensor_type one-hot |
| 8:10 | sensor_pos / pos_scale |
| 10 | `log1p(residual_to_prior).clamp(0,6)` |
| 11 | `log1p(residual_to_track_mean).clamp(0,6)` |
| 12 | `log1p(residual_to_track_min).clamp(0,6)` |
| 13 | valid |
| 14:16 | padding + index |

### 3.2 编码器设计

```python
class PosteriorEncoder(nn.Module):
    """P 节点编码器：EKF 后验 → GNN embedding"""
    def __init__(self, in_dim=9, hidden_dim=64):
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
    def forward(self, post_feat):  # [K, 3, 9] → [K, 3, 64]
        return self.net(post_feat)


class MeasurementEncoder(nn.Module):
    """M 节点编码器：量测特征 → GNN embedding（统一处理 track meas 和 evidence）"""
    def __init__(self, in_dim=18, hidden_dim=64):
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
    def forward(self, meas_feat):  # [K, 5, 18] → [K, 5, 64]
        return self.net(meas_feat)
```

**track M 节点和 evidence M 节点使用同一个 `MeasurementEncoder`**。如果输入维度不一致（18 vs 16），在进入编码器前用零填充对齐到 18 维，保证共享参数。

### 3.3 Type Embedding

每个节点加上可学习的类型标记：

```python
self.type_embed = nn.Embedding(2, 64)  # 0=P, 1=M

h_P_i = post_enc(post_i) + self.type_embed(0)   # [K, 3, 64]
h_M_j = meas_enc(meas_j)  + self.type_embed(1)   # [K, 5, 64]
```

网络习得 P 和 M 的语义差异，后续所有模块（M-M message passing、M→P cross attention 等）在此基础上运作。

---

## 4. 时间记忆模块

### 4.1 双窗口设计

```
M 节点（量测）：短窗口 — 5 步
    目的：捕捉 innov 尖峰、量测噪声的瞬时变化
    原因：量测信号变化快（微分信号），长窗口反而平滑掉有用信号

P 节点（后验）：长窗口 — 20 步
    目的：感知后验的缓慢漂移、与 peer 的持续偏离
    原因：后验变化慢（积分信号），短窗口看不到趋势
```

### 4.2 时间编码器

```python
class TemporalMemoryModule(nn.Module):
    """为每个节点维护时间记忆，输出时间增强的节点表示"""
    def __init__(self, hidden_dim=64):
        # M 节点：短窗口 GRU
        self.m_temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        # P 节点：长窗口 GRU
        self.p_temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, h_M_seq, h_P_seq):
        # h_M_seq: [B, L_short=5, 5, 64] — 最近 5 帧的 M embedding
        # h_P_seq: [B, L_long=20, 3, 64] — 最近 20 帧的 P embedding

        # 并行处理所有节点
        B, Lm, Nm, D = h_M_seq.shape
        _, h_M_last = self.m_temporal(h_M_seq.reshape(B*Nm, Lm, D))
        h_M_t = h_M_last.squeeze(0).reshape(B, Nm, D)          # [B, 5, 64]

        B, Lp, Np, D = h_P_seq.shape
        _, h_P_last = self.p_temporal(h_P_seq.reshape(B*Np, Lp, D))
        h_P_t = h_P_last.squeeze(0).reshape(B, Np, D)          # [B, 3, 64]

        return h_M_t, h_P_t
```

### 4.3 跨时间对比特征

时间记忆模块不仅提供序列编码后的表示，还输出显式的统计对比特征：

```python
def compute_cross_time_features(h_M_seq, h_P_seq, meas_raw, post_raw):
    """
    从原始特征序列中提取时间统计量。

    h_M_seq: [B, Lm, 5, 64] — M 节点 embedding 序列
    h_P_seq: [B, Lp, 3, 64] — P 节点 embedding 序列
    meas_raw: [B, Lm, 5, 18] — M 原始特征序列
    post_raw: [B, Lp, 3, 9] — P 原始特征序列

    返回: cross_time_feat [B, 3, 5]  (per P node × per feature)
    """
    features = {}

    # ── 量测侧 — 短期统计 (window=5) ──
    # innov 最近是否剧烈波动
    meas_short_volatility = std(innov_t-4:t)      # [B, 3]
    # 当前 innov 是处于波峰还是已回落
    meas_short_trend = innov_t - mean(innov_t-4:t) # [B, 3]

    # ── 量测侧 — 中期统计 (window=20) ──
    # NIS 近期是否持续偏高
    meas_long_mean_nis = mean(NIS_t-19:t)          # [B, 3]

    # ── 后验侧 — 中期统计 (window=20) ──
    # 后验与 peer median 的近期平均偏离
    post_peer_drift = mean(|xhat_i - peer_median|_t-19:t)  # [B, 3]

    # ── 交叉对比特征（关键） ──
    # 量测短期波动 / (后验中期漂移 + eps)
    # > 1: 量测剧烈变化但后验跟不上 → 刚被污染，预警
    # < 1: 量测和后验都稳定，或都已"适应"
    meas_post_divergence = meas_short_volatility / (post_peer_drift + 1e-6)  # [B, 3]

    return {
        "meas_short_volatility": meas_short_volatility,
        "meas_short_trend": meas_short_trend,
        "meas_long_mean_nis": meas_long_mean_nis,
        "post_peer_drift": post_peer_drift,
        "meas_post_divergence": meas_post_divergence,
    }
```

`meas_post_divergence` 是核心特征——在污染刚注入时最大（分子大分母小），深度污染后变小（分子小分母大），污染解除时再次变大（二次尖峰）。一个标量就编码了"量测vs后验时序错位"。

---

## 5. 消息传递三阶段

### 5.1 Stage 1: M-M 量测层自校准

```
输入：h_M ∈ R^{5×64} + temporal context
输出：h_M' ∈ R^{5×64}
目的：量测节点之间比对一致性，形成统一的量测层判断
```

```python
class MMGraphLayer(nn.Module):
    """量测层消息传递：5 个 M 节点自注意力"""
    def __init__(self, hidden_dim=64):
        self.attn = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.upd = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )

    def forward(self, h_M):
        # h_M: [B, 5, 64]
        hi = h_M.unsqueeze(2).expand(-1, -1, 5, -1)     # [B, 5, 5, 64]
        hj = h_M.unsqueeze(1).expand(-1, 5, -1, -1)
        e = self.attn(torch.cat([hi, hj], -1)).squeeze(-1)   # [B, 5, 5]
        e = e + torch.eye(5, device=e.device).unsqueeze(0) * (-1e9)  # no self-loop
        a = torch.softmax(e, dim=2)
        h_agg = torch.matmul(a, h_M)                           # [B, 5, 64]
        h_M_prime = self.upd(torch.cat([h_M, h_agg], -1))      # [B, 5, 64]
        return h_M_prime, a
```

**M-M 做什么**：例如，M4 (AOA 远距离低精度) 和 M5 (UWB 近距离高精度) 交流后，UWB 在 M 层的话语权自然更大。如果 M2 的 innov 异常而 M4/M5 的 evidence 正常，M 层形成一个内部共识："M2 的量测信号不可靠"。

### 5.2 Stage 2: M→P 跨层校准

```
输入：h_P ∈ R^{3×64}, h_M' ∈ R^{5×64}
输出：h_P^0 ∈ R^{3×64}（吸收了 M 层证据的后验表示）
目的：每个 P 节点独立去 M 层查询"证据如何看待我的后验"
```

```python
class MPCrossAttention(nn.Module):
    """M→P cross attention：P 为 query，M 为 key/value"""
    def __init__(self, hidden_dim=64):
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        self.fuse = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.scale = hidden_dim ** 0.5

    def forward(self, h_P, h_M_prime):
        # h_P: [B, 3, 64], h_M_prime: [B, 5, 64]
        Q = self.W_Q(h_P)                                    # [B, 3, 64]
        K = self.W_K(h_M_prime)                              # [B, 5, 64]
        V = self.W_V(h_M_prime)                              # [B, 5, 64]

        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, 3, 5]
        attn = torch.softmax(attn, dim=-1)

        context = torch.matmul(attn, V)                      # [B, 3, 64]
        h_P_enriched = self.fuse(torch.cat([h_P, context], -1))  # [B, 3, 64]

        return h_P_enriched, attn
```

**M→P 做什么**：和当前 Phase1RRGCF 的 evidence 池化广播不同，cross attention 让 P1 和 P2 看到**不同的 M 层信号**：

```text
P1 的 attention weights 可能集中在 M1/M3/M5（支持 P1 的证据）
P2 的 attention weights 可能集中在 M2（P2 自己的量测）和 M4/M5（外部证据）
P3 的 attention weights 同理

如果 M4 (AOA) 的 residual 显示 "P2 偏了"：
  P2 的 attention 会在 M4 上权重偏高 → P2 收到"你可能有问题的"信号
  P1 和 P3 的 attention 不会在 M4 上有异常权重
```

### 5.3 Stage 3: P-P 后验协商

```
输入：h_P^0 ∈ R^{3×64} + cross-time contrast features
输出：h_P' ∈ R^{3×64}
目的：后验节点之间协商融合权重
```

与当前 RGCF 的 `_graph()` 同构，但增加了时间对比特征的注入：

```python
class PPGraphLayer(nn.Module):
    """后验层消息传递：3 个 P 节点自注意力"""
    def __init__(self, hidden_dim=64, time_feat_dim=5):
        self.attn = nn.Sequential(
            nn.Linear(2 * hidden_dim + time_feat_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.upd = nn.Sequential(
            nn.Linear(2 * hidden_dim + time_feat_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )

    def forward(self, h_P_0, time_features):
        # h_P_0: [B, 3, 64]
        # time_features: [B, 3, 5] — 跨时间对比特征

        hi = h_P_0.unsqueeze(2).expand(-1, -1, 3, -1)
        hj = h_P_0.unsqueeze(1).expand(-1, 3, -1, -1)

        # 时间特征拼入 pairwise attention
        ti = time_features.unsqueeze(2).expand(-1, -1, 3, -1)
        tj = time_features.unsqueeze(1).expand(-1, 3, -1, -1)
        t_diff = torch.abs(ti - tj)

        e = self.attn(torch.cat([hi, hj, t_diff], -1)).squeeze(-1)
        e = e + torch.eye(3, device=e.device).unsqueeze(0) * (-1e9)
        a = torch.softmax(e, dim=2)

        h_agg = torch.matmul(a, h_P_0)
        h_P_prime = self.upd(torch.cat([h_P_0, h_agg, time_features], -1))
        return h_P_prime, a
```

**P-P 做什么**：P1 和 P3 发现彼此后验接近，P2 偏离且 `meas_post_divergence` 高 → 压低 P2 的 attention weight → P2 被"投票"降权。结合时间特征，网络可以学到"量测短期剧烈 + 后验与 peer 偏离 = 刚被污染，暂时压制"的模式。

---

## 6. 输出头与融合

### 6.1 输出头（只挂在 P 节点上）

```python
class OutputHeads(nn.Module):
    def __init__(self, hidden_dim=64):
        # 基础融合权重 logits
        self.node_logit = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # 可靠性自我评估
        self.reliability_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # 协方差校准
        self.cov_calib = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_P_prime):
        base_logits = self.node_logit(h_P_prime).squeeze(-1)          # [B, 3]
        reliability = torch.sigmoid(self.reliability_head(h_P_prime).squeeze(-1))  # [B, 3]
        cov_scale = (1.0 + F.softplus(self.cov_calib(h_P_prime).squeeze(-1))).clamp(max=25.0)

        return base_logits, reliability, cov_scale
```

### 6.2 最终融合权重

```python
final_logits = base_logits / temperature        # temperature=1.5
             + alpha * log(reliability + eps)   # alpha=1.0
             - beta  * log(cov_scale + eps)     # beta=0.35

alpha_i = softmax(final_logits)                 # 只在 P 节点上做 softmax
```

### 6.3 融合公式

与当前 RGCF 一致，使用 `info_diag`：

```python
x_f = Σ_i (alpha_i / P_diag_i * xhat_i) / Σ_i (alpha_i / P_diag_i)
```

M 节点全程不产生 alpha、不产生 delta_x、不参与融合。

---

## 7. 损失函数

### 7.1 损失总览

```text
L = L_state                      # 主监督
  + λ_oracle L_weight_oracle     # 后验可靠性 oracle
  + λ_tail   L_tail              # 尾部误差加权
  + λ_cov    L_cov_calibration   # 协方差校准
  + λ_meas   L_meas_consistency  # 量测一致性辅助（新）
  + λ_delta  L_delta_reg         # evidence correction 正则（A4 阶段）
```

### 7.2 L_state — 状态监督

```python
L_state = MSE(x_f[:, 0:2], truth[:, 0:2])          # position MSE
        + 0.2 * MSE(x_f[:, 2:4], truth[:, 2:4])    # velocity MSE
```

### 7.3 L_weight_oracle — 后验权重 oracle

与 Phase1RRGCF 的 gate target 一致——用真实位置误差生成 soft target：

```python
r_i* = exp(-||xhat_i[:2] - truth[:2]|| / tau)     # tau = 20m
L_weight_oracle = MSE(reliability_i, r_i*)          # 只在 P 节点上计算
```

### 7.4 L_tail — 尾部误差加权

```python
pos_err = ||x_f[:2] - truth[:2]||
tail_mask = pos_err > 25.0
L_tail = mean(pos_err[tail_mask] ** 2)              # 只对大误差样本加权
```

### 7.5 L_cov_calibration — 协方差校准

```python
L_cov = mean((cov_scale_i - 1.0)**2)                # 倾向 cov_scale=1（不膨胀）
      + margin * max(0, 1.0 - (cov_scale_fault_mean - cov_scale_normal_mean))
```

### 7.6 L_meas_consistency — 量测一致性辅助（新增）

不强迫 M 节点输出状态估计，只监督它学会判断量测一致性：

```python
# 在 M-M 层之后，M 节点的 embedding 应反映量测一致性
# 对同一传感器的量测，近期 NIS 异常的 M 节点和正常的 M 节点应有区分度

# 方法：用 M 层 attention 矩阵的熵做稀疏性奖励
L_meas_consistency = -mean(entropy(M_attn_weights))
# 鼓励 M 层注意力集中在信息量大的量测节点上，而非均匀分布
```

### 7.7 L_delta_reg — Evidence Correction 正则（A4 可选）

```python
# 仅在 A4 阶段启用
L_delta_reg = ||delta_x_evidence||_2
# 强制 evidence 对状态的修正量不能主导融合结果
```

### 7.8 损失权重

| 损失项 | 权重 | 阶段 |
|---|---|---|
| L_state | 1.0 | 全部 |
| L_weight_oracle | 0.05 | 全部 |
| L_tail | 0.02 | 全部 |
| L_cov_calibration | 0.01 | 全部 |
| L_meas_consistency | 0.005 | A2+ |
| L_delta_reg | 0.005 | A4 only |

---

## 8. 消融路径

### 8.1 建议消融顺序

当前 Phase1RRGCF 作为基线 (A1/A2/A3)，ME-RGCF 分阶段引入：

```
A1: RGCF-3T (当前 Phase1RRGCF 同构图，无 evidence，无时间记忆)
 │
A2: + clean track meas → 同构图 + meas_feat
 │
A3: + evidence broadcast → 同构图 + evidence 池化广播（当前 Phase1RRGCF）
 │
ME-A0: 异构图（无时间记忆）
   └─ 新增 M 节点、三类边、M→P cross attention
   └─ 验证"量测作为独立图节点"本身是否有增益
 │
ME-A1: + 时间记忆模块
   └─ 双窗口 GRU + 跨时间对比特征
   └─ 验证"量测-后验时序错位"的预警价值
 │
ME-A2: + 污染场景
   └─ 引入 bias_ramp / jump / dropout 故障
   └─ 验证异构图 + 时间记忆在故障场景下的全时段覆盖优势
 │
ME-A3: + evidence correction (delta_x)
   └─ M 节点输出小幅状态修正
   └─ 验证强约束下 evidence 的修正价值
```

### 8.2 每阶段预期收益

| 消融步 | 新增能力 | 预期最大增益场景 |
|---|---|---|
| ME-A0 异构图 | M 节点独立身份，per-P cross attention | evidence 与个别 track 冲突时（定向信号） |
| ME-A1 时间记忆 | 量测尖峰预警，EKF 适应检测 | 污染早期 (t=50-52) 和污染解除 (t=80) |
| ME-A2 污染 | 全时段故障覆盖（M 短窗 + P 长窗互补） | 任何有污染的鲁棒性场景 |
| ME-A3 delta_x | evidence 产生受约束的修正 | evidence 精度高但 track 均偏离的场景 |

---

## 9. 与当前 RGCF 的对比总结

| 维度 | 当前 Phase1RRGCF（Phase1R-A3） | ME-RGCF（异构图 + 时间记忆） |
|---|---|---|
| **图节点数** | 3 (P-only) | 8 (3P + 5M) |
| **节点类型** | 同构 | 异构，type embedding 区分 |
| **边类型** | 1 种 (P→P) | 3 种 (M→M, M→P, P→P)，各有独立 W |
| **evidence 注入** | 全局 softmax 池化 → broadcast | 独立 M 节点 + per-P cross attention |
| **M 能否定向给 P 发信号** | 不能 | 能（cross attention 按 P_i 分别计算） |
| **时间记忆** | 无 | M: 短窗口 GRU (5步), P: 长窗口 GRU (20步) |
| **跨时间对比** | 无 | meas_post_divergence 等统计特征 |
| **GNN 阶段数** | 1（单轮 self-attention） | 3（M-M → M→P → P-P） |
| **输出约束** | gate/logit 只在 P 上 | 同，M 节点无输出头 |
| **参数增量** | 基线 | +M 编码器 + 三类边参数 + GRU + cross attention |
| **适用场景** | nominal, 传感器数量少, 快速迭代 | 污染/故障, 传感器异构, 需要细粒度可解释性 |

---

## 10. 训练协议

### 10.1 数据规模

```text
train seeds: 10-69 (60 seeds)
val seeds:   70-89 (20 seeds)
test seeds:  90-109 (20 seeds)

场景：S1R (basic) + S2R (maneuver) mixed training
污染：ME-A0/ME-A1 阶段使用 clean；ME-A2+ 引入 bootstrap / random_window 故障
```

### 10.2 训练参数

```text
epochs:      80
batch_size:  64
lr:          1e-3, ReduceLROnPlateau(factor=0.5, patience=4)
weight_decay: 1e-5
grad_clip:   1.0
hidden_dim:  64
model_seeds: 0-4
```

### 10.3 两阶段训练策略

当引入时间记忆模块后，建议分两阶段训练：

```text
Phase A (warmup, epochs 1-20):
  冻结 M-M 和 M→P 模块的 GRU
  只用当前帧的瞬时特征训练 P-P GNN + 输出头
  目的：先学会基本的融合权重分配

Phase B (full, epochs 21-80):
  解冻全部模块
  联合训练 GRU + M-M + M→P + P-P
  目的：时间记忆模块学习捕捉时序模式
```

### 10.4 验收标准

| 检查项 | 通过标准 |
|---|---|
| M 节点不输出权重 | M 节点对应的 embedding 不进入 `final_logits` 和 `_decode` |
| M 节点参与消息传递 | M-M attention 矩阵非均匀（有区分度） |
| 时间记忆有效性 | `meas_post_divergence` 与真实污染起始时间对齐 |
| ME-A0 vs 当前 A3 | RMSE 不退化（异构不带来负面影响） |
| ME-A1 vs ME-A0 | 污染早期 (t=50-55) 的 P99 error < ME-A0 |
| ME-A2 vs ME-A1 | 全故障时段 Max error < ME-A1 |
