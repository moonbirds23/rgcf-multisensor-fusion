README：NF-DKF 融合分支当前研究状态与后续方向
1. 当前分支定位

当前研究聚焦于 NF-DKF 中的融合层（Fusion Layer），暂时不扩展到完整滤波框架的全面重构。
核心目标是：

先把 original 融合器 跑稳定；

明确当前训练/测试设置中“故障先验泄漏”的问题；

在 clean 场景 下建立可靠 baseline；

再逐步引入更强的泛化实验与结构增强。

当前的基本判断是：

现有 original 版本的节点输入主要来自 局部 EKF 后验信息；

这类输入虽然统一、稳定，但会弱化不同传感器在量测空间中的结构差异；

因此后续有必要考虑引入 量测异构流（measurement heterogeneous stream）；

但在此之前，必须先让 clean original baseline 稳定可复现。

2. 当前代码阶段的明确结论
2.1 original 当前本质

当前 original 融合器的输入，是每个节点统一格式的后验特征，核心是：

局部状态估计 
𝑥
^
x
^

协方差对角项 
d
i
a
g
(
𝑃
)
diag(P)

valid

这意味着 current original 更像一个：

后验状态级融合器

而不是直接利用原始量测信息的异构融合器

2.2 当前训练/测试存在的问题

现有旧版本训练与测试默认都在 带故障/污染的配置 下运行，因此存在以下问题：

训练阶段已经看过故障模式

如 dropout window

如 jump bias / pollution

因此测试结果更像：

同分布故障泛化

而不是 未知故障泛化

在这种前提下继续改结构，会导致：

难以判断模型鲁棒性来自结构本身

还是来自训练中已经暴露过故障分布

3. 当前阶段的首要任务
核心任务

先把 original clean 版本 跑稳定，作为后续一切扩展的基线。

当前明确策略

已经决定：

暂时不在“训练和测试都带污染”的旧版本上继续扩展；

先建立 完全 clean 的 original 版本；

训练、验证、测试默认都在 clean 分布下；

用它观察：

loss 是否稳定

多 seed 是否稳定

权重是否合理

baseline 是否可复现

当前对 clean 的定义

这里的 clean 指：

没有显式 dropout

没有显式 jump bias 污染

但仍保留多传感器异构性，例如：

GPS 类节点

Radar 类节点

不同 nominal noise level

换言之，clean 不是“完全同质化”，而是“无额外故障注入”。

4. 关于 valid 的当前认识
4.1 original 中的 valid

当前 original 中的 valid 只是一个最基础的 availability 标记，表示：

当前时刻该传感器是否成功提供量测

是否参与局部 EKF 更新

它不是：

可信度

质量分数

可学习 reliability

4.2 当前判断

后续不建议直接把 valid 这个物理语义变量完全神经化替换。
更合理的方向是：

保留硬 valid 作为 availability

另行设计可学习的 reliability gate

即后续若做门控，更推荐：

𝑔
=
𝑣
𝑎
𝑙
𝑖
𝑑
×
𝑟
𝑒
𝑙
𝑖
𝑎
𝑏
𝑖
𝑙
𝑖
𝑡
𝑦
g=valid×reliability

其中：

valid：硬可用性

reliability：连续、可学习、上下文相关的可信度

5. 当前保留的核心研究想法分支

下面这些想法都保留，但有先后顺序，不会同时全部推进。

分支 A：训练 clean，测试 fault

这是后续最优先的泛化实验方向之一。

目标是回答：

模型是否真的具备故障外推能力？

还是只是在训练时见过故障模式？

后续准备考虑的设置包括：

训练正常，测试 dropout

训练正常，测试 pollution

训练正常，测试 dropout + pollution

固定时间窗训练，随机时间窗测试

当前判断：
这类实验必须建立在 clean original 已稳定 的前提下再做。

分支 B：训练一类轨迹，测试类似轨迹

目标是检验：

模型是否记住了轨迹模板

还是学到了更一般的融合规律

当前判断：

先做“同机动族、不同参数”的泛化

暂时不急着做大跨度 OOD 轨迹

例如后续会考虑随机化：

初始速度

初始朝向

转弯角速度

机动持续时间

扰动幅度/周期

分支 C：双流特征设计

这是当前最重要的结构构想之一，但不是立刻开工的内容。

核心思想是：
单纯依赖 EKF 后验特征，会把不同传感器的量测差异“压平”。
因此后续考虑引入：

1）后验状态流（post-stream）

负责统一状态语义，保留 current original 的主干：

𝑥
^
x
^

d
i
a
g
(
𝑃
)
diag(P)

valid

2）量测异构流（meas-stream）

负责保留量测空间中的传感器差异和可靠性信息，例如：

innovation

NIS

measurement noise scale

sensor type embedding

geometry-related features

当前判断：

双流设计是合理方向；

但第一步应是 先验证 meas-stream 本身是否有效；

不应一开始就同时叠加 gate、时序模块、大量 raw history。

分支 D：Selective Sensor Fusion 思路迁移

已确认 Selective Sensor Fusion 论文的重要启发有三点：

不同来源特征的可靠性是动态变化的；

可以对某一类特征流引入 soft gate / hard gate；

gate 的生成应由多源上下文共同决定，而不是单流自评。

当前迁移判断：

若参考该论文，最适合被 gate 的对象是 meas-stream

不建议一开始 gate 整个节点总表示

更不建议先动稳定主干 post-stream

推荐的后续方向是：

ℎ
𝑝
𝑜
𝑠
𝑡
=
𝜙
𝑝
𝑜
𝑠
𝑡
(
𝑓
𝑝
𝑜
𝑠
𝑡
)
h
post
=ϕ
post
	​

(f
post
)
ℎ
𝑚
𝑒
𝑎
𝑠
=
𝜙
𝑚
𝑒
𝑎
𝑠
(
𝑓
𝑚
𝑒
𝑎
𝑠
)
h
meas
=ϕ
meas
	​

(f
meas
)
𝑔
𝑚
𝑒
𝑎
𝑠
=
𝜎
(
𝜓
(
[
ℎ
𝑝
𝑜
𝑠
𝑡
;
ℎ
𝑚
𝑒
𝑎
𝑠
]
)
)
g
meas
=σ(ψ([h
post
;h
meas
]))
ℎ
~
𝑚
𝑒
𝑎
𝑠
=
𝑔
𝑚
𝑒
𝑎
𝑠
⊙
ℎ
𝑚
𝑒
𝑎
𝑠
h
~
meas
=g
meas
⊙h
meas

然后再将：

[
ℎ
𝑝
𝑜
𝑠
𝑡
;
ℎ
~
𝑚
𝑒
𝑎
𝑠
]
[h
post
;
h
~
meas
]

送入最终融合器。

但当前判断依然是：

先验证双流，再验证 gated dual-stream

gate 不是第一步

分支 E：可学习 reliability / confidence

当前 confidence 版本中使用的置信度，本质上仍是 手工构造规则，主要基于：

valid

协方差

NIS

miss_count

后续长期方向之一是：

将 rule-based confidence 升级为 learned reliability predictor

可能通过门控网络或辅助监督方式实现

但当前暂不直接上这一模块，原因是：

original baseline 尚未 clean 稳定；

若过早引入 learned confidence，会和故障分布、双流结构、gate 作用混在一起；

难以进行干净的归因分析。

6. 当前建议的研究推进顺序

后续默认按以下顺序理解本项目该分支：

第一步：稳定 clean original

目标：

让 baseline 稳定

让训练/验证/测试全 clean

保证多 seed 可复现

第二步：做 clean train / fault test

目标：

检验故障泛化

分清鲁棒性来自哪里

第三步：做轨迹族泛化

目标：

检验是不是记住了轨迹模板

第四步：引入 meas-stream（不加 gate）

目标：

验证量测异构流本身是否带来增益

第五步：对 meas-stream 加 soft gate

目标：

参考 Selective 思路，检验选择性量测特征是否进一步提升

第六步：再考虑时序 gate / learned reliability

目标：

进一步提升动态场景下的可靠性建模能力

7. 当前明确不做的事

为了避免研究发散，当前阶段默认不做以下内容，除非后续明确重启：

不直接在旧的 fault-training 版本上继续堆新模块

不一开始就做“异构流 + gate + 时序 + 新损失”全家桶

不直接把 hard valid 完全替换成 learned variable

不直接把原始量测长历史序列大量堆入模型

不在没有 clean baseline 的前提下讨论最终创新结论

8. 当前对于“有效性”的基本标准

后续任何新增模块（例如 meas-stream、gate、reliability）都必须满足：

不能破坏 clean baseline 的训练稳定性

必须通过消融实验证明有效

必须与现有主干职责清晰分离

不能仅靠参数量增加掩盖问题

最好在泛化/故障场景中体现价值，而不只是 clean RMSE 微小下降

9. 后续提及时的默认理解

如果后续提到以下关键词，默认按如下方式理解：

“original”

指：

以 EKF 后验特征为主的统一节点融合器

当前优先使用 clean 版本作为 baseline

“clean”

指：

无 dropout

无 jump bias pollution

保留多传感器异构性

“fault test”

指：

在 clean training baseline 之后进行的故障泛化测试

“dual-stream / 双流”

指：

post-stream + meas-stream

不是替代 EKF，而是在融合层补回量测异构信息

“gate”

默认优先理解为：

针对 meas-stream 的 soft gate

灵感来自 Selective Sensor Fusion

不是直接替代 hard valid

“reliability / confidence”

默认理解为：

从 hand-crafted confidence 逐步走向 learned reliability

但当前不作为第一优先级模块

10. 一句话总结当前分支

当前分支的核心任务是：先建立 clean original 融合基线，再围绕“故障泛化、轨迹泛化、双流特征、量测流 gate、learned reliability”逐步推进，而不是在已有故障训练版本上直接叠加复杂结构。