from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import math
import numpy as np

from configs.scenario_config import MotionConfig


def deg2rad(d: float) -> float:
    return d * math.pi / 180.0


@dataclass
class TruthTrajectory:
    """
    标准 truth 轨迹容器。

    t: shape [K]
    x5: shape [K, 5], 状态顺序为 [px, py, v, psi, yaw_rate]
    x4: shape [K, 4], 状态顺序为 [px, py, vx, vy]
    """
    t: np.ndarray
    x5: np.ndarray
    x4: np.ndarray

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {
            "t": self.t,
            "x_true_5d": self.x5,
            "x_true_4d": self.x4,
        }


def _piecewise_yaw_and_accel(tk: float, cfg: MotionConfig):
    """
    按当前 clean 版的思路给一个分段机动策略。
    这里不追求和你旧函数逐行逐句完全相同，而是保持同一类动力学风格：
    - 前段带轻微加速
    - 中段转弯
    - 后段反向转弯
    - 全程叠加小扰动
    """
    # 分段规则可后续继续参数化，但当前先固定为“同机动族默认场景”
    a = 0.0
    yaw_rate = 0.0

    if 10.0 <= tk < 35.0:
        a = cfg.a_segmentA

    if 40.0 <= tk < 70.0:
        yaw_rate = deg2rad(cfg.yaw_rate_B_deg)

    if 85.0 <= tk < 85.0 + cfg.yaw_rate_D_duration:
        yaw_rate = deg2rad(cfg.yaw_rate_D_deg)

    # 轻微周期扰动，保持和你原设计一致的“非完全规则”风格
    yaw_rate += deg2rad(cfg.s_disturb_amp_deg) * math.sin(2.0 * math.pi * tk / cfg.s_disturb_period)

    return a, yaw_rate


def generate_truth_ctrvish(cfg: MotionConfig) -> TruthTrajectory:
    """
    基于 MotionConfig 生成 CTRV-ish truth trajectory。

    说明：
    1. 这一步只负责 truth 生成，不负责故障、不负责传感器、不负责滤波；
    2. 状态主内部表示为 [px, py, v, psi, yaw_rate]；
    3. 同时输出 [px, py, vx, vy] 版本，便于后续 EKF / 网络直接使用。
    """
    dt = cfg.dt
    K = int(cfg.T / dt) + 1
    t = np.linspace(0.0, cfg.T, K)

    x5 = np.zeros((K, 5), dtype=np.float64)

    px = float(cfg.init_px)
    py = float(cfg.init_py)
    v = float(cfg.init_v)
    psi = deg2rad(cfg.init_psi_deg)
    yaw_rate = 0.0

    for k in range(K):
        tk = t[k]

        # 保存当前时刻状态
        x5[k, 0] = px
        x5[k, 1] = py
        x5[k, 2] = v
        x5[k, 3] = psi
        x5[k, 4] = yaw_rate

        # 最后一帧不再推进
        if k == K - 1:
            break

        a_cmd, yaw_rate_cmd = _piecewise_yaw_and_accel(tk, cfg)

        # 速度更新
        v = max(0.0, v + a_cmd * dt)

        # 航向更新
        psi = psi + yaw_rate_cmd * dt
        yaw_rate = yaw_rate_cmd

        # 位置更新（近似积分）
        px = px + v * math.cos(psi) * dt
        py = py + v * math.sin(psi) * dt

    # 转成 [px, py, vx, vy]
    x4 = np.zeros((K, 4), dtype=np.float64)
    x4[:, 0] = x5[:, 0]
    x4[:, 1] = x5[:, 1]
    x4[:, 2] = x5[:, 2] * np.cos(x5[:, 3])
    x4[:, 3] = x5[:, 2] * np.sin(x5[:, 3])

    return TruthTrajectory(t=t, x5=x5, x4=x4)