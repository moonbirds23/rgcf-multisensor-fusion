from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import math
import numpy as np


def deg2rad(d: float) -> float:
    return d * math.pi / 180.0


def wrap_angle_rad(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class SensorBase(ABC):
    """
    传感器基类。
    当前约定：
    - 传感器本体只负责 nominal measurement
    - fault 不在这里做
    """

    def __init__(self, sid: int, name: str, sensor_type: str, pos: Tuple[float, float]):
        self.sid = sid
        self.name = name
        self.sensor_type = sensor_type
        self.sx, self.sy = pos

    @abstractmethod
    def measure(
        self,
        truth_state: np.ndarray,
        t: float,
        rng: np.random.Generator,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
        raise NotImplementedError


class GPS2D(SensorBase):
    """
    nominal GPS2D：
    - 自带小 bias RW（这是 nominal 传感器特性，不当作显式 fault）
    - 不自带 jump pollution
    - 不自带 dropout
    """

    def __init__(
        self,
        sid: int,
        name: str,
        pos: Tuple[float, float],
        sigma: float,
        bias_rw_sigma: float,
    ):
        super().__init__(sid=sid, name=name, sensor_type="gps2d", pos=pos)
        self.sigma = sigma
        self.bias_rw_sigma = bias_rw_sigma
        self.bias = np.zeros((2,), dtype=np.float64)

    def measure(self, truth_state: np.ndarray, t: float, rng: np.random.Generator):
        px, py = float(truth_state[0]), float(truth_state[1])

        # nominal 轻微 bias random walk
        self.bias += rng.normal(0.0, self.bias_rw_sigma, size=(2,))

        noise = rng.normal(0.0, self.sigma, size=(2,))
        z = np.array([px, py], dtype=np.float64) + self.bias + noise
        R = np.diag([self.sigma ** 2, self.sigma ** 2]).astype(np.float64)
        return z, R, True


class RadarRangeBearing(SensorBase):
    """
    nominal range-bearing radar
    """

    def __init__(
        self,
        sid: int,
        name: str,
        pos: Tuple[float, float],
        sigma_r: float,
        sigma_theta_deg: float,
    ):
        super().__init__(sid=sid, name=name, sensor_type="radar_rb", pos=pos)
        self.sigma_r = sigma_r
        self.sigma_theta = deg2rad(sigma_theta_deg)

    def measure(self, truth_state: np.ndarray, t: float, rng: np.random.Generator):
        px, py = float(truth_state[0]), float(truth_state[1])
        dx, dy = px - self.sx, py - self.sy
        r = math.sqrt(dx * dx + dy * dy)
        theta = math.atan2(dy, dx)

        z = np.array([
            r + rng.normal(0.0, self.sigma_r),
            wrap_angle_rad(theta + rng.normal(0.0, self.sigma_theta)),
        ], dtype=np.float64)

        R = np.diag([self.sigma_r ** 2, self.sigma_theta ** 2]).astype(np.float64)
        return z, R, True


class AOAOnly(SensorBase):
    """
    预留：AOA-only 节点
    当前 nominal 版本不带 dropout、不带 outlier fault 插件化；
    若未来启用，可把 outlier 也拆到 fault 层。
    """

    def __init__(
        self,
        sid: int,
        name: str,
        pos: Tuple[float, float],
        sigma_theta_deg: float,
    ):
        super().__init__(sid=sid, name=name, sensor_type="aoa_only", pos=pos)
        self.sigma_theta = deg2rad(sigma_theta_deg)

    def measure(self, truth_state: np.ndarray, t: float, rng: np.random.Generator):
        px, py = float(truth_state[0]), float(truth_state[1])
        dx, dy = px - self.sx, py - self.sy
        theta = math.atan2(dy, dx)

        z = np.array([
            wrap_angle_rad(theta + rng.normal(0.0, self.sigma_theta))
        ], dtype=np.float64)
        R = np.array([[self.sigma_theta ** 2]], dtype=np.float64)
        return z, R, True


class UWBRangeOnly(SensorBase):
    """
    预留：UWB range-only
    nominal 情况下可带距离相关噪声。
    """

    def __init__(
        self,
        sid: int,
        name: str,
        pos: Tuple[float, float],
        sigma_r: float,
        far_r0: float,
        far_k: float,
    ):
        super().__init__(sid=sid, name=name, sensor_type="uwb_range_only", pos=pos)
        self.sigma_r = sigma_r
        self.far_r0 = far_r0
        self.far_k = far_k

    def measure(self, truth_state: np.ndarray, t: float, rng: np.random.Generator):
        px, py = float(truth_state[0]), float(truth_state[1])
        dx, dy = px - self.sx, py - self.sy
        r = math.sqrt(dx * dx + dy * dy)

        sigma = self.sigma_r
        if r > self.far_r0:
            sigma = sigma + self.far_k * (r - self.far_r0)

        z = np.array([r + rng.normal(0.0, sigma)], dtype=np.float64)
        R = np.array([[sigma ** 2]], dtype=np.float64)
        return z, R, True