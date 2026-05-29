from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from .measurement_models import wrap_angle_rad


@dataclass
class EKFState:
    """
    单节点 EKF 状态容器
    """
    x: np.ndarray   # shape [4]
    P: np.ndarray   # shape [4,4]


class CVEKF:
    """
    统一 CV-EKF
    状态：x = [px, py, vx, vy]
    """

    def __init__(self, dt: float, sigma_a: float):
        self.dt = float(dt)
        self.sigma_a = float(sigma_a)

        self.F = np.array([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0, self.dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)

        q = self.sigma_a ** 2
        dt = self.dt
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2

        self.Q = q * np.array([
            [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
            [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
            [dt3 / 2.0, 0.0, dt2, 0.0],
            [0.0, dt3 / 2.0, 0.0, dt2],
        ], dtype=np.float64)

    def predict(self, state: EKFState) -> EKFState:
        x_pred = self.F @ state.x
        P_pred = self.F @ state.P @ self.F.T + self.Q
        return EKFState(x=x_pred, P=P_pred)

    def update(
        self,
        state: EKFState,
        z: np.ndarray,
        R: np.ndarray,
        h_fn: Callable,
        H_fn: Callable,
        angle_index: Optional[int] = None,
        *h_args,
    ) -> EKFState:
        """
        通用 EKF update
        - h_fn: 观测函数
        - H_fn: 观测 Jacobian
        - angle_index: 若观测包含角度项，需要对创新做 wrap
        """
        x_pred, P_pred = state.x, state.P

        z_hat = h_fn(x_pred, *h_args)
        H = H_fn(x_pred, *h_args)

        y = z - z_hat
        if angle_index is not None:
            y[angle_index] = wrap_angle_rad(float(y[angle_index]))

        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x_upd = x_pred + K @ y

        I = np.eye(P_pred.shape[0], dtype=np.float64)
        P_upd = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ R @ K.T

        return EKFState(x=x_upd, P=P_upd)