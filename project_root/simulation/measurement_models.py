from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def wrap_angle_rad(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# =========================================================
# 状态定义
# 这里统一采用 4 维 CV 状态：
# x = [px, py, vx, vy]
# =========================================================


def h_gps2d(x: np.ndarray) -> np.ndarray:
    """
    GPS2D 观测模型：
    z = [px, py]
    """
    return np.array([x[0], x[1]], dtype=np.float64)


def H_gps2d(x: np.ndarray) -> np.ndarray:
    """
    GPS2D 观测 Jacobian
    """
    H = np.zeros((2, 4), dtype=np.float64)
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    return H


def h_radar_rb(x: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """
    Range-bearing radar:
    z = [r, theta]
    """
    px, py = float(x[0]), float(x[1])
    dx, dy = px - sx, py - sy
    r = math.sqrt(dx * dx + dy * dy)
    theta = math.atan2(dy, dx)
    return np.array([r, theta], dtype=np.float64)


def H_radar_rb(x: np.ndarray, sx: float, sy: float, eps: float = 1e-8) -> np.ndarray:
    """
    Range-bearing radar Jacobian wrt [px, py, vx, vy]
    """
    px, py = float(x[0]), float(x[1])
    dx, dy = px - sx, py - sy
    r2 = max(dx * dx + dy * dy, eps)
    r = math.sqrt(r2)

    H = np.zeros((2, 4), dtype=np.float64)

    # dr/dx, dr/dy
    H[0, 0] = dx / r
    H[0, 1] = dy / r

    # dtheta/dx, dtheta/dy
    H[1, 0] = -dy / r2
    H[1, 1] = dx / r2

    return H


def h_aoa_only(x: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """
    预留：AOA-only
    z = [theta]
    """
    px, py = float(x[0]), float(x[1])
    dx, dy = px - sx, py - sy
    theta = math.atan2(dy, dx)
    return np.array([theta], dtype=np.float64)


def H_aoa_only(x: np.ndarray, sx: float, sy: float, eps: float = 1e-8) -> np.ndarray:
    px, py = float(x[0]), float(x[1])
    dx, dy = px - sx, py - sy
    r2 = max(dx * dx + dy * dy, eps)

    H = np.zeros((1, 4), dtype=np.float64)
    H[0, 0] = -dy / r2
    H[0, 1] = dx / r2
    return H


def h_uwb_range_only(x: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """
    预留：UWB range-only
    z = [r]
    """
    px, py = float(x[0]), float(x[1])
    dx, dy = px - sx, py - sy
    r = math.sqrt(dx * dx + dy * dy)
    return np.array([r], dtype=np.float64)


def H_uwb_range_only(x: np.ndarray, sx: float, sy: float, eps: float = 1e-8) -> np.ndarray:
    px, py = float(x[0]), float(x[1])
    dx, dy = px - sx, py - sy
    r2 = max(dx * dx + dy * dy, eps)
    r = math.sqrt(r2)

    H = np.zeros((1, 4), dtype=np.float64)
    H[0, 0] = dx / r
    H[0, 1] = dy / r
    return H