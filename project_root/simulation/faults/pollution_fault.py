from __future__ import annotations

from typing import Dict

import numpy as np

from configs.fault_config import PollutionFaultConfig
from .base_fault import FaultBase, FaultContext, FaultResult


class PollutionFault(FaultBase):
    """
    Sensor-aware measurement pollution.

    The injected bias follows each sensor's measurement space:
    - gps2d: x/y position bias in meters
    - radar_rb: range bias in meters and bearing bias in radians
    - aoa_only: bearing bias in radians
    - uwb_range_only: range bias in meters
    """

    name = "pollution"

    def __init__(self, config: PollutionFaultConfig):
        self.config = config
        self.bias_state: Dict[tuple[int, str], np.ndarray] = {}

    def _is_active(self, sensor_id: int, t: float) -> bool:
        if not self.config.enabled:
            return False
        if sensor_id not in self.config.target_sensor_ids:
            return False
        if self.config.active_time_window is None:
            return True

        t0, t1 = self.config.active_time_window
        return float(t0) <= float(t) <= float(t1)

    def _bias_key(self, sensor_id: int, sensor_type: str) -> tuple[int, str]:
        return int(sensor_id), str(sensor_type)

    def _get_bias(self, sensor_id: int, sensor_type: str, dim: int) -> np.ndarray:
        key = self._bias_key(sensor_id, sensor_type)
        if key not in self.bias_state:
            self.bias_state[key] = np.zeros((dim,), dtype=np.float64)
        return self.bias_state[key]

    def _jump_sigma_vec(self, sensor_type: str, dim: int) -> np.ndarray:
        base = float(self.config.jump_sigma)
        if sensor_type == "gps2d":
            return np.full((dim,), base, dtype=np.float64)
        if sensor_type == "radar_rb":
            return np.array([base, np.deg2rad(2.0)], dtype=np.float64)
        if sensor_type == "aoa_only":
            return np.array([np.deg2rad(2.0)], dtype=np.float64)
        if sensor_type == "uwb_range_only":
            return np.array([base], dtype=np.float64)
        return np.full((dim,), base, dtype=np.float64)

    def _sample_bias(self, sensor_type: str, dim: int, rng: np.random.Generator) -> np.ndarray:
        sigma = self._jump_sigma_vec(sensor_type, dim)
        return rng.normal(0.0, sigma, size=(dim,))

    def _bias_ramp(self, ctx: FaultContext, dim: int, rng: np.random.Generator) -> np.ndarray:
        t0, t1 = self.config.active_time_window or (ctx.t, ctx.t)
        denom = max(float(t1) - float(t0), 1e-6)
        ratio = float(np.clip((float(ctx.t) - float(t0)) / denom, 0.0, 1.0))
        key = self._bias_key(ctx.sensor_id, ctx.sensor_type)
        if key not in self.bias_state:
            self.bias_state[key] = self._sample_bias(ctx.sensor_type, dim, rng)
        return ratio * self.bias_state[key]

    def apply(self, ctx: FaultContext, rng: np.random.Generator) -> FaultResult:
        if not self._is_active(ctx.sensor_id, ctx.t):
            return FaultResult(
                z=ctx.z,
                R=ctx.R,
                valid=ctx.valid,
                triggered=False,
                fault_name=self.name,
            )

        if (ctx.z is None) or (ctx.R is None) or (not ctx.valid):
            return FaultResult(
                z=ctx.z,
                R=ctx.R,
                valid=ctx.valid,
                triggered=False,
                fault_name=self.name,
            )

        z = np.array(ctx.z, dtype=np.float64).reshape(-1).copy()
        R = np.array(ctx.R, dtype=np.float64).copy()
        dim = int(z.shape[0])
        fault_shape = str(getattr(self.config, "fault_shape", "jump"))

        if fault_shape == "bias_ramp":
            bias = self._bias_ramp(ctx, dim, rng)
            jumped = False
        else:
            bias = self._get_bias(ctx.sensor_id, ctx.sensor_type, dim=dim)
            if self.config.bias_rw_sigma > 0.0:
                bias += rng.normal(0.0, self.config.bias_rw_sigma, size=(dim,))

            jumped = False
            if self.config.jump_prob > 0.0 and rng.random() < self.config.jump_prob:
                if self.config.jump_prob >= 1.0:
                    bias[:] = self._sample_bias(ctx.sensor_type, dim, rng)
                else:
                    bias += self._sample_bias(ctx.sensor_type, dim, rng)
                jumped = True

        z_fault = z + bias

        return FaultResult(
            z=z_fault,
            R=R,
            valid=True,
            triggered=True,
            fault_name=self.name,
            info={
                "sensor_id": ctx.sensor_id,
                "sensor_type": ctx.sensor_type,
                "t": float(ctx.t),
                "fault_shape": fault_shape,
                "jumped": jumped,
                "bias": bias.tolist(),
            },
        )
