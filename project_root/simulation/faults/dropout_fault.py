from __future__ import annotations

from typing import List

import numpy as np

from configs.fault_config import DropoutWindow, DropoutFaultConfig
from .base_fault import FaultBase, FaultContext, FaultResult


class DropoutFault(FaultBase):
    """
    硬 dropout fault：
    在指定时间窗内，让指定传感器量测无效。
    """
    name = "dropout"

    def __init__(self, config: DropoutFaultConfig):
        self.config = config
        self.windows: List[DropoutWindow] = list(config.windows)

    def _match_any_window(self, sensor_id: int, t: float) -> bool:
        for w in self.windows:
            if sensor_id in w.sensor_ids and (w.t0 <= t <= w.t1):
                return True
        return False

    def apply(self, ctx: FaultContext, rng: np.random.Generator) -> FaultResult:
        if not self.config.enabled:
            return FaultResult(
                z=ctx.z,
                R=ctx.R,
                valid=ctx.valid,
                triggered=False,
                fault_name=self.name,
            )

        if self._match_any_window(ctx.sensor_id, ctx.t):
            return FaultResult(
                z=None,
                R=None,
                valid=False,
                triggered=True,
                fault_name=self.name,
                info={
                    "sensor_id": ctx.sensor_id,
                    "t": float(ctx.t),
                    "reason": "inside_dropout_window",
                },
            )

        return FaultResult(
            z=ctx.z,
            R=ctx.R,
            valid=ctx.valid,
            triggered=False,
            fault_name=self.name,
        )