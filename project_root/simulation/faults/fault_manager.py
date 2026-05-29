from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from core.types import ExperimentBundle
from .base_fault import FaultContext, FaultResult
from .dropout_fault import DropoutFault
from .pollution_fault import PollutionFault


@dataclass
class FaultApplyTrace:
    """
    单时刻单节点 fault 处理日志。
    后续可用于保存 CSV 或 debug。
    """
    sensor_id: int
    t: float
    fault_name: str
    triggered: bool
    info: Dict = field(default_factory=dict)


class FaultManager:
    """
    fault 总管理器。
    支持多个 fault 依次作用到同一量测上。

    当前默认顺序：
    1. dropout
    2. pollution

    这个顺序的含义是：
    - 如果先被 dropout 干掉，则后续 pollution 不再有意义
    - 这与“availability 先于 reliability/quality”也一致，
      与 readme 中保留 hard valid 的理解相吻合。:contentReference[oaicite:8]{index=8}
    """

    def __init__(self, faults: Optional[List] = None):
        self.faults = faults or []

    def apply(
        self,
        *,
        sensor_id: int,
        sensor_type: str,
        t: float,
        truth_state: np.ndarray,
        z: Optional[np.ndarray],
        R: Optional[np.ndarray],
        valid: bool,
        rng: np.random.Generator,
        metadata: Optional[Dict] = None,
    ):
        ctx = FaultContext(
            sensor_id=sensor_id,
            sensor_type=sensor_type,
            t=t,
            truth_state=truth_state,
            z=z,
            R=R,
            valid=valid,
            metadata=metadata or {},
        )

        traces: List[FaultApplyTrace] = []

        current_z = ctx.z
        current_R = ctx.R
        current_valid = ctx.valid

        for fault in self.faults:
            result: FaultResult = fault.apply(
                FaultContext(
                    sensor_id=ctx.sensor_id,
                    sensor_type=ctx.sensor_type,
                    t=ctx.t,
                    truth_state=ctx.truth_state,
                    z=current_z,
                    R=current_R,
                    valid=current_valid,
                    metadata=ctx.metadata,
                ),
                rng=rng,
            )

            traces.append(
                FaultApplyTrace(
                    sensor_id=sensor_id,
                    t=float(t),
                    fault_name=fault.name,
                    triggered=result.triggered,
                    info=result.info,
                )
            )

            current_z = result.z
            current_R = result.R
            current_valid = result.valid

            # 一旦变成 invalid，后续 fault 默认不再继续改
            if not current_valid:
                break

        return current_z, current_R, current_valid, traces


def build_fault_manager_from_bundle(bundle: ExperimentBundle) -> FaultManager:
    """
    根据 ExperimentBundle 构建 fault manager。
    """
    faults = []

    if bundle.fault.dropout.enabled:
        faults.append(DropoutFault(bundle.fault.dropout))

    if bundle.fault.pollution.enabled:
        faults.append(PollutionFault(bundle.fault.pollution))

    return FaultManager(faults=faults)