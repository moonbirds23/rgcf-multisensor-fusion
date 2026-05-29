from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class FaultContext:
    """
    单次量测的故障上下文信息。

    说明：
    - sensor_id: 当前量测来自哪个节点
    - sensor_type: 当前传感器类型，例如 gps2d / radar_rb
    - t: 当前时刻（秒）
    - truth_state: 当前真值状态（建议传入 [px, py, ...]）
    - z: 当前 nominal measurement（故障注入前）
    - R: 当前 nominal measurement covariance（故障注入前）
    - valid: 当前 nominal 是否有效
    - metadata: 可扩展信息，例如位置、几何量等
    """
    sensor_id: int
    sensor_type: str
    t: float
    truth_state: np.ndarray
    z: Optional[np.ndarray]
    R: Optional[np.ndarray]
    valid: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FaultResult:
    """
    故障处理后的输出。

    说明：
    - z / R / valid: 故障后的量测结果
    - triggered: 当前 fault 是否触发
    - fault_name: 触发的 fault 名称
    - info: 额外调试信息，后续可保存到日志/CSV
    """
    z: Optional[np.ndarray]
    R: Optional[np.ndarray]
    valid: bool
    triggered: bool = False
    fault_name: Optional[str] = None
    info: Dict[str, Any] = field(default_factory=dict)


class FaultBase(ABC):
    """
    所有故障插件的统一抽象基类。

    设计原则：
    1. 输入是 nominal measurement context
    2. 输出是修改后的 measurement result
    3. 故障层不关心 EKF，不关心网络，只关心量测本身
    """

    name: str = "base_fault"

    @abstractmethod
    def apply(self, ctx: FaultContext, rng: np.random.Generator) -> FaultResult:
        raise NotImplementedError