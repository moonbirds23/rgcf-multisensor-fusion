from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DropoutWindow:
    sensor_ids: List[int]
    t0: float
    t1: float


@dataclass
class DropoutFaultConfig:
    enabled: bool = False
    windows: List[DropoutWindow] = field(default_factory=list)


@dataclass
class PollutionFaultConfig:
    enabled: bool = False
    target_sensor_ids: List[int] = field(default_factory=list)
    bias_rw_sigma: float = 0.03
    jump_prob: float = 0.0
    jump_sigma: float = 0.0
    active_time_window: Optional[Tuple[float, float]] = None
    fault_shape: str = "jump"


@dataclass
class FaultConfig:
    mode: str = "clean"
    dropout: DropoutFaultConfig = field(default_factory=DropoutFaultConfig)
    pollution: PollutionFaultConfig = field(default_factory=PollutionFaultConfig)
    matrix_cases: List[Dict[str, Any]] = field(default_factory=list)
    candidate_sensor_ids: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    random_target_by_seed: bool = False
    window: Optional[Tuple[float, float]] = None
