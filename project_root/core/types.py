from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from configs.base_config import BaseConfig
from configs.fault_config import FaultConfig
from configs.model_config import ModelConfig
from configs.scenario_config import ScenarioConfig
from configs.train_config import TrainConfig


@dataclass
class ExperimentIdentity:
    preset_name: str
    experiment_name: str
    scene_name: str
    model_name: str
    fault_mode: str


@dataclass
class RuntimeFlags:
    do_simulate: bool = True
    do_train: bool = False
    do_eval: bool = False
    do_repeat: bool = False
    do_plot_weights: bool = False


@dataclass
class SceneRuntimeSpec:
    config: ScenarioConfig
    scene_name: str
    dt: float
    T: float
    num_sensors: int


@dataclass
class FaultRuntimeSpec:
    config: FaultConfig
    mode: str
    has_dropout: bool
    has_pollution: bool


@dataclass
class ModelRuntimeSpec:
    config: ModelConfig
    model_name: str
    in_dim: int
    hidden_dim: int
    use_post_stream: bool
    use_meas_stream: bool
    use_gate: bool


@dataclass
class TrainRuntimeSpec:
    config: TrainConfig
    epochs: int
    lr: float
    batch_size: int
    repeat_runs: int


@dataclass
class ExperimentBundle:
    identity: ExperimentIdentity
    base: BaseConfig
    scenario: ScenarioConfig
    fault: FaultConfig
    model: ModelConfig
    train: TrainConfig
    runtime_flags: RuntimeFlags = field(default_factory=RuntimeFlags)
    scene_spec: Optional[SceneRuntimeSpec] = None
    fault_spec: Optional[FaultRuntimeSpec] = None
    model_spec: Optional[ModelRuntimeSpec] = None
    train_spec: Optional[TrainRuntimeSpec] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunRequest:
    mode: str = "simulate"
    preset_name: str = "clean_baseline"
    device: Optional[str] = None
    epochs: Optional[int] = None
    lr: Optional[float] = None
    batch_size: Optional[int] = None
    hidden_dim: Optional[int] = None
    repeat_runs: Optional[int] = None
    experiment_name: Optional[str] = None
