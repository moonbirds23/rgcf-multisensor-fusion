from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List

from configs.scenario_config import ScenarioConfig, MotionConfig, SensorLayoutConfig, SensorNodeConfig
from core.types import ExperimentBundle
from .trajectory import TruthTrajectory, generate_truth_ctrvish
from .sensor_layouts import (
    build_default_clean_4sensor_layout,
    build_wide_baseline_4sensor_layout,
    build_asymmetric_4sensor_layout,
    build_hetero_4sensor_layout,
)


@dataclass
class ScenarioArtifacts:
    """
    场景产物容器：
    - truth: 轨迹
    - motion: 本场景实际使用的运动配置
    - sensor_layout: 本场景实际使用的传感器布局
    """
    truth: TruthTrajectory
    motion: MotionConfig
    sensor_layout: SensorLayoutConfig

    def to_dict(self) -> Dict:
        return {
            "truth": self.truth.to_dict(),
            "motion": self.motion,
            "sensor_layout": self.sensor_layout,
        }


class BaseScenario(ABC):
    """
    场景基类。

    设计原则：
    1. 场景层只定义 truth、布局和 nominal 配置；
    2. 不负责 fault；
    3. 不负责传感器类实例化；
    4. 不负责滤波和训练。
    """

    def __init__(self, scenario_cfg: ScenarioConfig):
        self.cfg = deepcopy(scenario_cfg)

    @property
    def scene_name(self) -> str:
        return self.cfg.scene_name

    @abstractmethod
    def build_motion_config(self) -> MotionConfig:
        pass

    @abstractmethod
    def build_sensor_layout(self) -> SensorLayoutConfig:
        pass

    def generate_truth(self) -> TruthTrajectory:
        motion = self.build_motion_config()
        return generate_truth_ctrvish(motion)

    def build(self) -> ScenarioArtifacts:
        motion = self.build_motion_config()
        sensor_layout = self.build_sensor_layout()
        truth = generate_truth_ctrvish(motion)
        return ScenarioArtifacts(
            truth=truth,
            motion=motion,
            sensor_layout=sensor_layout,
        )


class DefaultClean4SensorScenario(BaseScenario):
    """
    默认 clean 4 节点场景：
    应与当前 clean baseline 的主设定保持一致。
    """

    def build_motion_config(self) -> MotionConfig:
        return deepcopy(self.cfg.motion)

    def build_sensor_layout(self) -> SensorLayoutConfig:
        # 若 config 中已经指定了 layout，则直接使用；
        # 否则回落到默认 clean 4 节点布局。
        layout = deepcopy(self.cfg.sensor_layout)
        if len(layout.sensors) == 0:
            layout = build_default_clean_4sensor_layout()
        return layout


class WideBaseline4SensorScenario(BaseScenario):
    """
    更宽基线场景。
    先保持 truth 机动族不变，只改传感器布局，
    用于后续几何分布泛化测试。
    """

    def build_motion_config(self) -> MotionConfig:
        return deepcopy(self.cfg.motion)

    def build_sensor_layout(self) -> SensorLayoutConfig:
        return build_wide_baseline_4sensor_layout()


class Asymmetric4SensorScenario(BaseScenario):
    """
    非对称布局场景。
    适合后续检验几何不均衡条件下的融合行为。
    """

    def build_motion_config(self) -> MotionConfig:
        return deepcopy(self.cfg.motion)

    def build_sensor_layout(self) -> SensorLayoutConfig:
        return build_asymmetric_4sensor_layout()


class Hetero4SensorScenario(BaseScenario):
    """
    真异构4传感器场景：GPS + Radar + AOA + UWB
    四类传感器量测空间完全不同，信息互补，
    为 meas-stream gate 提供真实区分信号。
    """

    def build_motion_config(self) -> MotionConfig:
        return deepcopy(self.cfg.motion)

    def build_sensor_layout(self) -> SensorLayoutConfig:
        return build_hetero_4sensor_layout()


def scenario_artifacts_to_plain_dict(artifacts: ScenarioArtifacts) -> Dict:
    """
    给后续保存 JSON/调试打印预留的辅助函数。
    """
    out = {
        "scene_name": artifacts.motion,
        "num_sensors": len(artifacts.sensor_layout.sensors),
        "sensor_names": [s.name for s in artifacts.sensor_layout.sensors],
        "sensor_types": [s.sensor_type for s in artifacts.sensor_layout.sensors],
        "sensor_positions": [list(s.position) for s in artifacts.sensor_layout.sensors],
    }
    return out