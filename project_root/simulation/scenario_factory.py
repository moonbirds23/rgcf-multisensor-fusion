from __future__ import annotations

from copy import deepcopy

from core.types import ExperimentBundle
from .scenarios import (
    BaseScenario,
    DefaultClean4SensorScenario,
    WideBaseline4SensorScenario,
    Asymmetric4SensorScenario,
    Hetero4SensorScenario,
    Phase1BalancedHeteroNominalScenario,
    Phase1ClusteredHeteroNominalScenario,
    Phase1ManeuverHeteroNominalScenario,
)


def build_scenario_from_bundle(bundle: ExperimentBundle) -> BaseScenario:
    """
    从 ExperimentBundle 构造场景实例。

    当前支持：
    - default_clean_4sensor_scene
    - wide_baseline_4sensor_scene
    - asymmetric_4sensor_scene
    - phase1_s1_balanced_hetero_nominal
    - phase1_s2_clustered_hetero_nominal
    - phase1_s3_maneuver_hetero_nominal
    """
    scenario_cfg = deepcopy(bundle.scenario)
    scene_name = scenario_cfg.scene_name

    if scene_name == "default_clean_4sensor_scene":
        return DefaultClean4SensorScenario(scenario_cfg)

    if scene_name == "wide_baseline_4sensor_scene":
        return WideBaseline4SensorScenario(scenario_cfg)

    if scene_name == "asymmetric_4sensor_scene":
        return Asymmetric4SensorScenario(scenario_cfg)

    if scene_name == "hetero_4sensor_scene":
        return Hetero4SensorScenario(scenario_cfg)

    if scene_name == "phase1_s1_balanced_hetero_nominal":
        return Phase1BalancedHeteroNominalScenario(scenario_cfg)

    if scene_name == "phase1_s2_clustered_hetero_nominal":
        return Phase1ClusteredHeteroNominalScenario(scenario_cfg)

    if scene_name == "phase1_s3_maneuver_hetero_nominal":
        return Phase1ManeuverHeteroNominalScenario(scenario_cfg)

    raise ValueError(f"Unknown scene_name: {scene_name}")
