from __future__ import annotations

from copy import deepcopy
from typing import List

from core.types import ExperimentBundle
from configs.scenario_config import SensorNodeConfig
from .sensors import (
    SensorBase,
    GPS2D,
    RadarRangeBearing,
    AOAOnly,
    UWBRangeOnly,
)


def build_sensors_from_bundle(bundle: ExperimentBundle) -> List[SensorBase]:
    """
    从 ExperimentBundle 中读取 scene layout，并构造 nominal sensors。
    """
    sensors_cfg = deepcopy(bundle.scenario.sensor_layout.sensors)
    sensors: List[SensorBase] = []

    for node in sensors_cfg:
        sensors.append(build_single_sensor(node))

    return sensors


def build_single_sensor(node: SensorNodeConfig) -> SensorBase:
    """
    根据单个 SensorNodeConfig 构造 sensor 实例。
    """
    if node.sensor_type == "gps2d":
        return GPS2D(
            sid=node.sensor_id,
            name=node.name,
            pos=node.position,
            sigma=node.gps_sigma,
            bias_rw_sigma=node.gps_bias_rw_sigma,
        )

    if node.sensor_type == "radar_rb":
        return RadarRangeBearing(
            sid=node.sensor_id,
            name=node.name,
            pos=node.position,
            sigma_r=node.radar_sigma_r,
            sigma_theta_deg=node.radar_sigma_theta_deg,
        )

    if node.sensor_type == "aoa_only":
        return AOAOnly(
            sid=node.sensor_id,
            name=node.name,
            pos=node.position,
            sigma_theta_deg=node.aoa_sigma_theta_deg,
        )

    if node.sensor_type == "uwb_range_only":
        return UWBRangeOnly(
            sid=node.sensor_id,
            name=node.name,
            pos=node.position,
            sigma_r=node.uwb_sigma_r,
            far_r0=node.uwb_far_r0,
            far_k=node.uwb_far_k,
        )

    raise ValueError(f"Unsupported sensor_type: {node.sensor_type}")