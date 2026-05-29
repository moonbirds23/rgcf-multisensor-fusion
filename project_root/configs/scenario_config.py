from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Tuple

SensorType = Literal["gps2d", "radar_rb", "aoa_only", "uwb_range_only"]


@dataclass
class MotionConfig:
    dt: float = 0.1
    T: float = 120.0
    init_px: float = 0.0
    init_py: float = 0.0
    init_v: float = 15.0
    init_psi_deg: float = 20.0
    a_segmentA: float = 0.3
    yaw_rate_B_deg: float = 3.0
    yaw_rate_D_deg: float = -4.0
    yaw_rate_D_duration: float = 15.0
    s_disturb_amp_deg: float = 1.5
    s_disturb_period: float = 8.0
    sigma_a_base: float = 1.0
    sigma_a_turn: float = 1.5


@dataclass
class SensorNodeConfig:
    sensor_id: int
    name: str
    sensor_type: SensorType
    position: Tuple[float, float]
    gps_sigma: float = 5.0
    gps_bias_rw_sigma: float = 0.02
    radar_sigma_r: float = 3.0
    radar_sigma_theta_deg: float = 0.5
    aoa_sigma_theta_deg: float = 0.8
    uwb_sigma_r: float = 1.5
    uwb_far_r0: float = 1300.0
    uwb_far_k: float = 0.002


def _default_sensors() -> List[SensorNodeConfig]:
    return [
        SensorNodeConfig(1, "s1_gps", "gps2d", (-800.0, -200.0), gps_sigma=5.0, gps_bias_rw_sigma=0.02),
        SensorNodeConfig(2, "s2_radar", "radar_rb", (900.0, -100.0), radar_sigma_r=3.0, radar_sigma_theta_deg=0.5),
        SensorNodeConfig(3, "s3_gps_clean", "gps2d", (-300.0, 900.0), gps_sigma=2.5, gps_bias_rw_sigma=0.03),
        SensorNodeConfig(4, "s4_radar", "radar_rb", (700.0, 850.0), radar_sigma_r=2.0, radar_sigma_theta_deg=0.35),
    ]


@dataclass
class SensorLayoutConfig:
    sensors: List[SensorNodeConfig] = field(default_factory=_default_sensors)


@dataclass
class ScenarioConfig:
    scene_name: str = "default_clean_4sensor_scene"
    motion: MotionConfig = field(default_factory=MotionConfig)
    sensor_layout: SensorLayoutConfig = field(default_factory=SensorLayoutConfig)
    pos_scale: float = 1000.0
    vel_scale: float = 30.0
