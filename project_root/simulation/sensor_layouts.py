from __future__ import annotations

from copy import deepcopy

from configs.scenario_config import SensorLayoutConfig, SensorNodeConfig


def build_default_clean_4sensor_layout() -> SensorLayoutConfig:
    """
    对齐当前 clean baseline 的默认 4 节点布局。
    这个布局与 Step 1 的 scenario_config 默认值保持一致。
    """
    return SensorLayoutConfig(
        sensors=[
            SensorNodeConfig(
                sensor_id=1,
                name="s1_gps",
                sensor_type="gps2d",
                position=(-800.0, -200.0),
                gps_sigma=5.0,
                gps_bias_rw_sigma=0.02,
            ),
            SensorNodeConfig(
                sensor_id=2,
                name="s2_radar",
                sensor_type="radar_rb",
                position=(900.0, -100.0),
                radar_sigma_r=3.0,
                radar_sigma_theta_deg=0.5,
            ),
            SensorNodeConfig(
                sensor_id=3,
                name="s3_gps_clean",
                sensor_type="gps2d",
                position=(-300.0, 900.0),
                gps_sigma=2.5,
                gps_bias_rw_sigma=0.03,
            ),
            SensorNodeConfig(
                sensor_id=4,
                name="s4_radar",
                sensor_type="radar_rb",
                position=(700.0, 850.0),
                radar_sigma_r=2.0,
                radar_sigma_theta_deg=0.35,
            ),
        ]
    )


def build_wide_baseline_4sensor_layout() -> SensorLayoutConfig:
    """
    更宽基线布局：
    用于后续测试更大空间几何分散情况下的融合表现。
    这里只改传感器位置，不改 nominal 传感器类型与噪声等级。
    """
    layout = build_default_clean_4sensor_layout()
    layout = deepcopy(layout)

    layout.sensors[0].position = (-1200.0, -500.0)
    layout.sensors[1].position = (1300.0, -300.0)
    layout.sensors[2].position = (-900.0, 1300.0)
    layout.sensors[3].position = (1200.0, 1200.0)

    return layout


def build_hetero_4sensor_layout() -> SensorLayoutConfig:
    """
    真异构4传感器布局：GPS + Radar + AOA + UWB

    设计目标：
    - 四种传感器量测空间完全不同，信息互补
    - AOA/UWB 单独无法定位，必须依赖融合
    - EKF 局部估计质量差异显著，post-stream 自然体现不均衡
    - 为 meas-stream gate 提供真实区分信号
    """
    return SensorLayoutConfig(
        sensors=[
            SensorNodeConfig(
                sensor_id=1,
                name="s1_gps",
                sensor_type="gps2d",
                position=(-800.0, -200.0),
                gps_sigma=5.0,
                gps_bias_rw_sigma=0.02,
            ),
            SensorNodeConfig(
                sensor_id=2,
                name="s2_radar",
                sensor_type="radar_rb",
                position=(900.0, -100.0),
                radar_sigma_r=3.0,
                radar_sigma_theta_deg=0.5,
            ),
            SensorNodeConfig(
                sensor_id=3,
                name="s3_aoa",
                sensor_type="aoa_only",
                position=(-300.0, 900.0),
                aoa_sigma_theta_deg=0.8,
            ),
            SensorNodeConfig(
                sensor_id=4,
                name="s4_uwb",
                sensor_type="uwb_range_only",
                position=(700.0, 850.0),
                uwb_sigma_r=1.5,
                uwb_far_r0=1300.0,
                uwb_far_k=0.002,
            ),
        ]
    )


def build_asymmetric_4sensor_layout() -> SensorLayoutConfig:
    """
    非对称布局：
    用于后续制造更明显的几何信息不均衡场景。
    """
    layout = build_default_clean_4sensor_layout()
    layout = deepcopy(layout)

    layout.sensors[0].position = (-1000.0, -100.0)
    layout.sensors[1].position = (500.0, -50.0)
    layout.sensors[2].position = (-200.0, 1200.0)
    layout.sensors[3].position = (1500.0, 300.0)

    return layout