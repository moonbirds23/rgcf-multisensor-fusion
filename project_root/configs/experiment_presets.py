from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from .base_config import BaseConfig
from .fault_config import DropoutFaultConfig, DropoutWindow, FaultConfig, PollutionFaultConfig
from .model_config import ModelConfig
from .scenario_config import ScenarioConfig
from .train_config import TrainConfig


@dataclass
class ExperimentConfig:
    name: str
    base: BaseConfig
    scenario: ScenarioConfig
    fault: FaultConfig
    model: ModelConfig
    train: TrainConfig


def _clean_fault() -> FaultConfig:
    return FaultConfig(mode="clean")


def _drop_fault() -> FaultConfig:
    return FaultConfig(
        mode="drop",
        dropout=DropoutFaultConfig(enabled=True, windows=[DropoutWindow(sensor_ids=[2], t0=95.0, t1=105.0)]),
    )


def _pollution_fault(sensor_ids=None, window=None, shape="jump") -> FaultConfig:
    return FaultConfig(
        mode="pollution",
        pollution=PollutionFaultConfig(
            enabled=True,
            target_sensor_ids=list(sensor_ids or [3]),
            bias_rw_sigma=0.03,
            jump_prob=1.0 if window is not None else 0.01,
            jump_sigma=12.0 if window is not None else 8.0,
            active_time_window=window,
            fault_shape=shape,
        ),
    )


def _both_fault() -> FaultConfig:
    f = _pollution_fault([3])
    f.mode = "both"
    f.dropout = _drop_fault().dropout
    return f


def _matrix_fault() -> FaultConfig:
    return FaultConfig(
        mode="robust_matrix_mixed",
        matrix_cases=[
            *({"mode": "clean", "sensor_id": None} for _ in range(4)),
            {"mode": "pollution", "sensor_id": 1}, {"mode": "pollution", "sensor_id": 2},
            {"mode": "pollution", "sensor_id": 3}, {"mode": "pollution", "sensor_id": 4},
            {"mode": "pollution", "sensor_id": 1}, {"mode": "pollution", "sensor_id": 3},
            {"mode": "bias_ramp", "sensor_id": 1}, {"mode": "bias_ramp", "sensor_id": 2},
            {"mode": "bias_ramp", "sensor_id": 3}, {"mode": "bias_ramp", "sensor_id": 4},
            {"mode": "bias_ramp", "sensor_id": 2}, {"mode": "bias_ramp", "sensor_id": 4},
            {"mode": "dropout", "sensor_id": 1}, {"mode": "dropout", "sensor_id": 2},
            {"mode": "dropout", "sensor_id": 3}, {"mode": "dropout", "sensor_id": 4},
        ],
        candidate_sensor_ids=[1, 2, 3, 4],
        random_target_by_seed=True,
        window=(30.0, 70.0),
    )


def _random_window_pollution_fault() -> FaultConfig:
    return FaultConfig(mode="random_window_pollution", candidate_sensor_ids=[1, 2, 3, 4], random_target_by_seed=True, window=(30.0, 70.0))


def _bias_ramp_fault() -> FaultConfig:
    return FaultConfig(mode="bias_ramp", candidate_sensor_ids=[1, 2, 3, 4], random_target_by_seed=True, window=(30.0, 70.0))


def _dropout_window_fault() -> FaultConfig:
    return FaultConfig(mode="dropout_window", candidate_sensor_ids=[1, 2, 3, 4], random_target_by_seed=True, window=(30.0, 70.0))


def _v0(model: ModelConfig) -> ModelConfig:
    out = deepcopy(model)
    out.model_name = "original_gnn_fusion"
    out.use_post_stream = True
    out.use_meas_stream = False
    out.use_gate = False
    out.fusion_variant = "post_only"
    return out


def _v1(model: ModelConfig) -> ModelConfig:
    out = deepcopy(model)
    out.model_name = "post_meas_direct_fusion"
    out.use_post_stream = True
    out.use_meas_stream = True
    out.use_gate = False
    out.fusion_variant = "post_meas_direct"
    return out


def _v2(model: ModelConfig) -> ModelConfig:
    out = _v1(model)
    out.model_name = "post_meas_soft_gate_fusion"
    out.use_gate = True
    out.gate_init_bias = 0.0
    out.fusion_variant = "post_meas_soft_gate"
    return out


def _rgcf(model: ModelConfig) -> ModelConfig:
    out = _v2(model)
    out.fusion_variant = "post_meas_soft_gate_rgcf"
    out.use_cov_calibration = True
    out.use_gate_supervision = True
    out.use_balanced_gate_loss = True
    out.gate_supervision_weight = 0.05
    out.gate_prior_weight = 0.005
    out.gate_prior_mean = 0.75
    out.normal_gate_target = 0.8
    out.fault_gate_target = 0.2
    return out


def _v3(model: ModelConfig) -> ModelConfig:
    out = _v1(model)
    out.model_name = "post_meas_window_direct_fusion"
    out.use_temporal = True
    out.window_size = 6
    out.fusion_variant = "post_meas_window_direct"
    return out


def _pack(name: str, scenario: ScenarioConfig, fault: FaultConfig, model: ModelConfig, train: TrainConfig | None = None) -> ExperimentConfig:
    return ExperimentConfig(name=name, base=BaseConfig(), scenario=deepcopy(scenario), fault=deepcopy(fault), model=deepcopy(model), train=deepcopy(train or TrainConfig()))


def build_experiment_config(preset_name: str) -> ExperimentConfig:
    scenario = ScenarioConfig()
    model = ModelConfig()
    train = TrainConfig()

    table = {
        "clean_baseline": (_clean_fault(), _v0(model), "default_clean_4sensor_scene"),
        "drop_only": (_drop_fault(), _v0(model), "default_clean_4sensor_scene"),
        "pollution_only": (_pollution_fault(), _v0(model), "default_clean_4sensor_scene"),
        "both_faults": (_both_fault(), _v0(model), "default_clean_4sensor_scene"),
        "clean_baseline_v0": (_clean_fault(), _v0(model), "default_clean_4sensor_scene"),
        "clean_v1_post_meas_direct": (_clean_fault(), _v1(model), "default_clean_4sensor_scene"),
        "clean_v2_post_meas_softgate": (_clean_fault(), _v2(model), "default_clean_4sensor_scene"),
        "clean_v3_window_direct": (_clean_fault(), _v3(model), "default_clean_4sensor_scene"),
        "hetero_v0_post_only": (_clean_fault(), _v0(model), "hetero_4sensor_scene"),
        "hetero_v1_post_meas_direct": (_clean_fault(), _v1(model), "hetero_4sensor_scene"),
        "hetero_v2_post_meas_softgate": (_clean_fault(), _v2(model), "hetero_4sensor_scene"),
        "hetero_window_pollution_v2_post_meas_softgate": (_pollution_fault([1], (30.0, 70.0)), _v2(model), "hetero_4sensor_scene"),
        "hetero_v2_rgcf": (_clean_fault(), _rgcf(model), "hetero_4sensor_scene"),
        "hetero_clean_v2_rgcf": (_clean_fault(), _rgcf(model), "hetero_4sensor_scene"),
        "hetero_window_pollution_v2_rgcf": (_pollution_fault([1], (30.0, 70.0)), _rgcf(model), "hetero_4sensor_scene"),
        "hetero_robust_matrix_mixed_v2_rgcf": (_matrix_fault(), _rgcf(model), "hetero_4sensor_scene"),
        "hetero_random_sensor_window_pollution_v2_rgcf": (_random_window_pollution_fault(), _rgcf(model), "hetero_4sensor_scene"),
        "hetero_bias_ramp_v2_rgcf": (_bias_ramp_fault(), _rgcf(model), "hetero_4sensor_scene"),
        "hetero_dropout_window_v2_rgcf": (_dropout_window_fault(), _rgcf(model), "hetero_4sensor_scene"),
        "hetero_robust_matrix_mixed_post_only_gnn": (_matrix_fault(), _v0(model), "hetero_4sensor_scene"),
        "hetero_robust_matrix_mixed_dual_stream_plain_gnn": (_matrix_fault(), _v1(model), "hetero_4sensor_scene"),
        "hetero_v3_window_direct": (_clean_fault(), _v3(model), "hetero_4sensor_scene"),
        "hetero_window_pollution_v3_window_direct": (_pollution_fault([1], (30.0, 70.0)), _v3(model), "hetero_4sensor_scene"),
    }
    if preset_name not in table:
        raise ValueError(f"Unknown preset_name: {preset_name}")
    fault, model_cfg, scene_name = table[preset_name]
    scenario.scene_name = scene_name
    if "robust_matrix_mixed" in preset_name:
        train.train_seed_start, train.train_seed_end = 10, 69
        train.val_seed_start, train.val_seed_end = 70, 89
        train.test_seed_start, train.test_seed_end = 90, 109
    return _pack(preset_name, scenario, fault, model_cfg, train)
