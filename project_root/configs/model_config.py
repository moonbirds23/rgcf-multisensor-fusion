from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_name: str = "original_gnn_fusion"
    in_dim: int = 9
    post_in_dim: int = 9
    post_valid_idx: int = 8
    meas_in_dim: int = 14
    hidden_dim: int = 64
    meas_hidden_dim: int = 64
    gate_hidden_dim: int = 64
    gate_type: str = "soft_scalar"
    gate_on_meas_only: bool = True
    gate_init_bias: float = 1.0
    gate_weight_alpha: float = 1.0
    gate_eps: float = 1e-4
    use_cov_calibration: bool = False
    cov_calib_min_scale: float = 1.0
    cov_calib_max_scale: float = 20.0
    use_gate_supervision: bool = False
    use_balanced_gate_loss: bool = True
    gate_supervision_weight: float = 0.05
    gate_prior_weight: float = 0.005
    gate_prior_mean: float = 0.75
    normal_gate_target: float = 0.8
    fault_gate_target: float = 0.2
    return_weights_during_eval: bool = True
    use_post_stream: bool = True
    use_meas_stream: bool = False
    use_gate: bool = False
    use_temporal: bool = False
    window_size: int = 6
    temporal_enc_type: str = "gru"
    fusion_variant: str = "post_only"
