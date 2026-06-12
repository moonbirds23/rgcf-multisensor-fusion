from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_name: str = "original_gnn_fusion"
    in_dim: int = 9
    post_in_dim: int = 9
    post_valid_idx: int = 8
    use_peer_consistency_features: bool = False
    zero_peer_consistency_features: bool = False
    peer_consistency_mode: str = "median"
    meas_in_dim: int = 18
    hidden_dim: int = 64
    meas_hidden_dim: int = 64
    gate_hidden_dim: int = 64
    gate_type: str = "soft_scalar"
    gate_on_meas_only: bool = True
    use_meas_in_representation: bool = True
    use_gate_on_meas_feature: bool = True
    gate_init_bias: float = 1.0
    gate_weight_alpha: float = 1.0
    gate_eps: float = 1e-4
    base_logit_temperature: float = 1.0
    weight_uniform_mix: float = 0.0
    output_fusion_mode: str = "info_diag"
    use_cov_calibration: bool = False
    use_cov_in_fusion: bool = False
    cov_weight_beta: float = 0.0
    cov_calib_min_scale: float = 1.0
    cov_calib_max_scale: float = 20.0
    cov_prior_weight: float = 0.0
    cov_sep_weight: float = 0.0
    cov_fault_normal_margin: float = 1.0
    fault_weight_loss_weight: float = 0.0
    fault_weight_margin: float = 0.1
    use_gate_supervision: bool = False
    use_balanced_gate_loss: bool = True
    gate_supervision_weight: float = 0.05
    gate_prior_weight: float = 0.005
    gate_prior_mean: float = 0.75
    normal_gate_target: float = 0.8
    fault_gate_target: float = 0.2
    use_error_aware_gate_target: bool = False
    gate_error_tau: float = 5.0
    gate_target_min: float = 0.1
    return_weights_during_eval: bool = True
    use_post_stream: bool = True
    use_meas_stream: bool = False
    use_gate: bool = False
    use_temporal: bool = False
    window_size: int = 6
    temporal_enc_type: str = "gru"
    fusion_variant: str = "post_only"
