from __future__ import annotations

from core.types import ExperimentBundle
from .gnn_fusion import OriginalGNNFusion, PostMeasDirectFusion, PostMeasSoftGateFusion, PostMeasWindowDirectFusion


def build_model_from_bundle(bundle: ExperimentBundle):
    name = bundle.model.model_name
    common = dict(
        hidden_dim=int(bundle.model.hidden_dim),
        valid_idx=int(bundle.model.post_valid_idx),
        pos_scale=float(bundle.scenario.pos_scale),
        vel_scale=float(bundle.scenario.vel_scale),
        output_fusion_mode=str(getattr(bundle.model, "output_fusion_mode", "info_diag")),
    )
    if name == "original_gnn_fusion":
        return OriginalGNNFusion(in_dim=int(bundle.model.post_in_dim), **common)
    if name == "post_meas_direct_fusion":
        return PostMeasDirectFusion(post_in_dim=int(bundle.model.post_in_dim), meas_in_dim=int(bundle.model.meas_in_dim), meas_hidden_dim=int(bundle.model.meas_hidden_dim), **common)
    if name == "post_meas_soft_gate_fusion":
        return PostMeasSoftGateFusion(
            post_in_dim=int(bundle.model.post_in_dim),
            meas_in_dim=int(bundle.model.meas_in_dim),
            meas_hidden_dim=int(bundle.model.meas_hidden_dim),
            gate_hidden_dim=int(bundle.model.gate_hidden_dim),
            gate_init_bias=float(bundle.model.gate_init_bias),
            gate_weight_alpha=float(bundle.model.gate_weight_alpha),
            gate_eps=float(bundle.model.gate_eps),
            base_logit_temperature=float(getattr(bundle.model, "base_logit_temperature", 1.0)),
            weight_uniform_mix=float(getattr(bundle.model, "weight_uniform_mix", 0.0)),
            use_meas_in_representation=bool(getattr(bundle.model, "use_meas_in_representation", True)),
            use_gate_on_meas_feature=bool(getattr(bundle.model, "use_gate_on_meas_feature", True)),
            use_cov_calibration=bool(bundle.model.use_cov_calibration),
            cov_calib_min_scale=float(bundle.model.cov_calib_min_scale),
            cov_calib_max_scale=float(bundle.model.cov_calib_max_scale),
            use_cov_in_fusion=bool(getattr(bundle.model, "use_cov_in_fusion", False)),
            cov_weight_beta=float(getattr(bundle.model, "cov_weight_beta", 0.0)),
            **common,
        )
    if name == "post_meas_window_direct_fusion":
        return PostMeasWindowDirectFusion(post_in_dim=int(bundle.model.post_in_dim), meas_in_dim=int(bundle.model.meas_in_dim), meas_hidden_dim=int(bundle.model.meas_hidden_dim), window_size=int(bundle.model.window_size), **common)
    raise ValueError(f"Unsupported model_name: {name}")
