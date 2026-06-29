from __future__ import annotations

from core.types import ExperimentBundle
from .gnn_fusion import MeasurementEvaluatedRGCFA0, MeasurementEvaluatedRGCFA0Directional, OriginalGNNFusion, Phase1RRGCF, PostMeasDirectFusion, PostMeasSoftGateFusion, PostMeasWindowDirectFusion, SkepticalNeuralFusionA


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
    if name == "skeptical_neural_fusion_a":
        return SkepticalNeuralFusionA(
            post_in_dim=int(bundle.model.post_in_dim),
            meas_in_dim=int(bundle.model.meas_in_dim),
            meas_hidden_dim=int(bundle.model.meas_hidden_dim),
            gate_hidden_dim=int(bundle.model.gate_hidden_dim),
            gate_init_bias=float(bundle.model.gate_init_bias),
            gate_weight_alpha=float(bundle.model.gate_weight_alpha),
            gate_eps=float(bundle.model.gate_eps),
            base_logit_temperature=float(getattr(bundle.model, "base_logit_temperature", 2.0)),
            weight_uniform_mix=float(getattr(bundle.model, "weight_uniform_mix", 0.02)),
            cov_calib_min_scale=float(bundle.model.cov_calib_min_scale),
            cov_calib_max_scale=float(bundle.model.cov_calib_max_scale),
            cov_weight_beta=float(getattr(bundle.model, "cov_weight_beta", 0.5)),
            cap_min=float(getattr(bundle.model, "snf_cap_min", 0.05)),
            cap_max=float(getattr(bundle.model, "snf_cap_max", 0.85)),
            quarantine_penalty=float(getattr(bundle.model, "snf_quarantine_penalty", 2.0)),
            **common,
        )
    if name == "phase1r_rgcf":
        return Phase1RRGCF(
            post_in_dim=int(bundle.model.post_in_dim),
            meas_in_dim=int(bundle.model.meas_in_dim),
            evidence_in_dim=int(getattr(bundle.model, "evidence_in_dim", 16)),
            meas_hidden_dim=int(bundle.model.meas_hidden_dim),
            gate_hidden_dim=int(bundle.model.gate_hidden_dim),
            gate_init_bias=float(bundle.model.gate_init_bias),
            gate_weight_alpha=float(bundle.model.gate_weight_alpha),
            gate_eps=float(bundle.model.gate_eps),
            base_logit_temperature=float(getattr(bundle.model, "base_logit_temperature", 1.5)),
            weight_uniform_mix=float(getattr(bundle.model, "weight_uniform_mix", 0.02)),
            cov_calib_min_scale=float(bundle.model.cov_calib_min_scale),
            cov_calib_max_scale=float(bundle.model.cov_calib_max_scale),
            use_cov_in_fusion=bool(getattr(bundle.model, "use_cov_in_fusion", True)),
            cov_weight_beta=float(getattr(bundle.model, "cov_weight_beta", 0.35)),
            **common,
        )
    if name == "me_rgcf_a0":
        return MeasurementEvaluatedRGCFA0(
            post_in_dim=int(bundle.model.post_in_dim),
            meas_in_dim=int(getattr(bundle.model, "me_rgcf_m_in_dim", bundle.model.meas_in_dim)),
            evidence_in_dim=int(getattr(bundle.model, "evidence_in_dim", 16)),
            meas_hidden_dim=int(bundle.model.meas_hidden_dim),
            gate_hidden_dim=int(bundle.model.gate_hidden_dim),
            gate_init_bias=float(bundle.model.gate_init_bias),
            gate_weight_alpha=float(bundle.model.gate_weight_alpha),
            gate_eps=float(bundle.model.gate_eps),
            base_logit_temperature=float(getattr(bundle.model, "base_logit_temperature", 1.5)),
            weight_uniform_mix=float(getattr(bundle.model, "weight_uniform_mix", 0.02)),
            cov_calib_min_scale=float(bundle.model.cov_calib_min_scale),
            cov_calib_max_scale=float(bundle.model.cov_calib_max_scale),
            use_cov_in_fusion=bool(getattr(bundle.model, "use_cov_in_fusion", True)),
            cov_weight_beta=float(getattr(bundle.model, "cov_weight_beta", 0.35)),
            use_mm_attention=bool(getattr(bundle.model, "me_rgcf_use_mm_attention", True)),
            use_mp_attention=bool(getattr(bundle.model, "me_rgcf_use_mp_attention", True)),
            **common,
        )
    if name in {"me_rgcf_a0_dir", "me_rgcf_a0_dir_hs"}:
        return MeasurementEvaluatedRGCFA0Directional(
            post_in_dim=int(bundle.model.post_in_dim),
            meas_in_dim=int(getattr(bundle.model, "me_rgcf_m_in_dim", bundle.model.meas_in_dim)),
            evidence_in_dim=int(getattr(bundle.model, "evidence_in_dim", 16)),
            pair_dim=int(getattr(bundle.model, "me_rgcf_pair_dim", 8)),
            meas_hidden_dim=int(bundle.model.meas_hidden_dim),
            gate_hidden_dim=int(bundle.model.gate_hidden_dim),
            gate_init_bias=float(bundle.model.gate_init_bias),
            gate_weight_alpha=float(bundle.model.gate_weight_alpha),
            gate_eps=float(bundle.model.gate_eps),
            base_logit_temperature=float(getattr(bundle.model, "base_logit_temperature", 1.5)),
            weight_uniform_mix=float(getattr(bundle.model, "weight_uniform_mix", 0.02)),
            cov_calib_min_scale=float(bundle.model.cov_calib_min_scale),
            cov_calib_max_scale=float(bundle.model.cov_calib_max_scale),
            cov_weight_beta=float(getattr(bundle.model, "cov_weight_beta", 0.35)),
            use_mm_attention=bool(getattr(bundle.model, "me_rgcf_use_mm_attention", True)),
            use_mp_attention=bool(getattr(bundle.model, "me_rgcf_use_mp_attention", True)),
            identity_bias_init=float(getattr(bundle.model, "me_rgcf_identity_bias_init", 0.5)),
            evidence_residual_bias_init=float(getattr(bundle.model, "me_rgcf_evidence_residual_bias_init", 0.25)),
            **common,
        )
    if name == "post_meas_window_direct_fusion":
        return PostMeasWindowDirectFusion(post_in_dim=int(bundle.model.post_in_dim), meas_in_dim=int(bundle.model.meas_in_dim), meas_hidden_dim=int(bundle.model.meas_hidden_dim), window_size=int(bundle.model.window_size), **common)
    raise ValueError(f"Unsupported model_name: {name}")
