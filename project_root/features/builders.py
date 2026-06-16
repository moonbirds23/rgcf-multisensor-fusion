from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from core.types import ExperimentBundle
from .meas_features import EvidenceFeatureOutput, MeasFeatureOutput, build_evidence_node_features_from_sim, build_meas_node_features_from_sim, build_mp_pair_features_from_sim
from .post_features import PostFeatureOutput, build_post_node_features_from_sim

PHASE1R_EVIDENCE_MODELS = {"phase1r_rgcf", "me_rgcf_a0", "me_rgcf_a0_dir"}
PHASE1R_PAIR_MODELS = {"me_rgcf_a0_dir"}


@dataclass
class FeatureBundle:
    post: PostFeatureOutput
    meas: Optional[MeasFeatureOutput]
    evidence: Optional[EvidenceFeatureOutput]
    mp_pair: Optional[np.ndarray]
    target: np.ndarray
    t: np.ndarray


def build_feature_bundle_from_sim(sim: Dict[str, np.ndarray], bundle: ExperimentBundle) -> FeatureBundle:
    model_name = str(getattr(bundle.model, "model_name", ""))
    post = build_post_node_features_from_sim(sim, bundle)
    meas = build_meas_node_features_from_sim(sim, bundle) if bool(getattr(bundle.model, "use_meas_stream", False)) or bool(getattr(bundle.model, "use_gate", False)) or bool(getattr(bundle.model, "use_temporal", False)) else None
    evidence = build_evidence_node_features_from_sim(sim, bundle) if model_name in PHASE1R_EVIDENCE_MODELS else None
    mp_pair = build_mp_pair_features_from_sim(sim, bundle) if model_name in PHASE1R_PAIR_MODELS else None
    return FeatureBundle(post=post, meas=meas, evidence=evidence, mp_pair=mp_pair, target=post.target, t=post.t)
