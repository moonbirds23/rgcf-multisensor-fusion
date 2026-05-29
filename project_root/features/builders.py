from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from core.types import ExperimentBundle
from .meas_features import MeasFeatureOutput, build_meas_node_features_from_sim
from .post_features import PostFeatureOutput, build_post_node_features_from_sim


@dataclass
class FeatureBundle:
    post: PostFeatureOutput
    meas: Optional[MeasFeatureOutput]
    target: np.ndarray
    t: np.ndarray


def build_feature_bundle_from_sim(sim: Dict[str, np.ndarray], bundle: ExperimentBundle) -> FeatureBundle:
    post = build_post_node_features_from_sim(sim, bundle)
    meas = build_meas_node_features_from_sim(sim, bundle) if bool(getattr(bundle.model, "use_meas_stream", False)) or bool(getattr(bundle.model, "use_gate", False)) or bool(getattr(bundle.model, "use_temporal", False)) else None
    return FeatureBundle(post=post, meas=meas, target=post.target, t=post.t)
