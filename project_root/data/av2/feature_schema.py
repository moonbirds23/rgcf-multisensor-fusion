"""Frozen AV2 Pilot v1 feature-array contract.

This module is deliberately independent of the AV2 SDK, the sensor simulator,
and PEFNet.  It validates the boundary between feature-shard construction and
model input before a shard is written or loaded.  It does *not* construct
features and consequently cannot make an invalid feature set look valid.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np


AV2_FEATURE_SCHEMA_VERSION = "av2_pilot_feature_v1"
POSTERIOR_SOURCE_IDS = ("T1", "T2", "T3")
EVIDENCE_SOURCE_IDS = ("E1", "E2")

POST_FEATURE_DIM = 9
MEAS_FEATURE_DIM = 18
EVIDENCE_FEATURE_DIM = 16
PAIR_FEATURE_DIM = 8
NUM_POSTERIORS = 3
NUM_EVIDENCE = 2
NUM_MEASUREMENT_NODES = NUM_POSTERIORS + NUM_EVIDENCE


class AV2FeatureSchemaError(ValueError):
    """Raised when a feature shard violates the frozen Pilot v1 contract."""


@dataclass(frozen=True)
class AV2FeatureValidationReport:
    """Small audit record returned after all feature-array checks pass."""

    schema_version: str
    num_steps: int
    valid_posteriors: int
    valid_track_measurements: int
    valid_evidence_measurements: int


def _array(name: str, value: object, expected_shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != expected_shape:
        raise AV2FeatureSchemaError(
            f"{name} must have shape {expected_shape}, got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise AV2FeatureSchemaError(f"{name} must be a real numeric array, got {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise AV2FeatureSchemaError(f"{name} must contain only finite values")
    return array


def _binary_mask(name: str, value: object, expected_shape: tuple[int, ...]) -> np.ndarray:
    mask = _array(name, value, expected_shape).astype(np.float32, copy=False)
    if not np.all((mask == 0.0) | (mask == 1.0)):
        raise AV2FeatureSchemaError(f"{name} must contain exactly 0 or 1")
    return mask


def _require_equal(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if not np.array_equal(actual, expected):
        raise AV2FeatureSchemaError(f"{name} is inconsistent with its required mask or role layout")


def _require_masked_zero(name: str, values: np.ndarray, mask: np.ndarray) -> None:
    invalid = mask == 0.0
    if np.any(values[invalid] != 0.0):
        raise AV2FeatureSchemaError(
            f"{name} must be all zero where its node mask is zero; stale features are forbidden"
        )


def _validate_source_ids(
    posterior_source_ids: Sequence[str], evidence_source_ids: Sequence[str]
) -> None:
    if tuple(posterior_source_ids) != POSTERIOR_SOURCE_IDS:
        raise AV2FeatureSchemaError(
            "posterior_source_ids must be exactly ('T1', 'T2', 'T3'); "
            "E1/E2 may not become posterior nodes"
        )
    if tuple(evidence_source_ids) != EVIDENCE_SOURCE_IDS:
        raise AV2FeatureSchemaError(
            "evidence_source_ids must be exactly ('E1', 'E2'); evidence-only node roles are frozen"
        )


def _validate_pair_layout(
    pair: np.ndarray,
    post_mask: np.ndarray,
    meas_mask: np.ndarray,
    evidence_mask: np.ndarray,
    *,
    pair_specificity: str,
) -> None:
    # Pair channels: self-track, other-track, evidence, log-residual,
    # centred-residual, residual-rank, posterior-valid, measurement-valid.
    expected_roles = np.zeros((NUM_POSTERIORS, NUM_MEASUREMENT_NODES, 3), dtype=pair.dtype)
    for posterior_index in range(NUM_POSTERIORS):
        for measurement_index in range(NUM_POSTERIORS):
            expected_roles[posterior_index, measurement_index, 0 if measurement_index == posterior_index else 1] = 1.0
        expected_roles[posterior_index, NUM_POSTERIORS:, 2] = 1.0
    _require_equal("mp_pair_feat[..., 0:3]", pair[0, ..., 0:3], expected_roles)
    if not np.all(pair[..., 0:3] == expected_roles[None, ...]):
        raise AV2FeatureSchemaError("mp_pair_feat role channels must be time-invariant and canonical")

    expected_post_mask = np.broadcast_to(
        post_mask[:, :, None], (post_mask.shape[0], NUM_POSTERIORS, NUM_MEASUREMENT_NODES)
    )
    _require_equal("mp_pair_feat[..., 6]", pair[..., 6], expected_post_mask)
    measurement_masks = np.concatenate((meas_mask, evidence_mask), axis=1)
    expected_measurement_mask = np.broadcast_to(
        measurement_masks[:, None, :],
        (post_mask.shape[0], NUM_POSTERIORS, NUM_MEASUREMENT_NODES),
    )
    _require_equal("mp_pair_feat[..., 7]", pair[..., 7], expected_measurement_mask)

    # Version 1 reserves relation values for E1/E2 only.  Track pair relations
    # are carried by their node features, not by a second ambiguous encoding.
    if np.any(pair[:, :, :NUM_POSTERIORS, 3:6] != 0.0):
        raise AV2FeatureSchemaError("track measurement pair relation channels 3:6 must be zero in schema v1")

    valid_pair = expected_post_mask * expected_measurement_mask
    if np.any(pair[..., 3:6][valid_pair == 0.0] != 0.0):
        raise AV2FeatureSchemaError(
            "mp_pair_feat relation channels 3:6 must be zero unless both endpoints are valid"
        )

    if pair_specificity not in {"off", "aggregate", "per_timestep"}:
        raise ValueError("pair_specificity must be one of: 'off', 'aggregate', 'per_timestep'")
    if pair_specificity == "off":
        return

    # A common integration error is broadcasting an evidence feature to all
    # posterior nodes.  We cannot infer the physical residual from arrays
    # alone, but we can reject that structurally observable failure mode.
    for evidence_index in range(NUM_EVIDENCE):
        measurement_index = NUM_POSTERIORS + evidence_index
        differentiating_steps = 0
        eligible_steps = 0
        for time_index in range(pair.shape[0]):
            valid_post = post_mask[time_index] == 1.0
            if evidence_mask[time_index, evidence_index] != 1.0 or np.count_nonzero(valid_post) < 2:
                continue
            eligible_steps += 1
            relation = pair[time_index, valid_post, measurement_index, 3:6]
            if not np.all(relation == relation[0]):
                differentiating_steps += 1
            elif pair_specificity == "per_timestep":
                raise AV2FeatureSchemaError(
                    "evidence pair relations are broadcast across posterior nodes at "
                    f"time index {time_index}, evidence node E{evidence_index + 1}"
                )
        if pair_specificity == "aggregate" and eligible_steps and differentiating_steps == 0:
            raise AV2FeatureSchemaError(
                f"evidence node E{evidence_index + 1} has no posterior-specific pair relation; "
                "broadcast evidence features are forbidden"
            )


def validate_av2_feature_arrays(
    *,
    post_feat: object,
    post_mask: object,
    meas_feat: object,
    meas_mask: object,
    evidence_feat: object,
    evidence_mask: object,
    mp_pair_feat: object,
    target: Optional[object] = None,
    timestamps_ns: Optional[object] = None,
    eval_mask: Optional[object] = None,
    posterior_source_ids: Sequence[str] = POSTERIOR_SOURCE_IDS,
    evidence_source_ids: Sequence[str] = EVIDENCE_SOURCE_IDS,
    pair_specificity: str = "aggregate",
) -> AV2FeatureValidationReport:
    """Validate a complete AV2 Pilot v1 assembled feature tensor set.

    This is intentionally stricter than legacy Phase1R arrays: AV2 masked
    nodes must be zeroed, E1/E2 are evidence-only, and the relation values of
    evidence pairs must not be a broadcast copy across all posterior nodes.
    ``pair_specificity='per_timestep'`` is useful in synthetic builder tests;
    ``'aggregate'`` is the shard-level default and tolerates a genuine tie at
    an isolated time step.  No AV2 files are opened by this function.
    """
    post = np.asarray(post_feat)
    if post.ndim != 3:
        raise AV2FeatureSchemaError(f"post_feat must be rank 3, got rank {post.ndim}")
    if post.shape[0] <= 0:
        raise AV2FeatureSchemaError("post_feat must contain at least one time step")
    steps = post.shape[0]
    post = _array("post_feat", post, (steps, NUM_POSTERIORS, POST_FEATURE_DIM))
    post_valid = _binary_mask("post_mask", post_mask, (steps, NUM_POSTERIORS))
    meas = _array("meas_feat", meas_feat, (steps, NUM_POSTERIORS, MEAS_FEATURE_DIM))
    meas_valid = _binary_mask("meas_mask", meas_mask, (steps, NUM_POSTERIORS))
    evidence = _array("evidence_feat", evidence_feat, (steps, NUM_EVIDENCE, EVIDENCE_FEATURE_DIM))
    evidence_valid = _binary_mask("evidence_mask", evidence_mask, (steps, NUM_EVIDENCE))
    pair = _array(
        "mp_pair_feat",
        mp_pair_feat,
        (steps, NUM_POSTERIORS, NUM_MEASUREMENT_NODES, PAIR_FEATURE_DIM),
    )

    _validate_source_ids(posterior_source_ids, evidence_source_ids)
    _require_equal("post_feat[..., 8]", post[..., 8], post_valid)
    _require_equal("meas_feat[..., 12]", meas[..., 12], meas_valid)
    _require_equal("evidence_feat[..., 13]", evidence[..., 13], evidence_valid)
    _require_masked_zero("post_feat", post, post_valid)
    _require_masked_zero("meas_feat", meas, meas_valid)
    _require_masked_zero("evidence_feat", evidence, evidence_valid)
    _validate_pair_layout(
        pair, post_valid, meas_valid, evidence_valid, pair_specificity=pair_specificity
    )

    if target is not None:
        _array("target", target, (steps, 4))
    if timestamps_ns is not None:
        timestamps = np.asarray(timestamps_ns)
        if timestamps.shape != (steps,) or not np.issubdtype(timestamps.dtype, np.integer):
            raise AV2FeatureSchemaError("timestamps_ns must be an integer array with shape (T,)")
        if np.any(np.diff(timestamps) <= 0):
            raise AV2FeatureSchemaError("timestamps_ns must be strictly increasing")
    if eval_mask is not None:
        _binary_mask("eval_mask", eval_mask, (steps,))

    return AV2FeatureValidationReport(
        schema_version=AV2_FEATURE_SCHEMA_VERSION,
        num_steps=steps,
        valid_posteriors=int(np.count_nonzero(post_valid)),
        valid_track_measurements=int(np.count_nonzero(meas_valid)),
        valid_evidence_measurements=int(np.count_nonzero(evidence_valid)),
    )


def validate_av2_feature_shard(
    shard: Mapping[str, object], *, pair_specificity: str = "aggregate"
) -> AV2FeatureValidationReport:
    """Mapping-oriented wrapper used immediately before shard serialization."""
    required = (
        "post_feat", "post_mask", "meas_feat", "meas_mask", "evidence_feat",
        "evidence_mask", "mp_pair_feat",
    )
    missing = [name for name in required if name not in shard]
    if missing:
        raise AV2FeatureSchemaError(f"feature shard is missing required keys: {', '.join(missing)}")
    version = shard.get("feature_schema_version", AV2_FEATURE_SCHEMA_VERSION)
    if version != AV2_FEATURE_SCHEMA_VERSION:
        raise AV2FeatureSchemaError(
            f"feature_schema_version must be {AV2_FEATURE_SCHEMA_VERSION!r}, got {version!r}"
        )
    return validate_av2_feature_arrays(
        post_feat=shard["post_feat"], post_mask=shard["post_mask"],
        meas_feat=shard["meas_feat"], meas_mask=shard["meas_mask"],
        evidence_feat=shard["evidence_feat"], evidence_mask=shard["evidence_mask"],
        mp_pair_feat=shard["mp_pair_feat"], target=shard.get("target"),
        timestamps_ns=shard.get("timestamps_ns"), eval_mask=shard.get("eval_mask"),
        posterior_source_ids=shard.get("posterior_source_ids", POSTERIOR_SOURCE_IDS),
        evidence_source_ids=shard.get("evidence_source_ids", EVIDENCE_SOURCE_IDS),
        pair_specificity=pair_specificity,
    )
