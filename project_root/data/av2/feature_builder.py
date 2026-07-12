"""Causal AV2 Pilot v1 feature assembly for existing PEFNet input contracts.

This module is the narrow adapter between :class:`AV2SensorEkfOutputs` and
the frozen AV2 feature schema.  It deliberately does not open AV2 files, run
an EKF, write shards, or alter the legacy Phase1R feature builders.  Features
are assembled only from current-or-earlier EKF outputs and measurements;
``target`` is copied through exclusively as supervision.

The EKF output contract exposes posterior, not prior, states.  Consequently
the measurement agreement values here are *causal post-fit residuals* (rather
than mislabeled innovations).  This is explicit in the public diagnostics and
keeps the adapter honest without changing the completed EKF interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from configs.av2_config import AV2_PILOT_V1, Av2PilotProtocol
from simulation.av2_sensor_ekf import AV2SensorEkfOutputs, SensorMeasurement
from simulation.measurement_models import h_aoa_only, h_gps2d, h_radar_rb, h_uwb_range_only, wrap_angle_rad

from .feature_schema import (
    AV2_FEATURE_SCHEMA_VERSION,
    EVIDENCE_SOURCE_IDS,
    POSTERIOR_SOURCE_IDS,
    AV2FeatureValidationReport,
    validate_av2_feature_arrays,
)
from .trajectory_transform import compute_eval_mask


DEFAULT_POSITION_SCALE: Final[float] = 1000.0
DEFAULT_VELOCITY_SCALE: Final[float] = 30.0
def _readonly_copy(values: np.ndarray, *, dtype: np.dtype | None = None) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class AV2FeatureDiagnostics:
    """Non-model arrays retained for uncertainty and causal-feature audits.

    ``posterior_covariance_internal`` is never used to create fusion-facing
    posterior features.  It remains available for EKF/NEES diagnostics, while
    the feature tensor is intentionally derived from ``reported`` covariance.
    ``postfit_residual_norm`` has ``[T, 5, 3]`` layout; only evidence rows
    (sensor indexes 3 and 4) are populated for all three posteriors.
    """

    posterior_covariance_internal: np.ndarray
    posterior_covariance_reported: np.ndarray
    postfit_residual_norm: np.ndarray


@dataclass(frozen=True)
class AV2FeatureArrays:
    """Complete single-trajectory AV2 feature set satisfying schema v1."""

    post_feat: np.ndarray
    post_mask: np.ndarray
    meas_feat: np.ndarray
    meas_mask: np.ndarray
    evidence_feat: np.ndarray
    evidence_mask: np.ndarray
    mp_pair_feat: np.ndarray
    target: np.ndarray
    timestamps_ns: np.ndarray
    eval_mask: np.ndarray
    loss_mask: np.ndarray
    diagnostics: AV2FeatureDiagnostics
    validation_report: AV2FeatureValidationReport

    def as_shard(self) -> dict[str, np.ndarray | str | tuple[str, ...]]:
        """Return only serializable, model-facing fields plus frozen metadata.

        Diagnostics intentionally stay outside the model shard boundary.  A
        caller that persists them for analysis must do so under an explicitly
        named diagnostics artifact rather than silently feeding them to PEFNet.
        """
        return {
            "feature_schema_version": AV2_FEATURE_SCHEMA_VERSION,
            "posterior_source_ids": POSTERIOR_SOURCE_IDS,
            "evidence_source_ids": EVIDENCE_SOURCE_IDS,
            "post_feat": self.post_feat,
            "post_mask": self.post_mask,
            "meas_feat": self.meas_feat,
            "meas_mask": self.meas_mask,
            "evidence_feat": self.evidence_feat,
            "evidence_mask": self.evidence_mask,
            "mp_pair_feat": self.mp_pair_feat,
            "target": self.target,
            "timestamps_ns": self.timestamps_ns,
            "eval_mask": self.eval_mask,
            "loss_mask": self.loss_mask,
        }


def _require_scale(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _validate_target(target: np.ndarray, steps: int) -> np.ndarray:
    array = np.asarray(target, dtype=np.float32)
    if array.shape != (steps, 4) or not np.all(np.isfinite(array)):
        raise ValueError(f"target must be a finite array with shape ({steps}, 4)")
    return array


def _measurement_prediction(
    sensor_index: int, state: np.ndarray, protocol: Av2PilotProtocol
) -> np.ndarray:
    if sensor_index == 0:
        return h_gps2d(state)
    position = protocol.sensors[sensor_index].local_position_xy_m
    if position is None:
        raise RuntimeError(f"sensor index {sensor_index} requires a local position")
    sx, sy = position
    if sensor_index in (1, 2):
        return h_radar_rb(state, sx, sy)
    if sensor_index == 3:
        return h_aoa_only(state, sx, sy)
    if sensor_index == 4:
        return h_uwb_range_only(state, sx, sy)
    raise RuntimeError(f"unknown AV2 sensor index {sensor_index}")


def _measurement_residual(
    sensor_index: int,
    measurement: SensorMeasurement,
    state: np.ndarray,
    protocol: Av2PilotProtocol,
) -> np.ndarray:
    if not measurement.valid or measurement.z is None:
        raise ValueError("a residual requires a valid measurement")
    residual = np.asarray(measurement.z, dtype=np.float64) - _measurement_prediction(
        sensor_index, state, protocol
    )
    if sensor_index in (1, 2, 3):
        residual[-1] = wrap_angle_rad(float(residual[-1]))
    return residual


def _reported_diag(measurement: SensorMeasurement) -> np.ndarray:
    if measurement.R_reported is None:
        raise ValueError("a valid measurement requires reported covariance")
    diagonal = np.diag(np.asarray(measurement.R_reported, dtype=np.float64))
    if diagonal.ndim != 1 or diagonal.size == 0 or not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
        raise ValueError("reported measurement covariance must have a finite positive diagonal")
    return diagonal


def _padded_two(values: np.ndarray) -> np.ndarray:
    result = np.zeros(2, dtype=np.float64)
    result[: min(2, values.size)] = values[:2]
    return result


def _causal_rolling_mean(values: np.ndarray, valid: np.ndarray, *, window: int = 30) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float32)
    for source in range(values.shape[1]):
        for step in range(values.shape[0]):
            start = max(0, step + 1 - window)
            keep = valid[start : step + 1, source] > 0.5
            if np.any(keep):
                output[step, source] = float(np.mean(values[start : step + 1, source][keep]))
    return output


def _peer_deviations(mean: np.ndarray, post_mask: np.ndarray, pos_scale: float, vel_scale: float) -> tuple[np.ndarray, np.ndarray]:
    steps = mean.shape[0]
    position = np.zeros((steps, 3), dtype=np.float32)
    velocity = np.zeros((steps, 3), dtype=np.float32)
    for step in range(steps):
        valid_indexes = np.flatnonzero(post_mask[step] > 0.5)
        for index in valid_indexes:
            peers = valid_indexes[valid_indexes != index]
            if peers.size == 0:
                continue
            centre = np.median(mean[step, peers], axis=0)
            position[step, index] = float(np.linalg.norm(mean[step, index, :2] - centre[:2]) / pos_scale)
            velocity[step, index] = float(np.linalg.norm(mean[step, index, 2:] - centre[2:]) / vel_scale)
    return position, velocity


def _canonical_pair_roles(steps: int, post_mask: np.ndarray, meas_mask: np.ndarray, evidence_mask: np.ndarray) -> np.ndarray:
    pair = np.zeros((steps, 3, 5, 8), dtype=np.float32)
    for posterior_index in range(3):
        for measurement_index in range(3):
            pair[:, posterior_index, measurement_index, 0 if posterior_index == measurement_index else 1] = 1.0
        pair[:, posterior_index, 3:, 2] = 1.0
    pair[..., 6] = post_mask[:, :, None]
    pair[..., 7] = np.concatenate((meas_mask, evidence_mask), axis=1)[:, None, :]
    return pair


def _fill_evidence_pair_features(
    pair: np.ndarray,
    residual_norm: np.ndarray,
    post_mask: np.ndarray,
    evidence_mask: np.ndarray,
) -> None:
    """Fill E1/E2 relation values independently for every posterior node."""
    for step in range(pair.shape[0]):
        for evidence_index in range(2):
            if evidence_mask[step, evidence_index] == 0.0:
                continue
            valid_posteriors = np.flatnonzero(post_mask[step] > 0.5)
            if valid_posteriors.size == 0:
                continue
            values = residual_norm[step, 3 + evidence_index, valid_posteriors]
            if not np.all(np.isfinite(values)):
                raise ValueError("valid evidence/posterior pairs must have finite residuals")
            log_values = np.log1p(np.maximum(values, 0.0))
            # Pair inputs must be local to (posterior, evidence).  Peer-centred
            # values and ranks make an untouched posterior row change when a
            # different posterior is perturbed, which violates the Level B+
            # locality contract.  These two bounded monotonic transforms use
            # only the current pair's own residual.
            bounded_log = np.tanh(log_values / 3.0)
            unit_score = log_values / (1.0 + log_values)
            measurement_index = 3 + evidence_index
            pair[step, valid_posteriors, measurement_index, 3] = log_values
            pair[step, valid_posteriors, measurement_index, 4] = bounded_log
            pair[step, valid_posteriors, measurement_index, 5] = unit_score


def build_av2_feature_arrays(
    outputs: AV2SensorEkfOutputs,
    target: np.ndarray,
    *,
    position_scale: float = DEFAULT_POSITION_SCALE,
    velocity_scale: float = DEFAULT_VELOCITY_SCALE,
    warmup_seconds: float = 1.0,
    protocol: Av2PilotProtocol = AV2_PILOT_V1,
) -> AV2FeatureArrays:
    """Assemble canonical AV2 feature arrays from one completed causal EKF run.

    ``target`` has no influence on any input feature, node mask, residual, or
    relation value.  It is copied only after the causal inputs are assembled.
    ``loss_mask`` is exactly the timestamp-derived warmup mask, so consumers
    must use it (or its same-valued ``eval_mask`` alias) for both loss and
    reporting rather than scoring the EKF warmup prefix.
    """
    protocol.validate()
    position_scale = _require_scale("position_scale", position_scale)
    velocity_scale = _require_scale("velocity_scale", velocity_scale)
    if not np.isfinite(warmup_seconds) or warmup_seconds < 0.0:
        raise ValueError("warmup_seconds must be finite and non-negative")

    timestamps = np.asarray(outputs.timestamps_ns, dtype=np.int64)
    steps = timestamps.size
    target_array = _validate_target(target, steps)
    mean = np.asarray(outputs.posterior_mean, dtype=np.float64)
    reported_covariance = np.asarray(outputs.posterior_covariance_reported, dtype=np.float64)
    internal_covariance = np.asarray(outputs.posterior_covariance_internal, dtype=np.float64)
    post_mask = np.asarray(outputs.posterior_available, dtype=np.float32)
    meas_mask = np.asarray(outputs.posterior_measurement_valid, dtype=np.float32)
    evidence_mask = np.asarray(outputs.evidence_valid, dtype=np.float32)

    post_feat = np.zeros((steps, 3, 9), dtype=np.float32)
    for step in range(steps):
        for posterior_index in range(3):
            if post_mask[step, posterior_index] == 0.0:
                continue
            state = mean[step, posterior_index]
            covariance = reported_covariance[step, posterior_index]
            if not np.all(np.isfinite(state)) or not np.all(np.isfinite(covariance)):
                raise ValueError("available posterior must have finite mean and reported covariance")
            diagonal = np.diag(covariance)
            if np.any(diagonal <= 0.0):
                raise ValueError("available posterior reported covariance must have positive diagonal")
            post_feat[step, posterior_index, :4] = (
                state[0] / position_scale,
                state[1] / position_scale,
                state[2] / velocity_scale,
                state[3] / velocity_scale,
            )
            post_feat[step, posterior_index, 4:8] = np.log1p(diagonal)
            post_feat[step, posterior_index, 8] = 1.0

    meas_feat = np.zeros((steps, 3, 18), dtype=np.float32)
    log_nis = np.zeros((steps, 3), dtype=np.float32)
    whitened_norm = np.zeros((steps, 3), dtype=np.float32)
    residual_norm = np.full((steps, 5, 3), np.nan, dtype=np.float64)
    for step in range(steps):
        records = outputs.measurements[step]
        for sensor_index in range(3):
            measurement = records[sensor_index]
            if bool(measurement.valid) != bool(meas_mask[step, sensor_index]):
                raise ValueError("posterior_measurement_valid must match measurement records")
            if meas_mask[step, sensor_index] == 0.0:
                continue
            if post_mask[step, sensor_index] == 0.0:
                raise ValueError("a valid track measurement requires its posterior to be available")
            residual = _measurement_residual(
                sensor_index, measurement, mean[step, sensor_index], protocol
            )
            diagonal = _reported_diag(measurement)
            white = residual / np.sqrt(diagonal)
            nis = float(np.dot(white, white))
            residual_norm[step, sensor_index, sensor_index] = float(np.linalg.norm(white))
            meas_feat[step, sensor_index, 0:2] = np.tanh(_padded_two(white) / 5.0)
            meas_feat[step, sensor_index, 2] = np.clip(np.log1p(nis), 0.0, 5.0)
            meas_feat[step, sensor_index, 3:5] = np.clip(np.log1p(_padded_two(diagonal)), 0.0, 8.0)
            sensor_position = protocol.sensors[sensor_index].local_position_xy_m
            if sensor_position is None:
                sensor_position = (0.0, 0.0)
            dx = (mean[step, sensor_index, 0] - sensor_position[0]) / position_scale
            dy = (mean[step, sensor_index, 1] - sensor_position[1]) / position_scale
            meas_feat[step, sensor_index, 5:8] = (dx, dy, np.hypot(dx, dy))
            meas_feat[step, sensor_index, 8 + sensor_index] = 1.0
            meas_feat[step, sensor_index, 12] = 1.0
            log_nis[step, sensor_index] = meas_feat[step, sensor_index, 2]
            whitened_norm[step, sensor_index] = float(np.linalg.norm(white))
    meas_feat[..., 13] = _causal_rolling_mean(log_nis, meas_mask)
    meas_feat[..., 14] = np.clip(_causal_rolling_mean(whitened_norm, meas_mask), 0.0, 20.0)
    peer_position, peer_velocity = _peer_deviations(mean, post_mask, position_scale, velocity_scale)
    meas_feat[..., 15] = np.where(meas_mask > 0.5, np.clip(peer_position, 0.0, 5.0), 0.0)
    meas_feat[..., 16] = np.where(meas_mask > 0.5, np.clip(peer_velocity, 0.0, 5.0), 0.0)
    meas_feat[..., 17] = meas_mask
    meas_feat[meas_mask == 0.0] = 0.0

    evidence_feat = np.zeros((steps, 2, 16), dtype=np.float32)
    for step in range(steps):
        records = outputs.measurements[step]
        valid_posteriors = np.flatnonzero(post_mask[step] > 0.5)
        consensus = np.median(mean[step, valid_posteriors], axis=0) if valid_posteriors.size else None
        for evidence_index in range(2):
            sensor_index = 3 + evidence_index
            measurement = records[sensor_index]
            if bool(measurement.valid) != bool(evidence_mask[step, evidence_index]):
                raise ValueError("evidence_valid must match measurement records")
            if evidence_mask[step, evidence_index] == 0.0:
                continue
            diagonal = _reported_diag(measurement)
            z = np.asarray(measurement.z, dtype=np.float64)
            evidence_feat[step, evidence_index, 0:2] = np.tanh(_padded_two(z / np.sqrt(diagonal)) / 10.0)
            evidence_feat[step, evidence_index, 2:4] = np.clip(np.log1p(_padded_two(diagonal)), 0.0, 8.0)
            evidence_feat[step, evidence_index, 4 + (2 + evidence_index)] = 1.0
            sensor_position = protocol.sensors[sensor_index].local_position_xy_m
            if sensor_position is None:
                raise RuntimeError("evidence sensor must have a fixed local position")
            evidence_feat[step, evidence_index, 8:10] = (
                sensor_position[0] / position_scale,
                sensor_position[1] / position_scale,
            )
            pair_residuals: list[float] = []
            for posterior_index in valid_posteriors:
                residual = _measurement_residual(
                    sensor_index, measurement, mean[step, posterior_index], protocol
                )
                norm = float(np.linalg.norm(residual / np.sqrt(diagonal)))
                residual_norm[step, sensor_index, posterior_index] = norm
                pair_residuals.append(norm)
            if consensus is not None:
                consensus_residual = _measurement_residual(
                    sensor_index, measurement, consensus, protocol
                )
                consensus_norm = float(np.linalg.norm(consensus_residual / np.sqrt(diagonal)))
                evidence_feat[step, evidence_index, 10] = np.clip(np.log1p(consensus_norm), 0.0, 6.0)
            if pair_residuals:
                evidence_feat[step, evidence_index, 11] = np.clip(np.log1p(np.mean(pair_residuals)), 0.0, 6.0)
                evidence_feat[step, evidence_index, 12] = np.clip(np.log1p(np.min(pair_residuals)), 0.0, 6.0)
            evidence_feat[step, evidence_index, 13] = 1.0
            evidence_feat[step, evidence_index, 14] = 1.0
            evidence_feat[step, evidence_index, 15] = float(evidence_index + 1) / 2.0

    pair = _canonical_pair_roles(steps, post_mask, meas_mask, evidence_mask)
    _fill_evidence_pair_features(pair, residual_norm, post_mask, evidence_mask)
    eval_mask = compute_eval_mask(timestamps, warmup_seconds=warmup_seconds).astype(np.float32)
    report = validate_av2_feature_arrays(
        post_feat=post_feat,
        post_mask=post_mask,
        meas_feat=meas_feat,
        meas_mask=meas_mask,
        evidence_feat=evidence_feat,
        evidence_mask=evidence_mask,
        mp_pair_feat=pair,
        target=target_array,
        timestamps_ns=timestamps,
        eval_mask=eval_mask,
        pair_specificity="aggregate",
    )
    diagnostics = AV2FeatureDiagnostics(
        posterior_covariance_internal=_readonly_copy(internal_covariance, dtype=np.float64),
        posterior_covariance_reported=_readonly_copy(reported_covariance, dtype=np.float64),
        postfit_residual_norm=_readonly_copy(residual_norm, dtype=np.float64),
    )
    return AV2FeatureArrays(
        post_feat=_readonly_copy(post_feat, dtype=np.float32),
        post_mask=_readonly_copy(post_mask, dtype=np.float32),
        meas_feat=_readonly_copy(meas_feat, dtype=np.float32),
        meas_mask=_readonly_copy(meas_mask, dtype=np.float32),
        evidence_feat=_readonly_copy(evidence_feat, dtype=np.float32),
        evidence_mask=_readonly_copy(evidence_mask, dtype=np.float32),
        mp_pair_feat=_readonly_copy(pair, dtype=np.float32),
        target=_readonly_copy(target_array, dtype=np.float32),
        timestamps_ns=_readonly_copy(timestamps, dtype=np.int64),
        eval_mask=_readonly_copy(eval_mask, dtype=np.float32),
        loss_mask=_readonly_copy(eval_mask, dtype=np.float32),
        diagnostics=diagnostics,
        validation_report=report,
    )
