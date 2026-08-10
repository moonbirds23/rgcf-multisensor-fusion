"""Deterministic CI-EU and centralized multisensor EKF baselines.

The functions in this module consume the same cached AV2 posterior and raw
measurement records as PEFNet.  They never read truth and contain no learned
parameters or test-time tuning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from configs.av2_config import Av2PilotProtocol
from simulation.av2_sensor_ekf import AV2SensorEkfOutputs, SensorMeasurement
from simulation.ekf import CVEKF, EKFState
from simulation.fusion_baselines import ci_fuse_two
from simulation.measurement_models import (
    H_aoa_only,
    H_gps2d,
    H_radar_rb,
    H_uwb_range_only,
    h_aoa_only,
    h_gps2d,
    h_radar_rb,
    h_uwb_range_only,
    wrap_angle_rad,
)


@dataclass(frozen=True)
class CIEUSequenceResult:
    xhat: np.ndarray
    Phat: np.ndarray
    valid_mask: np.ndarray
    ci_weights: np.ndarray
    evidence_dim: np.ndarray
    nis: np.ndarray


@dataclass(frozen=True)
class CentralizedEKFResult:
    xhat: np.ndarray
    Phat: np.ndarray
    valid_mask: np.ndarray
    active_measurement_dims: np.ndarray
    nis: np.ndarray


@dataclass(frozen=True)
class _MeasurementBlock:
    sensor_index: int
    z: np.ndarray
    R: np.ndarray


def _block_prediction(
    block: _MeasurementBlock,
    state: np.ndarray,
    protocol: Av2PilotProtocol,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    sensor_index = block.sensor_index
    spec = protocol.sensors[sensor_index]
    if sensor_index == 0:
        return h_gps2d(state), H_gps2d(state), ()
    position = spec.local_position_xy_m
    if position is None:
        raise RuntimeError(f"{spec.name}: fixed sensor position is required")
    sx, sy = position
    if sensor_index in (1, 2):
        return h_radar_rb(state, sx, sy), H_radar_rb(state, sx, sy), (1,)
    if sensor_index == 3:
        return h_aoa_only(state, sx, sy), H_aoa_only(state, sx, sy), (0,)
    if sensor_index == 4:
        return h_uwb_range_only(state, sx, sy), H_uwb_range_only(state, sx, sy), ()
    raise RuntimeError(f"unsupported sensor index {sensor_index}")


def _block_diagonal(blocks: Sequence[np.ndarray]) -> np.ndarray:
    dimension = sum(block.shape[0] for block in blocks)
    result = np.zeros((dimension, dimension), dtype=np.float64)
    offset = 0
    for block in blocks:
        size = block.shape[0]
        result[offset : offset + size, offset : offset + size] = block
        offset += size
    return result


def _joint_update(
    state: EKFState,
    blocks: Sequence[_MeasurementBlock],
    protocol: Av2PilotProtocol,
) -> tuple[EKFState, float]:
    """Apply one joint EKF update with all Jacobians at the same state."""
    if not blocks:
        return EKFState(state.x.copy(), state.P.copy()), float("nan")
    predictions: list[np.ndarray] = []
    jacobians: list[np.ndarray] = []
    angle_indexes: list[int] = []
    offset = 0
    for block in blocks:
        prediction, jacobian, local_angles = _block_prediction(block, state.x, protocol)
        predictions.append(prediction)
        jacobians.append(jacobian)
        angle_indexes.extend(offset + index for index in local_angles)
        offset += prediction.size
    z = np.concatenate([block.z for block in blocks]).astype(np.float64, copy=False)
    zhat = np.concatenate(predictions)
    H = np.vstack(jacobians)
    R = _block_diagonal([block.R for block in blocks])
    innovation = z - zhat
    for index in angle_indexes:
        innovation[index] = wrap_angle_rad(float(innovation[index]))
    P = np.asarray(state.P, dtype=np.float64)
    S = H @ P @ H.T + R
    PHt = P @ H.T
    gain = np.linalg.solve(S, PHt.T).T
    updated_x = state.x + gain @ innovation
    identity = np.eye(P.shape[0], dtype=np.float64)
    residual_map = identity - gain @ H
    updated_P = residual_map @ P @ residual_map.T + gain @ R @ gain.T
    updated_P = 0.5 * (updated_P + updated_P.T)
    nis = float(innovation @ np.linalg.solve(S, innovation))
    if not np.all(np.isfinite(updated_x)) or not np.all(np.isfinite(updated_P)):
        raise FloatingPointError("joint EKF update produced non-finite output")
    return EKFState(updated_x, updated_P), nis


def _measurement_block(
    record: SensorMeasurement,
    sensor_index: int,
) -> _MeasurementBlock | None:
    if not record.valid:
        return None
    if record.z is None or record.R_actual is None:
        raise ValueError("valid cached measurement is missing z or R_actual")
    z = np.asarray(record.z, dtype=np.float64).reshape(-1)
    R = np.asarray(record.R_actual, dtype=np.float64)
    if R.shape != (z.size, z.size):
        raise ValueError("measurement covariance shape mismatch")
    return _MeasurementBlock(sensor_index, z, R)


def fuse_av2_ci_tracks(
    states: np.ndarray,
    covariances: np.ndarray,
    valid: np.ndarray,
    *,
    n_grid: int = 31,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
    """Reproduce the frozen AV2 sequential pairwise CI implementation."""
    states = np.asarray(states, dtype=np.float64)
    covariances = np.asarray(covariances, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    indexes = np.flatnonzero(valid)
    weights = np.zeros(3, dtype=np.float64)
    if indexes.size == 0:
        return None, None, weights
    first = int(indexes[0])
    fused_x = states[first].copy()
    fused_P = covariances[first].copy()
    weights[first] = 1.0
    for index_value in indexes[1:]:
        index = int(index_value)
        fused_x, fused_P, prior_weight = ci_fuse_two(
            fused_x,
            fused_P,
            states[index],
            covariances[index],
            objective="logdet",
            n_grid=n_grid,
        )
        weights *= prior_weight
        weights[index] = 1.0 - prior_weight
    return fused_x, fused_P, weights


def run_ci_eu_sequence(
    outputs: AV2SensorEkfOutputs,
    protocol: Av2PilotProtocol,
    *,
    ci_grid_points: int = 31,
) -> CIEUSequenceResult:
    """Fuse T1/T2/T3 with AV2 CI, then jointly update with E1/E2."""
    steps = len(outputs.timestamps_ns)
    xhat = np.full((steps, 4), np.nan, dtype=np.float64)
    Phat = np.full((steps, 4, 4), np.nan, dtype=np.float64)
    valid_mask = np.zeros(steps, dtype=bool)
    weights = np.zeros((steps, 3), dtype=np.float64)
    evidence_dim = np.zeros(steps, dtype=np.int64)
    nis = np.full(steps, np.nan, dtype=np.float64)
    for step in range(steps):
        fused_x, fused_P, step_weights = fuse_av2_ci_tracks(
            outputs.posterior_mean[step],
            outputs.posterior_covariance_reported[step],
            outputs.posterior_available[step],
            n_grid=ci_grid_points,
        )
        weights[step] = step_weights
        if fused_x is None or fused_P is None:
            continue
        blocks = [
            block
            for sensor_index in (3, 4)
            if (block := _measurement_block(outputs.measurements[step][sensor_index], sensor_index))
            is not None
        ]
        state, step_nis = _joint_update(EKFState(fused_x, fused_P), blocks, protocol)
        xhat[step], Phat[step] = state.x, state.P
        valid_mask[step] = True
        evidence_dim[step] = sum(block.z.size for block in blocks)
        nis[step] = step_nis
    return CIEUSequenceResult(xhat, Phat, valid_mask, weights, evidence_dim, nis)


def run_centralized_multisensor_ekf(
    outputs: AV2SensorEkfOutputs,
    protocol: Av2PilotProtocol,
    *,
    enabled_sensors: Sequence[bool] = (True, True, True, True, True),
) -> CentralizedEKFResult:
    """Run the formal joint-update CM-EKF from cached five-sensor measurements."""
    enabled = np.asarray(enabled_sensors, dtype=bool)
    if enabled.shape != (5,):
        raise ValueError("enabled_sensors must contain five booleans")
    steps = len(outputs.timestamps_ns)
    xhat = np.full((steps, 4), np.nan, dtype=np.float64)
    Phat = np.full((steps, 4, 4), np.nan, dtype=np.float64)
    valid_mask = np.zeros(steps, dtype=bool)
    active_dims = np.zeros(steps, dtype=np.int64)
    nis = np.full(steps, np.nan, dtype=np.float64)
    state: EKFState | None = None
    for step in range(steps):
        records = outputs.measurements[step]
        initialized_with_t1 = False
        if state is None:
            t1 = records[0]
            if not enabled[0] or not t1.valid:
                continue
            if t1.z is None:
                raise ValueError("valid T1 initialization measurement has no value")
            position = np.asarray(t1.z, dtype=np.float64)
            velocity = np.asarray(protocol.ekf.initial_velocity_mps, dtype=np.float64)
            covariance = np.zeros((4, 4), dtype=np.float64)
            covariance[:2, :2] = np.diag(protocol.ekf.gps_initial_position_variance_m2)
            covariance[2:, 2:] = (
                np.eye(2, dtype=np.float64) * protocol.ekf.initial_velocity_variance_m2ps2
            )
            state = EKFState(np.r_[position, velocity], covariance)
            initialized_with_t1 = True
        else:
            dt = float((outputs.timestamps_ns[step] - outputs.timestamps_ns[step - 1]) * 1e-9)
            state = CVEKF(dt=dt, sigma_a=protocol.ekf.acceleration_sigma_mps2).predict(state)

        blocks: list[_MeasurementBlock] = []
        for sensor_index, record in enumerate(records):
            if not enabled[sensor_index] or (initialized_with_t1 and sensor_index == 0):
                continue
            block = _measurement_block(record, sensor_index)
            if block is not None:
                blocks.append(block)
        state, step_nis = _joint_update(state, blocks, protocol)
        xhat[step], Phat[step] = state.x, state.P
        valid_mask[step] = True
        active_dims[step] = sum(block.z.size for block in blocks)
        nis[step] = step_nis
    return CentralizedEKFResult(xhat, Phat, valid_mask, active_dims, nis)
