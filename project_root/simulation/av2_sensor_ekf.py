"""Causal AV2 Pilot sensor simulation and independent CV-EKF track sources.

This module is deliberately separate from the Phase1R simulator.  It consumes
an in-memory local-frame truth trajectory and the frozen ``AV2_PILOT_V1``
configuration; it neither reads AV2 files nor builds PEFNet features or shards.
At index ``k`` it only consumes truth at ``k`` to create a synthetic
measurement, and a delayed measurement is always drawn from an earlier index.

T1/T2/T3 own independent filters and emit posterior states.  E1/E2 emit only
measurement evidence and are never filtered or exposed as posteriors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np

from configs.av2_config import AV2_PILOT_V1, Av2PilotProtocol, SensorSpec
from .ekf import CVEKF, EKFState
from .measurement_models import (
    H_gps2d,
    H_radar_rb,
    h_gps2d,
    h_radar_rb,
    wrap_angle_rad,
)


FaultKind = Literal["nominal", "bias_ramp", "covariance_mismatch", "dropout", "time_delay"]
_POSTERIOR_NAMES = ("T1", "T2", "T3")
_EVIDENCE_NAMES = ("E1", "E2")
_SENSOR_NAMES = _POSTERIOR_NAMES + _EVIDENCE_NAMES


def _readonly(values: np.ndarray, *, dtype: Optional[np.dtype] = None) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _validate_timestamps(timestamps_ns: np.ndarray) -> np.ndarray:
    timestamps = np.asarray(timestamps_ns)
    if timestamps.ndim != 1 or timestamps.size == 0:
        raise ValueError("timestamps_ns must be a non-empty one-dimensional array")
    if not np.issubdtype(timestamps.dtype, np.integer):
        raise TypeError("timestamps_ns must use an integer nanosecond dtype")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps_ns must be strictly increasing")
    return timestamps.astype(np.int64, copy=False)


def _sensor_noise(spec: SensorSpec, beta: float) -> np.ndarray:
    """Return physical measurement covariance, including frozen beta scaling."""
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("measurement_noise_beta must be finite and positive")
    standard_deviations = np.array([value for _, value in spec.noise_standard_deviations])
    # The frozen protocol defines a standard-deviation multiplier beta, hence
    # both physical and reported measurement covariance are R^(beta)=beta^2 R.
    return np.diag(standard_deviations**2 * beta**2).astype(np.float64)


def _is_due(elapsed_s: float, previous_elapsed_s: Optional[float], rate_hz: float) -> bool:
    """Timestamp-based rate gate that never assumes a fixed AV2 sample rate."""
    if previous_elapsed_s is None:
        return True
    # A sample is emitted when an elapsed-time cadence boundary was crossed.
    return int(np.floor(elapsed_s * rate_hz + 1e-12)) > int(
        np.floor(previous_elapsed_s * rate_hz + 1e-12)
    )


def _measurement_function(spec: SensorSpec, x: np.ndarray) -> np.ndarray:
    if spec.measurement_model == "gps_2d":
        return h_gps2d(x)
    sx, sy = spec.local_position_xy_m  # type: ignore[misc]
    if spec.measurement_model == "range_bearing":
        return h_radar_rb(x, sx, sy)
    if spec.measurement_model == "aoa_bearing":
        return np.array([np.arctan2(x[1] - sy, x[0] - sx)], dtype=np.float64)
    if spec.measurement_model == "uwb_range":
        return np.array([np.hypot(x[0] - sx, x[1] - sy)], dtype=np.float64)
    raise ValueError(f"unsupported AV2 sensor model {spec.measurement_model!r}")


def _angle_index(spec: SensorSpec) -> Optional[int]:
    if spec.measurement_model == "range_bearing":
        return 1
    if spec.measurement_model == "aoa_bearing":
        return 0
    return None


@dataclass(frozen=True)
class PrincipalFault:
    """A materialized single principal AV2 Pilot condition for one trajectory.

    ``terminal_bias`` is expressed in the target's native measurement space:
    GPS is ``[x, y]``; radar is ``[range, bearing]``; AOA and UWB have one
    component.  A bias ramp is zero before ``onset_seconds`` and persists at
    its terminal value after the ramp completes.
    """

    kind: FaultKind = "nominal"
    target_sensor: Optional[str] = None
    onset_seconds: float = 0.0
    duration_seconds: float = 0.0
    terminal_bias: Optional[np.ndarray] = None
    delay_steps: int = 0
    covariance_alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in {"nominal", "bias_ramp", "covariance_mismatch", "dropout", "time_delay"}:
            raise ValueError(f"unknown fault kind {self.kind!r}")
        if self.kind == "nominal":
            if self.target_sensor is not None:
                raise ValueError("nominal fault must not target a sensor")
            return
        if self.target_sensor not in _SENSOR_NAMES:
            raise ValueError("non-nominal fault must target T1/T2/T3/E1/E2")
        if not np.isfinite(self.onset_seconds) or self.onset_seconds < 0.0:
            raise ValueError("fault onset_seconds must be finite and non-negative")
        if self.kind in {"bias_ramp", "dropout"} and (
            not np.isfinite(self.duration_seconds) or self.duration_seconds <= 0.0
        ):
            raise ValueError(f"{self.kind} requires a positive duration_seconds")
        if self.kind == "bias_ramp":
            if self.terminal_bias is None:
                raise ValueError("bias_ramp requires terminal_bias")
            bias = np.asarray(self.terminal_bias, dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(bias)):
                raise ValueError("terminal_bias must be finite")
            object.__setattr__(self, "terminal_bias", _readonly(bias, dtype=np.float64))
        if self.kind == "time_delay" and self.delay_steps not in AV2_PILOT_V1.faults.delay_steps:
            raise ValueError("time_delay must use one of the frozen delay steps")
        if self.kind == "covariance_mismatch" and self.covariance_alpha not in AV2_PILOT_V1.faults.covariance_mismatch_alphas:
            raise ValueError("covariance_mismatch alpha is not frozen in AV2 Pilot v1.0")
        if self.kind == "covariance_mismatch" and self.target_sensor not in _POSTERIOR_NAMES:
            raise ValueError("covariance mismatch is defined only for posterior sources")

    def active(self, elapsed_seconds: float) -> bool:
        if self.kind == "nominal":
            return False
        if self.kind == "dropout":
            return self.onset_seconds <= elapsed_seconds < self.onset_seconds + self.duration_seconds
        return elapsed_seconds >= self.onset_seconds

    def bias_at(self, elapsed_seconds: float) -> Optional[np.ndarray]:
        if self.kind != "bias_ramp" or elapsed_seconds < self.onset_seconds:
            return None
        scale = min(1.0, (elapsed_seconds - self.onset_seconds) / self.duration_seconds)
        return np.asarray(self.terminal_bias, dtype=np.float64) * scale


@dataclass(frozen=True)
class SensorMeasurement:
    """One causal measurement/evidence record for a single timestamp."""

    sensor_name: str
    role: Literal["posterior", "evidence"]
    emitted: bool
    valid: bool
    z: Optional[np.ndarray]
    R_actual: Optional[np.ndarray]
    R_reported: Optional[np.ndarray]
    source_timestamp_ns: Optional[int]
    fault_active: bool

    def __post_init__(self) -> None:
        for name in ("z", "R_actual", "R_reported"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value, dtype=np.float64)
                object.__setattr__(self, name, _readonly(array, dtype=np.float64))
        if self.valid and (self.z is None or self.R_actual is None or self.R_reported is None):
            raise ValueError("a valid measurement must carry z, actual R, and reported R")
        if self.source_timestamp_ns is not None:
            object.__setattr__(self, "source_timestamp_ns", int(self.source_timestamp_ns))


@dataclass(frozen=True)
class AV2SensorEkfOutputs:
    """Numerical outputs only; no PEFNet feature construction occurs here."""

    timestamps_ns: np.ndarray
    posterior_mean: np.ndarray  # [K, 3, 4], NaN before first source update
    posterior_covariance_internal: np.ndarray  # [K, 3, 4, 4]
    posterior_covariance_reported: np.ndarray  # [K, 3, 4, 4]
    posterior_available: np.ndarray  # [K, 3]; remains true through track dropout
    posterior_measurement_valid: np.ndarray  # [K, 3]
    evidence_valid: np.ndarray  # [K, 2]
    measurements: tuple[tuple[SensorMeasurement, ...], ...]  # [K][T1..E2]

    def __post_init__(self) -> None:
        timestamps = _validate_timestamps(self.timestamps_ns)
        steps = timestamps.size
        expected = {
            "posterior_mean": (steps, 3, 4),
            "posterior_covariance_internal": (steps, 3, 4, 4),
            "posterior_covariance_reported": (steps, 3, 4, 4),
            "posterior_available": (steps, 3),
            "posterior_measurement_valid": (steps, 3),
            "evidence_valid": (steps, 2),
        }
        object.__setattr__(self, "timestamps_ns", _readonly(timestamps))
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
            object.__setattr__(self, name, _readonly(value))
        if len(self.measurements) != steps or any(len(row) != 5 for row in self.measurements):
            raise ValueError("measurements must have [K][5] records in T1/T2/T3/E1/E2 order")


def sample_principal_fault(
    rng: np.random.Generator, *, protocol: Av2PilotProtocol = AV2_PILOT_V1
) -> PrincipalFault:
    """Materialize exactly one training condition using only frozen ranges."""
    protocol.validate()
    labels, probabilities = zip(*protocol.faults.condition_probabilities)
    kind = str(rng.choice(labels, p=probabilities))
    if kind == "nominal":
        return PrincipalFault()

    specs = {spec.name: spec for spec in protocol.sensors}
    if kind == "bias_ramp":
        role = "posterior" if rng.random() < protocol.faults.bias_ramp.posterior_source_probability else "evidence"
        candidates = [spec.name for spec in protocol.sensors if spec.role == role]
        target = str(rng.choice(candidates))
        spec = specs[target]
        onset = rng.uniform(protocol.faults.bias_ramp.onset_seconds.minimum, protocol.faults.bias_ramp.onset_seconds.maximum)
        duration = rng.uniform(protocol.faults.bias_ramp.ramp_duration_seconds.minimum, protocol.faults.bias_ramp.ramp_duration_seconds.maximum)
        if target == "T1":
            magnitude = rng.uniform(protocol.faults.bias_ramp.gps_terminal_bias_m.minimum, protocol.faults.bias_ramp.gps_terminal_bias_m.maximum)
            terminal = magnitude * np.array([np.cos(rng.uniform(-np.pi, np.pi)), np.sin(rng.uniform(-np.pi, np.pi))])
        elif target in {"T2", "T3"}:
            terminal = np.zeros(2, dtype=np.float64)
            if rng.random() < protocol.faults.bias_ramp.radar_range_or_bearing_probability:
                terminal[0] = rng.choice((-1.0, 1.0)) * rng.uniform(protocol.faults.bias_ramp.radar_range_terminal_bias_m.minimum, protocol.faults.bias_ramp.radar_range_terminal_bias_m.maximum)
            else:
                magnitude = rng.uniform(protocol.faults.bias_ramp.radar_bearing_terminal_bias_deg.minimum, protocol.faults.bias_ramp.radar_bearing_terminal_bias_deg.maximum)
                terminal[1] = rng.choice((-1.0, 1.0)) * np.deg2rad(magnitude)
        elif target == "E1":
            magnitude = rng.uniform(protocol.faults.bias_ramp.aoa_terminal_bias_deg.minimum, protocol.faults.bias_ramp.aoa_terminal_bias_deg.maximum)
            terminal = np.array([rng.choice((-1.0, 1.0)) * np.deg2rad(magnitude)])
        else:
            magnitude = rng.uniform(protocol.faults.bias_ramp.uwb_terminal_bias_m.minimum, protocol.faults.bias_ramp.uwb_terminal_bias_m.maximum)
            terminal = np.array([rng.choice((-1.0, 1.0)) * magnitude])
        return PrincipalFault("bias_ramp", target, onset, duration, terminal)

    if kind == "dropout":
        return PrincipalFault(
            "dropout",
            str(rng.choice(_SENSOR_NAMES)),
            rng.uniform(protocol.faults.dropout_onset_seconds.minimum, protocol.faults.dropout_onset_seconds.maximum),
            rng.uniform(protocol.faults.dropout_duration_seconds.minimum, protocol.faults.dropout_duration_seconds.maximum),
        )
    if kind == "time_delay":
        return PrincipalFault("time_delay", str(rng.choice(_SENSOR_NAMES)), delay_steps=int(rng.choice(protocol.faults.delay_steps)))
    return PrincipalFault(
        "covariance_mismatch",
        str(rng.choice(_POSTERIOR_NAMES)),
        covariance_alpha=float(rng.choice(protocol.faults.covariance_mismatch_alphas)),
    )


def _initial_state(spec: SensorSpec, measurement: SensorMeasurement, protocol: Av2PilotProtocol) -> EKFState:
    """Causal first-measurement initialization; velocity is never inferred ahead."""
    z = np.asarray(measurement.z, dtype=np.float64)
    velocity = np.asarray(protocol.ekf.initial_velocity_mps, dtype=np.float64)
    velocity_covariance = protocol.ekf.initial_velocity_variance_m2ps2
    if spec.name == "T1":
        position = z
        P_position = np.diag(protocol.ekf.gps_initial_position_variance_m2)
    else:
        sx, sy = spec.local_position_xy_m  # type: ignore[misc]
        distance, bearing = z
        position = np.array([sx + distance * np.cos(bearing), sy + distance * np.sin(bearing)])
        jacobian = np.array(
            [[np.cos(bearing), -distance * np.sin(bearing)], [np.sin(bearing), distance * np.cos(bearing)]],
            dtype=np.float64,
        )
        P_position = jacobian @ np.asarray(measurement.R_actual) @ jacobian.T
        eigenvalues, eigenvectors = np.linalg.eigh(P_position)
        P_position = (eigenvectors * np.maximum(eigenvalues, protocol.ekf.radar_minimum_position_eigenvalue_m2)) @ eigenvectors.T
    state = np.array([position[0], position[1], velocity[0], velocity[1]], dtype=np.float64)
    covariance = np.zeros((4, 4), dtype=np.float64)
    covariance[:2, :2] = P_position
    covariance[2:, 2:] = np.eye(2) * velocity_covariance
    return EKFState(state, covariance)


def _update_track(ekf: CVEKF, state: EKFState, spec: SensorSpec, measurement: SensorMeasurement) -> EKFState:
    if spec.name == "T1":
        return ekf.update(state, np.asarray(measurement.z), np.asarray(measurement.R_actual), h_gps2d, H_gps2d)
    sx, sy = spec.local_position_xy_m  # type: ignore[misc]
    return ekf.update(
        state,
        np.asarray(measurement.z),
        np.asarray(measurement.R_actual),
        h_radar_rb,
        H_radar_rb,
        1,
        sx,
        sy,
    )


def run_av2_sensor_ekfs(
    timestamps_ns: np.ndarray,
    truth_states: np.ndarray,
    *,
    rng: np.random.Generator,
    fault: PrincipalFault = PrincipalFault(),
    measurement_noise_beta: float = 1.0,
    protocol: Av2PilotProtocol = AV2_PILOT_V1,
) -> AV2SensorEkfOutputs:
    """Generate all five sensor streams and causally filter T1/T2/T3.

    ``truth_states[k]`` is used only to generate the physical measurement at
    ``k``.  It is not used for filter initialization, prediction, update, or
    any later output.  The caller owns truth supervision and warmup masking.
    """
    protocol.validate()
    timestamps = _validate_timestamps(timestamps_ns)
    truth = np.asarray(truth_states, dtype=np.float64)
    if truth.shape != (timestamps.size, 4) or not np.all(np.isfinite(truth)):
        raise ValueError("truth_states must be a finite array with shape [K, 4]")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if fault.kind != "nominal" and fault.target_sensor not in _SENSOR_NAMES:
        raise ValueError("fault target is invalid")

    specs = tuple(protocol.sensors)
    count = timestamps.size
    mean = np.full((count, 3, 4), np.nan, dtype=np.float64)
    P_internal = np.full((count, 3, 4, 4), np.nan, dtype=np.float64)
    P_reported = np.full((count, 3, 4, 4), np.nan, dtype=np.float64)
    available = np.zeros((count, 3), dtype=bool)
    posterior_measurement_valid = np.zeros((count, 3), dtype=bool)
    evidence_valid = np.zeros((count, 2), dtype=bool)
    filter_states: list[Optional[EKFState]] = [None, None, None]
    # One entry per *timeline* step, including rate-gated non-emissions.  This
    # makes a delay of d mean exactly z[k-d], rather than the d-th earlier
    # emitted measurement of a lower-rate sensor.
    timeline_history: list[list[SensorMeasurement]] = [[] for _ in specs]
    records: list[tuple[SensorMeasurement, ...]] = []
    elapsed = (timestamps - timestamps[0]).astype(np.float64) * 1e-9

    for k, timestamp in enumerate(timestamps):
        previous_elapsed = None if k == 0 else float(elapsed[k - 1])
        current_records: list[SensorMeasurement] = []
        for sensor_index, spec in enumerate(specs):
            emitted = _is_due(float(elapsed[k]), previous_elapsed, spec.rate_hz)
            R_actual = _sensor_noise(spec, measurement_noise_beta) if emitted else None
            z: Optional[np.ndarray] = None
            if emitted:
                z = _measurement_function(spec, truth[k]) + rng.multivariate_normal(
                    np.zeros(len(R_actual)), R_actual
                )
                angle = _angle_index(spec)
                if angle is not None:
                    z[angle] = wrap_angle_rad(float(z[angle]))

            active = fault.target_sensor == spec.name and fault.active(float(elapsed[k]))
            if emitted and fault.kind == "bias_ramp" and active:
                z = z + fault.bias_at(float(elapsed[k]))  # type: ignore[operator]
                angle = _angle_index(spec)
                if angle is not None:
                    z[angle] = wrap_angle_rad(float(z[angle]))

            valid = emitted and not (fault.kind == "dropout" and active)
            source_timestamp: Optional[int] = int(timestamp) if valid else None
            R_reported = None if R_actual is None else R_actual.copy()
            raw = SensorMeasurement(
                spec.name,
                spec.role,  # type: ignore[arg-type]
                emitted,
                valid,
                z if valid else None,
                R_actual if valid else None,
                R_reported if valid else None,
                source_timestamp,
                active,
            )

            # Delay is an exact timeline-index lookup z[k-delay].  A source
            # step without an emitted valid measurement stays invalid: there is
            # no interpolation, reuse, or future backfill.
            final = raw
            if fault.kind == "time_delay" and fault.target_sensor == spec.name and emitted:
                delay = fault.delay_steps
                if k < delay or not timeline_history[sensor_index][k - delay].valid:
                    final = SensorMeasurement(spec.name, spec.role, True, False, None, None, None, None, True)  # type: ignore[arg-type]
                else:
                    earlier = timeline_history[sensor_index][k - delay]
                    final = SensorMeasurement(
                        spec.name, spec.role, True, True, earlier.z, earlier.R_actual, earlier.R_reported,
                        earlier.source_timestamp_ns, True,
                    )  # type: ignore[arg-type]
            timeline_history[sensor_index].append(raw)

            if fault.kind == "covariance_mismatch" and fault.target_sensor == spec.name and final.valid:
                # The EKF always uses R_actual and P_internal.  Only the final
                # fusion-facing posterior covariance is scaled below.
                final = SensorMeasurement(
                    final.sensor_name, final.role, final.emitted, final.valid, final.z,
                    final.R_actual, final.R_reported, final.source_timestamp_ns, True,
                )
            current_records.append(final)

        # Each posterior source predicts from its own prior state with actual
        # timestamp delta.  An unavailable measurement means prediction only.
        dt = None if k == 0 else float((timestamps[k] - timestamps[k - 1]) * 1e-9)
        for track_index, spec in enumerate(specs[:3]):
            measurement = current_records[track_index]
            state = filter_states[track_index]
            if state is None:
                if measurement.valid:
                    state = _initial_state(spec, measurement, protocol)
            else:
                state = CVEKF(dt=dt, sigma_a=protocol.ekf.acceleration_sigma_mps2).predict(state)  # type: ignore[arg-type]
                if measurement.valid:
                    state = _update_track(CVEKF(dt=dt, sigma_a=protocol.ekf.acceleration_sigma_mps2), state, spec, measurement)  # type: ignore[arg-type]
            filter_states[track_index] = state
            posterior_measurement_valid[k, track_index] = measurement.valid
            if state is not None:
                available[k, track_index] = True
                mean[k, track_index] = state.x
                P_internal[k, track_index] = state.P
                alpha = fault.covariance_alpha if fault.kind == "covariance_mismatch" and fault.target_sensor == spec.name else 1.0
                P_reported[k, track_index] = alpha * state.P
        evidence_valid[k] = [current_records[3].valid, current_records[4].valid]
        records.append(tuple(current_records))

    return AV2SensorEkfOutputs(
        timestamps, mean, P_internal, P_reported, available,
        posterior_measurement_valid, evidence_valid, tuple(records),
    )
