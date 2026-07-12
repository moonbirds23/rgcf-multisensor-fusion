"""Deterministic, metadata-only AV2 motion classification.

Classification may use the complete reference trajectory for manifest
stratification and result grouping.  Its output must never be supplied as a
PEFNet input feature.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .trajectory_transform import AV2Trajectory


class MotionType(str, Enum):
    STRAIGHT = "straight"
    ACCELERATING = "accelerating"
    TURNING = "turning"
    HIGH_DYNAMIC = "high_dynamic"


@dataclass(frozen=True)
class MotionClassifierConfig:
    """Thresholds used for deterministic manifest metadata, in SI units."""

    turning_heading_change_rad: float = 0.35
    acceleration_speed_change_mps: float = 2.0
    high_dynamic_acceleration_mps2: float = 3.0

    def __post_init__(self) -> None:
        values = (
            self.turning_heading_change_rad,
            self.acceleration_speed_change_mps,
            self.high_dynamic_acceleration_mps2,
        )
        if not all(np.isfinite(values)) or any(value < 0.0 for value in values):
            raise ValueError("motion-classifier thresholds must be finite and non-negative")


@dataclass(frozen=True)
class MotionClassification:
    motion_type: MotionType
    valid_steps: int
    mean_speed_mps: float
    heading_change_rad: float
    speed_change_mps: float
    peak_acceleration_mps2: float


def _unwrap_heading_change(headings_rad: np.ndarray) -> float:
    return float(np.unwrap(headings_rad)[-1] - np.unwrap(headings_rad)[0])


def _segment_headings(positions_m: np.ndarray) -> np.ndarray:
    displacement = np.diff(positions_m, axis=0)
    if displacement.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    nonzero = np.linalg.norm(displacement, axis=1) > 1e-9
    return np.arctan2(displacement[nonzero, 1], displacement[nonzero, 0])


def classify_motion(
    trajectory: AV2Trajectory, *, config: MotionClassifierConfig = MotionClassifierConfig()
) -> MotionClassification:
    """Classify a complete reference trajectory with fixed priority ordering.

    Priority is ``high_dynamic`` > ``turning`` > ``accelerating`` >
    ``straight``.  This makes ties deterministic and ensures a high-acceleration
    turn is not double-counted in a single manifest stratum.
    """
    valid = trajectory.observed
    if not np.any(valid):
        raise ValueError("cannot classify a trajectory with no observed states")

    timestamps = trajectory.timestamps_ns[valid]
    positions = trajectory.positions_m[valid]
    velocities = trajectory.velocities_mps[valid]
    speed = np.linalg.norm(velocities, axis=1)
    valid_steps = int(speed.size)
    elapsed = (timestamps[-1] - timestamps[0]) * 1e-9
    total_distance = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    mean_speed = total_distance / elapsed if elapsed > 0.0 else 0.0
    speed_change = float(speed[-1] - speed[0])

    if trajectory.headings_rad is not None:
        heading_change = _unwrap_heading_change(trajectory.headings_rad[valid])
    else:
        headings = _segment_headings(positions)
        heading_change = _unwrap_heading_change(headings) if headings.size >= 2 else 0.0

    if valid_steps < 2:
        peak_acceleration = 0.0
    else:
        dt = np.diff(timestamps).astype(np.float64) * 1e-9
        speed_delta = np.diff(speed)
        peak_acceleration = float(np.max(np.abs(speed_delta / dt)))

    if peak_acceleration >= config.high_dynamic_acceleration_mps2:
        motion_type = MotionType.HIGH_DYNAMIC
    elif abs(heading_change) >= config.turning_heading_change_rad:
        motion_type = MotionType.TURNING
    elif abs(speed_change) >= config.acceleration_speed_change_mps:
        motion_type = MotionType.ACCELERATING
    else:
        motion_type = MotionType.STRAIGHT

    return MotionClassification(
        motion_type=motion_type,
        valid_steps=valid_steps,
        mean_speed_mps=mean_speed,
        heading_change_rad=heading_change,
        speed_change_mps=speed_change,
        peak_acceleration_mps2=peak_acceleration,
    )
