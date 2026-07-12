"""Causal, timestamp-aware AV2 trajectory transformations.

The functions in this module operate only on an in-memory trajectory supplied
by a caller.  They neither scan a dataset nor import the AV2 SDK.  A complete
trajectory may be retained offline as reference truth, but ``causal_prefix``
is the object intended for a time-step estimator: at index ``k`` it contains
only samples through ``k``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _readonly_copy(values: np.ndarray, *, dtype: Optional[np.dtype] = None) -> np.ndarray:
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
    return timestamps


def _validate_matrix(name: str, values: np.ndarray, length: int) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (length, 2):
        raise ValueError(f"{name} must have shape ({length}, 2), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True)
class AV2Trajectory:
    """A focal-vehicle reference trajectory independent of AV2 file formats.

    ``timestamps_ns`` are strictly increasing, so every interval is derived
    from actual timestamps rather than an assumed 10 Hz sampling rate.
    ``headings_rad`` are optional only because some adapters may need to derive
    them from an AV2 schema field before constructing this object.  Local-frame
    conversion requires the initial heading explicitly or in this field.
    """

    timestamps_ns: np.ndarray
    positions_m: np.ndarray
    velocities_mps: np.ndarray
    observed: np.ndarray
    headings_rad: Optional[np.ndarray] = None
    scenario_id: str = ""
    focal_track_id: str = ""

    def __post_init__(self) -> None:
        timestamps = _validate_timestamps(self.timestamps_ns)
        length = timestamps.size
        positions = _validate_matrix("positions_m", self.positions_m, length)
        velocities = _validate_matrix("velocities_mps", self.velocities_mps, length)
        observed = np.asarray(self.observed, dtype=bool)
        if observed.shape != (length,):
            raise ValueError(f"observed must have shape ({length},), got {observed.shape}")

        headings = None
        if self.headings_rad is not None:
            headings = np.asarray(self.headings_rad, dtype=np.float64)
            if headings.shape != (length,):
                raise ValueError(
                    f"headings_rad must have shape ({length},), got {headings.shape}"
                )
            if not np.all(np.isfinite(headings)):
                raise ValueError("headings_rad must contain only finite values")

        object.__setattr__(self, "timestamps_ns", _readonly_copy(timestamps))
        object.__setattr__(self, "positions_m", _readonly_copy(positions, dtype=np.float64))
        object.__setattr__(self, "velocities_mps", _readonly_copy(velocities, dtype=np.float64))
        object.__setattr__(self, "observed", _readonly_copy(observed, dtype=bool))
        if headings is not None:
            object.__setattr__(self, "headings_rad", _readonly_copy(headings, dtype=np.float64))

    @property
    def num_steps(self) -> int:
        return int(self.timestamps_ns.size)

    @property
    def elapsed_seconds(self) -> np.ndarray:
        """Elapsed seconds relative to the first sample, as a fresh array."""
        return (self.timestamps_ns - self.timestamps_ns[0]).astype(np.float64) * 1e-9

    @property
    def dt_seconds(self) -> np.ndarray:
        """Consecutive timestamp intervals; its length is ``num_steps - 1``."""
        return np.diff(self.timestamps_ns).astype(np.float64) * 1e-9


def compute_eval_mask(timestamps_ns: np.ndarray, *, warmup_seconds: float = 1.0) -> np.ndarray:
    """Return the timestamp-based post-warmup mask without assuming a frame rate."""
    if not np.isfinite(warmup_seconds) or warmup_seconds < 0.0:
        raise ValueError("warmup_seconds must be a finite non-negative number")
    timestamps = _validate_timestamps(timestamps_ns)
    elapsed_seconds = (timestamps - timestamps[0]).astype(np.float64) * 1e-9
    return elapsed_seconds >= warmup_seconds


def to_initial_local_frame(
    trajectory: AV2Trajectory, *, initial_heading_rad: Optional[float] = None
) -> AV2Trajectory:
    """Transform positions and velocities into the frame at the first sample.

    The origin and rotation are calculated solely from the first sample and its
    heading.  This is safe for causal estimation: no later truth value affects
    any transformed input.  The input object and all its underlying arrays are
    never mutated.
    """
    heading = initial_heading_rad
    if heading is None:
        if trajectory.headings_rad is None:
            raise ValueError("initial_heading_rad is required when headings_rad is absent")
        heading = float(trajectory.headings_rad[0])
    if not np.isfinite(heading):
        raise ValueError("initial_heading_rad must be finite")

    cosine, sine = np.cos(heading), np.sin(heading)
    rotation_minus_heading = np.array([[cosine, sine], [-sine, cosine]])
    positions = (trajectory.positions_m - trajectory.positions_m[0]) @ rotation_minus_heading.T
    velocities = trajectory.velocities_mps @ rotation_minus_heading.T
    headings = None
    if trajectory.headings_rad is not None:
        # Heading is metadata rather than a model input, but retain it without
        # artificial +/-pi discontinuities for audits and motion grouping.
        # Subtract the unwrapped initial heading so crossing the branch cut is
        # represented as a continuous turn.
        unwrapped_heading = np.unwrap(trajectory.headings_rad)
        headings = unwrapped_heading - float(unwrapped_heading[0])

    return AV2Trajectory(
        timestamps_ns=trajectory.timestamps_ns,
        positions_m=positions,
        velocities_mps=velocities,
        observed=trajectory.observed,
        headings_rad=headings,
        scenario_id=trajectory.scenario_id,
        focal_track_id=trajectory.focal_track_id,
    )


def causal_prefix(trajectory: AV2Trajectory, inclusive_index: int) -> AV2Trajectory:
    """Return an immutable copy of samples 0 through ``inclusive_index`` only.

    This explicit prefix is the boundary used by online filtering code.  It
    prevents a caller from accidentally receiving a future sample through a
    trajectory object intended for a current time step.
    """
    if inclusive_index < 0 or inclusive_index >= trajectory.num_steps:
        raise IndexError(
            f"inclusive_index must be in [0, {trajectory.num_steps - 1}], got {inclusive_index}"
        )
    end = inclusive_index + 1
    headings = None if trajectory.headings_rad is None else trajectory.headings_rad[:end]
    return AV2Trajectory(
        timestamps_ns=trajectory.timestamps_ns[:end],
        positions_m=trajectory.positions_m[:end],
        velocities_mps=trajectory.velocities_mps[:end],
        observed=trajectory.observed[:end],
        headings_rad=headings,
        scenario_id=trajectory.scenario_id,
        focal_track_id=trajectory.focal_track_id,
    )
