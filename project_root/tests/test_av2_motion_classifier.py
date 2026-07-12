from __future__ import annotations

import unittest

import numpy as np

from data.av2.motion_classifier import MotionType, classify_motion
from data.av2.trajectory_transform import AV2Trajectory


def make_trajectory(*, headings: np.ndarray, velocities: np.ndarray) -> AV2Trajectory:
    count = len(headings)
    timestamps = np.arange(count, dtype=np.int64) * 1_000_000_000
    positions = np.column_stack((np.arange(count, dtype=float), np.zeros(count)))
    return AV2Trajectory(
        timestamps_ns=timestamps,
        positions_m=positions,
        velocities_mps=velocities,
        observed=np.ones(count, dtype=bool),
        headings_rad=headings,
    )


class MotionClassifierTests(unittest.TestCase):
    def test_deterministic_straight_and_turning_categories(self) -> None:
        straight = make_trajectory(
            headings=np.zeros(4), velocities=np.tile([1.0, 0.0], (4, 1))
        )
        turning = make_trajectory(
            headings=np.array([0.0, 0.2, 0.4, 0.6]), velocities=np.tile([1.0, 0.0], (4, 1))
        )
        self.assertEqual(classify_motion(straight).motion_type, MotionType.STRAIGHT)
        self.assertEqual(classify_motion(turning).motion_type, MotionType.TURNING)
        self.assertEqual(classify_motion(turning), classify_motion(turning))

    def test_high_dynamic_has_priority_over_other_categories(self) -> None:
        trajectory = make_trajectory(
            headings=np.array([0.0, 0.3, 0.6, 0.9]),
            velocities=np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 0.0], [4.0, 0.0]]),
        )
        self.assertEqual(classify_motion(trajectory).motion_type, MotionType.HIGH_DYNAMIC)


if __name__ == "__main__":
    unittest.main()
