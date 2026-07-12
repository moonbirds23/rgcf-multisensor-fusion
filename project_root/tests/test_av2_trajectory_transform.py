from __future__ import annotations

import unittest

import numpy as np

from data.av2.trajectory_transform import (
    AV2Trajectory,
    causal_prefix,
    compute_eval_mask,
    to_initial_local_frame,
)


class AV2TrajectoryTransformTests(unittest.TestCase):
    def _trajectory(self) -> AV2Trajectory:
        return AV2Trajectory(
            timestamps_ns=np.array([1_000_000_000, 1_100_000_000, 1_300_000_000, 2_050_000_000]),
            positions_m=np.array([[10.0, 20.0], [10.0, 21.0], [9.0, 21.0], [7.0, 21.0]]),
            velocities_mps=np.array([[0.0, 2.0], [0.0, 2.0], [-2.0, 0.0], [-2.0, 0.0]]),
            observed=np.array([True, True, True, True]),
            headings_rad=np.full(4, np.pi / 2.0),
            scenario_id="synthetic",
            focal_track_id="focal",
        )

    def test_initial_origin_heading_rotates_positions_and_velocities(self) -> None:
        local = to_initial_local_frame(self._trajectory())
        np.testing.assert_allclose(local.positions_m[0], [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(local.positions_m[1], [1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(local.velocities_mps[0], [2.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(local.velocities_mps[2], [0.0, 2.0], atol=1e-12)
        np.testing.assert_allclose(local.headings_rad, np.zeros(4), atol=1e-12)

    def test_transform_does_not_mutate_input_or_share_mutable_arrays(self) -> None:
        trajectory = self._trajectory()
        original_positions = trajectory.positions_m.copy()
        original_velocities = trajectory.velocities_mps.copy()
        local = to_initial_local_frame(trajectory)
        np.testing.assert_array_equal(trajectory.positions_m, original_positions)
        np.testing.assert_array_equal(trajectory.velocities_mps, original_velocities)
        self.assertFalse(np.shares_memory(trajectory.positions_m, local.positions_m))
        with self.assertRaises(ValueError):
            local.positions_m[0, 0] = 99.0

    def test_irregular_timestamps_define_dt_and_warmup_mask(self) -> None:
        trajectory = self._trajectory()
        np.testing.assert_allclose(trajectory.dt_seconds, [0.1, 0.2, 0.75])
        np.testing.assert_array_equal(
            compute_eval_mask(trajectory.timestamps_ns, warmup_seconds=0.25),
            [False, False, True, True],
        )

    def test_local_headings_are_unwrapped_across_the_pi_branch_cut(self) -> None:
        trajectory = AV2Trajectory(
            timestamps_ns=np.array([0, 100_000_000, 200_000_000], dtype=np.int64),
            positions_m=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            velocities_mps=np.tile([1.0, 0.0], (3, 1)),
            observed=np.array([True, True, True]),
            headings_rad=np.array([3.0, -3.1, -2.9]),
        )

        local = to_initial_local_frame(trajectory)

        self.assertTrue(np.all(np.diff(local.headings_rad) > 0.0))
        np.testing.assert_allclose(local.headings_rad[0], 0.0, atol=1e-12)

    def test_causal_prefix_never_exposes_or_depends_on_future_samples(self) -> None:
        trajectory = self._trajectory()
        prefix = causal_prefix(trajectory, 1)
        self.assertEqual(prefix.num_steps, 2)
        np.testing.assert_array_equal(prefix.timestamps_ns, trajectory.timestamps_ns[:2])
        future_changed = AV2Trajectory(
            timestamps_ns=trajectory.timestamps_ns,
            positions_m=np.vstack((trajectory.positions_m[:2], [[999.0, -999.0], [888.0, -888.0]])),
            velocities_mps=np.vstack((trajectory.velocities_mps[:2], [[99.0, -99.0], [88.0, -88.0]])),
            observed=trajectory.observed,
            headings_rad=trajectory.headings_rad,
        )
        future_changed_prefix = causal_prefix(future_changed, 1)
        np.testing.assert_array_equal(prefix.positions_m, future_changed_prefix.positions_m)
        np.testing.assert_array_equal(prefix.velocities_mps, future_changed_prefix.velocities_mps)


if __name__ == "__main__":
    unittest.main()
