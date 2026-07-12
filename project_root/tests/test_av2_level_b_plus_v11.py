from __future__ import annotations

import unittest

import numpy as np

from configs.av2_config import AV2_PILOT_V1
from configs.av2_level_b_plus_v11 import AGGRESSIVE_PROTOCOL
from data.av2.feature_builder import build_av2_feature_arrays
from simulation.av2_sensor_ekf import run_av2_sensor_ekfs


def _regular_truth() -> tuple[np.ndarray, np.ndarray]:
    """A short, deterministic trajectory spanning several rate boundaries."""
    timestamps = np.arange(11, dtype=np.int64) * 100_000_000
    seconds = timestamps.astype(np.float64) * 1e-9
    truth = np.column_stack(
        (
            15.0 + 4.0 * seconds,
            -8.0 + 1.2 * seconds,
            np.full(timestamps.size, 4.0),
            np.full(timestamps.size, 1.2),
        )
    )
    return timestamps, truth


class AV2LevelBPlusV11Tests(unittest.TestCase):
    def test_aggressive_protocol_has_only_the_authorised_sensor_values(self) -> None:
        specs = AGGRESSIVE_PROTOCOL.sensors
        self.assertEqual(tuple(spec.name for spec in specs), ("T1", "T2", "T3", "E1", "E2"))
        self.assertEqual(tuple(spec.local_position_xy_m for spec in specs), (
            None, (80.0, -50.0), (-60.0, 70.0), (90.0, 65.0), (-85.0, -50.0),
        ))
        self.assertEqual(tuple(spec.rate_hz for spec in specs), (5.0, 5.0, 5.0, 5.0, 10.0))
        self.assertEqual(tuple(spec.noise_standard_deviations for spec in specs), (
            (("x_m", 4.5), ("y_m", 4.5)),
            (("range_m", 2.0), ("bearing_deg", 0.40)),
            (("range_m", 2.4), ("bearing_deg", 0.45)),
            (("bearing_deg", 0.8),),
            (("range_m", 1.2),),
        ))
        # The baseline object remains frozen and is not mutated by the v1.1 exception.
        self.assertEqual(AV2_PILOT_V1.sensors[0].rate_hz, 10.0)
        self.assertEqual(AV2_PILOT_V1.sensors[4].local_position_xy_m, (-85.0, 50.0))

    def test_aggressive_sensor_simulation_uses_rates_and_noise(self) -> None:
        timestamps, truth = _regular_truth()
        outputs = run_av2_sensor_ekfs(
            timestamps, truth, rng=np.random.default_rng(73), protocol=AGGRESSIVE_PROTOCOL
        )

        # T1/T2/T3/E1 are 5 Hz on this 10 Hz timestamp sequence; E2 remains 10 Hz.
        five_hz = np.array([True, False, True, False, True, False, True, False, True, False, True])
        np.testing.assert_array_equal(outputs.posterior_measurement_valid[:, 0], five_hz)
        np.testing.assert_array_equal(outputs.posterior_measurement_valid[:, 1], five_hz)
        np.testing.assert_array_equal(outputs.posterior_measurement_valid[:, 2], five_hz)
        np.testing.assert_array_equal(outputs.evidence_valid[:, 0], five_hz)
        np.testing.assert_array_equal(outputs.evidence_valid[:, 1], np.ones(11, dtype=bool))

        expected_noise_diagonals = ((20.25, 20.25), (4.0, 0.16), (5.76, 0.2025), (0.64,), (1.44,))
        for sensor_index, expected in enumerate(expected_noise_diagonals):
            measurement = outputs.measurements[0][sensor_index]
            self.assertTrue(measurement.valid)
            np.testing.assert_allclose(np.diag(measurement.R_actual), expected, rtol=0.0, atol=1e-12)
            np.testing.assert_allclose(measurement.R_reported, measurement.R_actual, rtol=0.0, atol=0.0)

    def test_feature_builder_uses_passed_protocol_geometry_including_e2_y(self) -> None:
        timestamps, truth = _regular_truth()
        outputs = run_av2_sensor_ekfs(
            timestamps, truth, rng=np.random.default_rng(79), protocol=AGGRESSIVE_PROTOCOL
        )
        arrays = build_av2_feature_arrays(
            outputs, truth.astype(np.float32), protocol=AGGRESSIVE_PROTOCOL
        )
        active = np.flatnonzero(arrays.evidence_mask[:, 1] > 0.5)
        self.assertGreater(active.size, 0)
        np.testing.assert_allclose(arrays.evidence_feat[active, 1, 8], -0.085, rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(arrays.evidence_feat[active, 1, 9], -0.050, rtol=0.0, atol=1e-7)

        # Default behaviour is still the frozen Pilot v1 geometry when no protocol is passed.
        baseline = run_av2_sensor_ekfs(timestamps, truth, rng=np.random.default_rng(79))
        default_arrays = build_av2_feature_arrays(baseline, truth.astype(np.float32))
        default_active = np.flatnonzero(default_arrays.evidence_mask[:, 1] > 0.5)
        np.testing.assert_allclose(default_arrays.evidence_feat[default_active, 1, 9], 0.050, rtol=0.0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
