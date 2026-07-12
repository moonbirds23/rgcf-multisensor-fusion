from __future__ import annotations

import unittest

import numpy as np

from simulation.av2_sensor_ekf import PrincipalFault, run_av2_sensor_ekfs, sample_principal_fault


def synthetic_truth(count: int = 11) -> tuple[np.ndarray, np.ndarray]:
    # Deliberately irregular timestamps: the implementation must not use a
    # fixed 10 Hz frame interval for its CV transition.
    delta_ns = np.array([0, 80, 210, 330, 470, 600, 740, 900, 1040, 1210, 1400]) * 1_000_000
    timestamps = delta_ns[:count].astype(np.int64)
    seconds = timestamps.astype(np.float64) * 1e-9
    truth = np.column_stack((5.0 + 4.0 * seconds, -3.0 + 1.5 * seconds, np.full(count, 4.0), np.full(count, 1.5)))
    return timestamps, truth


class AV2SensorEkfTests(unittest.TestCase):
    def _run(self, fault: PrincipalFault = PrincipalFault()):
        timestamps, truth = synthetic_truth()
        return run_av2_sensor_ekfs(timestamps, truth, rng=np.random.default_rng(7), fault=fault)

    def test_nominal_tracks_are_independent_and_evidence_has_no_posterior(self) -> None:
        output = self._run()
        self.assertTrue(np.all(output.posterior_available[0]))
        self.assertEqual(output.posterior_mean.shape, (11, 3, 4))
        self.assertEqual(output.evidence_valid.shape, (11, 2))
        self.assertEqual(tuple(record.role for record in output.measurements[0]), ("posterior", "posterior", "posterior", "evidence", "evidence"))
        self.assertTrue(np.all(np.isfinite(output.posterior_covariance_internal[output.posterior_available])))
        self.assertFalse(np.allclose(output.posterior_mean[-1, 0], output.posterior_mean[-1, 1]))

    def test_dropout_predicts_track_without_measurement_and_masks_evidence(self) -> None:
        track = self._run(PrincipalFault("dropout", "T2", onset_seconds=0.2, duration_seconds=0.3))
        self.assertTrue(track.posterior_available[3, 1])
        self.assertFalse(track.posterior_measurement_valid[3, 1])
        self.assertTrue(np.all(np.isfinite(track.posterior_mean[3, 1])))
        evidence = self._run(PrincipalFault("dropout", "E1", onset_seconds=0.2, duration_seconds=0.3))
        self.assertFalse(evidence.evidence_valid[3, 0])
        self.assertIsNone(evidence.measurements[3][3].z)

    def test_delayed_measurements_are_early_invalid_and_never_use_future_timestamp(self) -> None:
        output = self._run(PrincipalFault("time_delay", "T1", delay_steps=2))
        self.assertFalse(output.posterior_measurement_valid[0, 0])
        self.assertFalse(output.posterior_measurement_valid[1, 0])
        for k, row in enumerate(output.measurements):
            measurement = row[0]
            if measurement.valid:
                self.assertLessEqual(measurement.source_timestamp_ns, int(output.timestamps_ns[k]))
        self.assertEqual(output.measurements[2][0].source_timestamp_ns, int(output.timestamps_ns[0]))

    def test_reported_covariance_mismatch_does_not_change_internal_filter(self) -> None:
        nominal = self._run()
        mismatch = self._run(PrincipalFault("covariance_mismatch", "T2", covariance_alpha=4.0))
        np.testing.assert_allclose(mismatch.posterior_mean, nominal.posterior_mean, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(mismatch.posterior_covariance_internal, nominal.posterior_covariance_internal, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(mismatch.posterior_covariance_reported[:, 1], 4.0 * mismatch.posterior_covariance_internal[:, 1], equal_nan=True)
        np.testing.assert_allclose(mismatch.posterior_covariance_reported[:, 0], mismatch.posterior_covariance_internal[:, 0], equal_nan=True)

    def test_bias_ramp_is_in_native_measurement_space(self) -> None:
        output = self._run(PrincipalFault("bias_ramp", "E1", onset_seconds=0.2, duration_seconds=0.2, terminal_bias=np.array([0.1])))
        baseline = self._run()
        self.assertFalse(np.allclose(output.measurements[2][3].z, baseline.measurements[2][3].z))
        # It reaches the requested terminal bearing bias after onset + duration.
        delta = output.measurements[5][3].z - baseline.measurements[5][3].z
        np.testing.assert_allclose(delta, [0.1], atol=1e-12)

    def test_noise_beta_scales_actual_and_reported_measurement_covariance_squared(self) -> None:
        timestamps, truth = synthetic_truth()
        unit = run_av2_sensor_ekfs(timestamps, truth, rng=np.random.default_rng(29))
        scaled = run_av2_sensor_ekfs(
            timestamps, truth, rng=np.random.default_rng(29), measurement_noise_beta=1.5
        )
        for sensor_index in range(5):
            unit_measurement = unit.measurements[0][sensor_index]
            scaled_measurement = scaled.measurements[0][sensor_index]
            np.testing.assert_allclose(scaled_measurement.R_actual, 2.25 * unit_measurement.R_actual)
            np.testing.assert_allclose(scaled_measurement.R_reported, 2.25 * unit_measurement.R_reported)

    def test_materialized_faults_respect_frozen_protocol(self) -> None:
        rng = np.random.default_rng(13)
        for _ in range(100):
            fault = sample_principal_fault(rng)
            self.assertIn(fault.kind, {"nominal", "bias_ramp", "covariance_mismatch", "dropout", "time_delay"})

    def test_prefix_outputs_cannot_depend_on_future_truth(self) -> None:
        timestamps, truth = synthetic_truth()
        changed_future = truth.copy()
        changed_future[5:, :2] += 50_000.0
        original = run_av2_sensor_ekfs(timestamps, truth, rng=np.random.default_rng(23))
        changed = run_av2_sensor_ekfs(timestamps, changed_future, rng=np.random.default_rng(23))
        np.testing.assert_allclose(original.posterior_mean[:5], changed.posterior_mean[:5], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(
            original.posterior_covariance_internal[:5],
            changed.posterior_covariance_internal[:5],
            atol=0.0,
            rtol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
