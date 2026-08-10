from __future__ import annotations

import unittest

import numpy as np

from configs.av2_nominal_1000_v32 import NOMINAL_PROTOCOL
from simulation.av2_sensor_ekf import AV2SensorEkfOutputs, SensorMeasurement, run_av2_sensor_ekfs
from simulation.supplementary_baselines import (
    fuse_av2_ci_tracks,
    run_centralized_multisensor_ekf,
    run_ci_eu_sequence,
)


def _outputs(steps: int = 21) -> AV2SensorEkfOutputs:
    timestamps = np.arange(steps, dtype=np.int64) * 100_000_000
    seconds = timestamps.astype(np.float64) * 1e-9
    truth = np.column_stack(
        (10.0 + 5.0 * seconds, -4.0 + 2.0 * seconds, np.full(steps, 5.0), np.full(steps, 2.0))
    )
    return run_av2_sensor_ekfs(
        timestamps,
        truth,
        rng=np.random.default_rng(17),
        protocol=NOMINAL_PROTOCOL,
    )


def _without_evidence(outputs: AV2SensorEkfOutputs) -> AV2SensorEkfOutputs:
    records = []
    for row in outputs.measurements:
        changed = list(row)
        for sensor_index, name in ((3, "E1"), (4, "E2")):
            changed[sensor_index] = SensorMeasurement(
                name,
                "evidence",
                row[sensor_index].emitted,
                False,
                None,
                None,
                None,
                None,
                False,
            )
        records.append(tuple(changed))
    return AV2SensorEkfOutputs(
        outputs.timestamps_ns,
        outputs.posterior_mean,
        outputs.posterior_covariance_internal,
        outputs.posterior_covariance_reported,
        outputs.posterior_available,
        outputs.posterior_measurement_valid,
        np.zeros_like(outputs.evidence_valid),
        tuple(records),
    )


class SupplementaryBaselineTests(unittest.TestCase):
    def test_ci_eu_without_evidence_is_exactly_frozen_av2_ci(self) -> None:
        outputs = _without_evidence(_outputs())
        result = run_ci_eu_sequence(outputs, NOMINAL_PROTOCOL)
        for step in range(len(outputs.timestamps_ns)):
            x, covariance, weights = fuse_av2_ci_tracks(
                outputs.posterior_mean[step],
                outputs.posterior_covariance_reported[step],
                outputs.posterior_available[step],
            )
            np.testing.assert_allclose(result.xhat[step], x, atol=0.0, rtol=0.0)
            np.testing.assert_allclose(result.Phat[step], covariance, atol=0.0, rtol=0.0)
            np.testing.assert_allclose(result.ci_weights[step], weights, atol=0.0, rtol=0.0)
        self.assertTrue(np.all(result.evidence_dim == 0))
        self.assertTrue(np.all(np.isnan(result.nis)))

    def test_ci_eu_uses_dynamic_evidence_dimensions_and_spd_covariance(self) -> None:
        result = run_ci_eu_sequence(_outputs(), NOMINAL_PROTOCOL)
        np.testing.assert_array_equal(result.evidence_dim[:3], [2, 1, 2])
        self.assertTrue(np.allclose(result.ci_weights.sum(axis=1), 1.0))
        self.assertGreater(float(np.linalg.eigvalsh(result.Phat[result.valid_mask]).min()), 0.0)
        self.assertTrue(np.all(np.isfinite(result.nis[result.evidence_dim > 0])))

    def test_cm_ekf_t1_only_matches_independent_t1_track(self) -> None:
        outputs = _outputs()
        result = run_centralized_multisensor_ekf(
            outputs,
            NOMINAL_PROTOCOL,
            enabled_sensors=(True, False, False, False, False),
        )
        np.testing.assert_allclose(result.xhat, outputs.posterior_mean[:, 0], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(
            result.Phat,
            outputs.posterior_covariance_internal[:, 0],
            atol=1e-12,
            rtol=1e-12,
        )
        np.testing.assert_array_equal(result.active_measurement_dims[:3], [0, 0, 2])

    def test_cm_ekf_joint_update_dimensions_are_dynamic_and_deterministic(self) -> None:
        outputs = _outputs()
        first = run_centralized_multisensor_ekf(outputs, NOMINAL_PROTOCOL)
        second = run_centralized_multisensor_ekf(outputs, NOMINAL_PROTOCOL)
        np.testing.assert_array_equal(first.active_measurement_dims[:3], [6, 1, 8])
        np.testing.assert_array_equal(first.xhat, second.xhat)
        np.testing.assert_array_equal(first.Phat, second.Phat)
        self.assertGreater(float(np.linalg.eigvalsh(first.Phat[first.valid_mask]).min()), 0.0)
        self.assertTrue(np.all(np.isfinite(first.xhat[first.valid_mask])))


if __name__ == "__main__":
    unittest.main()
