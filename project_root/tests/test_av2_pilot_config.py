from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import unittest

from configs.av2_config import AV2_PILOT_V1, PROTOCOL_DOCUMENT_RELATIVE_PATH, PROTOCOL_DOCUMENT_SHA256, PROTOCOL_VERSION


class Av2PilotConfigTests(unittest.TestCase):
    def test_protocol_document_fingerprint_is_frozen(self) -> None:
        document = Path(__file__).parents[1] / PROTOCOL_DOCUMENT_RELATIVE_PATH
        self.assertEqual(hashlib.sha256(document.read_bytes()).hexdigest(), PROTOCOL_DOCUMENT_SHA256)

    def test_sensors_ekf_and_causal_trajectory_match_protocol(self) -> None:
        protocol = AV2_PILOT_V1
        self.assertEqual(protocol.version, PROTOCOL_VERSION)
        self.assertEqual(tuple(sensor.name for sensor in protocol.sensors), ("T1", "T2", "T3", "E1", "E2"))
        self.assertEqual(tuple(sensor.local_position_xy_m for sensor in protocol.sensors), (None, (80.0, -50.0), (-60.0, 70.0), (90.0, 65.0), (-85.0, 50.0)))
        self.assertEqual(tuple(sensor.rate_hz for sensor in protocol.sensors), (10.0, 5.0, 5.0, 5.0, 10.0))
        self.assertEqual(tuple(sensor.noise_standard_deviations for sensor in protocol.sensors), ((('x_m', 3.0), ('y_m', 3.0)), (('range_m', 2.5), ('bearing_deg', 0.6)), (('range_m', 3.0), ('bearing_deg', 0.8)), (('bearing_deg', 1.0),), (('range_m', 1.5),)))
        self.assertEqual(protocol.ekf.acceleration_sigma_mps2, 2.0)
        self.assertEqual(protocol.ekf.initial_velocity_variance_m2ps2, 36.0)
        self.assertEqual(protocol.trajectory.warmup_seconds, 1.0)
        self.assertTrue(protocol.trajectory.causal_processing)
        self.assertFalse(protocol.trajectory.future_state_as_input)

    def test_faults_manifests_seeds_and_gates_match_protocol(self) -> None:
        protocol = AV2_PILOT_V1
        self.assertEqual(protocol.faults.condition_probabilities, (("nominal", 0.60), ("bias_ramp", 0.15), ("covariance_mismatch", 0.10), ("dropout", 0.10), ("time_delay", 0.05)))
        self.assertEqual(protocol.faults.delay_steps, (1, 2, 3))
        self.assertEqual(protocol.faults.covariance_mismatch_alphas, (0.25, 0.50, 2.0, 4.0))
        self.assertEqual((protocol.smoke_manifest.train_scenarios, protocol.smoke_manifest.validation_scenarios, protocol.smoke_manifest.development_holdout_scenarios), (50, 20, 20))
        self.assertEqual((protocol.decision_manifest.train_scenarios, protocol.decision_manifest.validation_scenarios, protocol.decision_manifest.development_holdout_scenarios), (500, 100, 200))
        self.assertEqual(protocol.smoke_seeds.model_seeds, (0,))
        self.assertEqual(protocol.smoke_seeds.measurement_seeds, ())
        self.assertEqual(protocol.decision_seeds.model_seeds, (0, 1, 2))
        self.assertEqual(protocol.decision_seeds.measurement_seeds, (100, 101, 102))
        self.assertEqual(protocol.engineering_gates.spd_covariance_rate_minimum, 0.999)
        self.assertEqual(protocol.decision_gates.mixed_rmse_improvement_vs_ci_minimum, 0.03)
        self.assertEqual(protocol.decision_gates.nominal_rmse_degradation_vs_ci_maximum, 0.02)
        self.assertEqual(protocol.decision_gates.nominal_p95_degradation_vs_ci_maximum, 0.03)
        self.assertEqual(protocol.decision_gates.mixed_p95_improvement_vs_ci_minimum, 0.05)
        self.assertEqual(protocol.decision_gates.external_evidence_rmse_improvement_minimum, 0.01)
        self.assertEqual(protocol.decision_gates.external_evidence_p95_improvement_minimum, 0.02)
        self.assertEqual(protocol.decision_gates.nll_degradation_vs_ci_maximum, 0.05)
        self.assertEqual(protocol.budget.pilot_gpu_hours_maximum, 8.0)

    def test_dataclasses_are_immutable_and_validation_rejects_protocol_drift(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            AV2_PILOT_V1.trajectory.warmup_seconds = 2.0  # type: ignore[misc]
        altered = replace(AV2_PILOT_V1, automatic_threshold_adjustment=True)
        with self.assertRaisesRegex(ValueError, "frozen"):
            altered.validate()


if __name__ == "__main__":
    unittest.main()
