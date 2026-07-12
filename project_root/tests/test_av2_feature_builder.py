from __future__ import annotations

import unittest

import numpy as np

from data.av2.feature_builder import build_av2_feature_arrays
from data.av2.feature_schema import validate_av2_feature_arrays
from simulation.av2_sensor_ekf import PrincipalFault, run_av2_sensor_ekfs


def _timestamps_and_truth(count: int = 14) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.array(
        [0, 90, 215, 340, 490, 630, 780, 930, 1080, 1240, 1410, 1580, 1760, 1950],
        dtype=np.int64,
    )[:count] * 1_000_000
    seconds = timestamps.astype(np.float64) * 1e-9
    truth = np.column_stack(
        (
            15.0 + 4.0 * seconds,
            -8.0 + 1.2 * seconds,
            np.full(count, 4.0),
            np.full(count, 1.2),
        )
    )
    return timestamps, truth


class AV2FeatureBuilderTests(unittest.TestCase):
    def _outputs_and_target(self, fault: PrincipalFault = PrincipalFault()):
        timestamps, target = _timestamps_and_truth()
        outputs = run_av2_sensor_ekfs(
            timestamps, target, rng=np.random.default_rng(31), fault=fault
        )
        return outputs, target.astype(np.float32)

    def test_builds_schema_valid_canonical_arrays_and_warmup_masks(self) -> None:
        outputs, target = self._outputs_and_target()
        arrays = build_av2_feature_arrays(outputs, target, warmup_seconds=1.0)

        self.assertEqual(arrays.post_feat.shape, (14, 3, 9))
        self.assertEqual(arrays.meas_feat.shape, (14, 3, 18))
        self.assertEqual(arrays.evidence_feat.shape, (14, 2, 16))
        self.assertEqual(arrays.mp_pair_feat.shape, (14, 3, 5, 8))
        self.assertTrue(np.array_equal(arrays.loss_mask, arrays.eval_mask))
        self.assertTrue(np.all(arrays.eval_mask[:8] == 0.0))
        self.assertTrue(np.all(arrays.eval_mask[8:] == 1.0))
        self.assertEqual(arrays.validation_report.num_steps, 14)
        validate_av2_feature_arrays(
            post_feat=arrays.post_feat,
            post_mask=arrays.post_mask,
            meas_feat=arrays.meas_feat,
            meas_mask=arrays.meas_mask,
            evidence_feat=arrays.evidence_feat,
            evidence_mask=arrays.evidence_mask,
            mp_pair_feat=arrays.mp_pair_feat,
            target=arrays.target,
            timestamps_ns=arrays.timestamps_ns,
            eval_mask=arrays.eval_mask,
            pair_specificity="per_timestep",
        )
        self.assertTrue(np.array_equal(arrays.post_feat[..., 8], arrays.post_mask))
        self.assertTrue(np.array_equal(arrays.meas_feat[..., 12], arrays.meas_mask))
        self.assertTrue(np.array_equal(arrays.evidence_feat[..., 13], arrays.evidence_mask))
        self.assertTrue(np.all(arrays.mp_pair_feat[:, :, :3, 3:6] == 0.0))

    def test_post_features_use_reported_not_internal_covariance_and_preserve_both(self) -> None:
        outputs, target = self._outputs_and_target(
            PrincipalFault("covariance_mismatch", "T2", covariance_alpha=4.0)
        )
        arrays = build_av2_feature_arrays(outputs, target)

        valid = arrays.post_mask[:, 1] > 0.5
        expected = np.log1p(np.diagonal(outputs.posterior_covariance_reported[:, 1], axis1=1, axis2=2))
        internal = np.log1p(np.diagonal(outputs.posterior_covariance_internal[:, 1], axis1=1, axis2=2))
        np.testing.assert_allclose(arrays.post_feat[valid, 1, 4:8], expected[valid], rtol=0.0, atol=1e-6)
        self.assertGreater(np.max(np.abs(arrays.post_feat[valid, 1, 4:8] - internal[valid])), 0.1)
        np.testing.assert_allclose(
            arrays.diagnostics.posterior_covariance_internal,
            outputs.posterior_covariance_internal,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            arrays.diagnostics.posterior_covariance_reported,
            outputs.posterior_covariance_reported,
            equal_nan=True,
        )

    def test_evidence_pair_relations_are_posterior_specific_and_dropout_is_zeroed(self) -> None:
        outputs, target = self._outputs_and_target(
            PrincipalFault("dropout", "E1", onset_seconds=0.35, duration_seconds=0.30)
        )
        arrays = build_av2_feature_arrays(outputs, target)
        active = (arrays.evidence_mask[:, 0] > 0.5) & (np.sum(arrays.post_mask, axis=1) >= 2)
        self.assertTrue(np.any(active))
        for step in np.flatnonzero(active):
            valid_posts = arrays.post_mask[step] > 0.5
            relation = arrays.mp_pair_feat[step, valid_posts, 3, 3:6]
            self.assertFalse(np.all(relation == relation[0]))
        dropped = arrays.evidence_mask[:, 0] == 0.0
        self.assertTrue(np.any(dropped))
        self.assertTrue(np.all(arrays.evidence_feat[dropped, 0] == 0.0))
        self.assertTrue(np.all(arrays.mp_pair_feat[dropped, :, 3, 3:6] == 0.0))

    def test_target_and_future_truth_do_not_change_causal_features(self) -> None:
        timestamps, truth = _timestamps_and_truth()
        changed_future = truth.copy()
        changed_future[7:, :2] += 100_000.0
        original_outputs = run_av2_sensor_ekfs(timestamps, truth, rng=np.random.default_rng(42))
        changed_outputs = run_av2_sensor_ekfs(timestamps, changed_future, rng=np.random.default_rng(42))
        original = build_av2_feature_arrays(original_outputs, truth.astype(np.float32))
        changed = build_av2_feature_arrays(changed_outputs, changed_future.astype(np.float32))
        for field in ("post_feat", "meas_feat", "evidence_feat", "mp_pair_feat"):
            np.testing.assert_allclose(getattr(original, field)[:7], getattr(changed, field)[:7], rtol=0.0, atol=0.0)

        alternative_target = truth.astype(np.float32).copy()
        alternative_target[:, :2] += 99_999.0
        relabeled = build_av2_feature_arrays(original_outputs, alternative_target)
        for field in ("post_feat", "meas_feat", "evidence_feat", "mp_pair_feat", "post_mask", "meas_mask", "evidence_mask"):
            np.testing.assert_array_equal(getattr(original, field), getattr(relabeled, field))
        np.testing.assert_array_equal(relabeled.target, alternative_target)


if __name__ == "__main__":
    unittest.main()
