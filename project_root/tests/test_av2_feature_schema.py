from __future__ import annotations

import unittest

import numpy as np

from data.av2.feature_schema import (
    AV2FeatureSchemaError,
    AV2_FEATURE_SCHEMA_VERSION,
    validate_av2_feature_arrays,
    validate_av2_feature_shard,
)


def _valid_arrays(steps: int = 4):
    post = np.zeros((steps, 3, 9), dtype=np.float32)
    post_mask = np.ones((steps, 3), dtype=np.float32)
    post[..., 8] = post_mask
    meas = np.zeros((steps, 3, 18), dtype=np.float32)
    meas_mask = np.ones((steps, 3), dtype=np.float32)
    meas[..., 12] = meas_mask
    evidence = np.zeros((steps, 2, 16), dtype=np.float32)
    evidence_mask = np.ones((steps, 2), dtype=np.float32)
    evidence[..., 13] = evidence_mask
    pair = np.zeros((steps, 3, 5, 8), dtype=np.float32)
    for p in range(3):
        for m in range(3):
            pair[:, p, m, 0 if p == m else 1] = 1.0
        pair[:, p, 3:, 2] = 1.0
    pair[..., 6] = post_mask[:, :, None]
    pair[..., 7] = np.concatenate((meas_mask, evidence_mask), axis=1)[:, None, :]
    # Deliberately distinct evidence residual relation for every posterior.
    for t in range(steps):
        for p in range(3):
            for e in range(2):
                pair[t, p, 3 + e, 3:6] = (t + 1, p + e + 0.25, p / 2.0)
    return post, post_mask, meas, meas_mask, evidence, evidence_mask, pair


class AV2FeatureSchemaTests(unittest.TestCase):
    def test_valid_complete_array_set_passes_and_reports_counts(self) -> None:
        arrays = _valid_arrays()
        report = validate_av2_feature_arrays(
            post_feat=arrays[0], post_mask=arrays[1], meas_feat=arrays[2], meas_mask=arrays[3],
            evidence_feat=arrays[4], evidence_mask=arrays[5], mp_pair_feat=arrays[6],
            target=np.zeros((4, 4), dtype=np.float32),
            timestamps_ns=np.array([0, 100, 230, 350], dtype=np.int64),
            eval_mask=np.array([0, 0, 1, 1], dtype=np.uint8),
            pair_specificity="per_timestep",
        )
        self.assertEqual(report.schema_version, AV2_FEATURE_SCHEMA_VERSION)
        self.assertEqual(report.valid_posteriors, 12)
        self.assertEqual(report.valid_evidence_measurements, 8)

    def test_masked_node_must_not_retain_stale_feature_values(self) -> None:
        arrays = list(_valid_arrays())
        arrays[1][1, 2] = 0.0
        arrays[0][1, 2, 8] = 0.0
        arrays[0][1, 2, 0] = 3.0
        with self.assertRaisesRegex(AV2FeatureSchemaError, "stale features"):
            validate_av2_feature_arrays(
                post_feat=arrays[0], post_mask=arrays[1], meas_feat=arrays[2], meas_mask=arrays[3],
                evidence_feat=arrays[4], evidence_mask=arrays[5], mp_pair_feat=arrays[6],
            )

    def test_evidence_only_source_roles_are_frozen(self) -> None:
        arrays = _valid_arrays()
        with self.assertRaisesRegex(AV2FeatureSchemaError, "may not become posterior"):
            validate_av2_feature_arrays(
                post_feat=arrays[0], post_mask=arrays[1], meas_feat=arrays[2], meas_mask=arrays[3],
                evidence_feat=arrays[4], evidence_mask=arrays[5], mp_pair_feat=arrays[6],
                posterior_source_ids=("T1", "T2", "E1"),
            )

    def test_broadcast_evidence_pair_relation_is_rejected(self) -> None:
        arrays = list(_valid_arrays())
        arrays[6][:, :, 3, 3:6] = np.array([1.0, 0.5, 0.0], dtype=np.float32)
        with self.assertRaisesRegex(AV2FeatureSchemaError, "broadcast"):
            validate_av2_feature_arrays(
                post_feat=arrays[0], post_mask=arrays[1], meas_feat=arrays[2], meas_mask=arrays[3],
                evidence_feat=arrays[4], evidence_mask=arrays[5], mp_pair_feat=arrays[6],
            )

    def test_pair_mask_and_role_layout_cannot_drift(self) -> None:
        arrays = list(_valid_arrays())
        arrays[6][0, 1, 4, 6] = 0.0
        with self.assertRaisesRegex(AV2FeatureSchemaError, "required mask or role layout"):
            validate_av2_feature_arrays(
                post_feat=arrays[0], post_mask=arrays[1], meas_feat=arrays[2], meas_mask=arrays[3],
                evidence_feat=arrays[4], evidence_mask=arrays[5], mp_pair_feat=arrays[6],
            )

    def test_mapping_requires_versioned_complete_shard(self) -> None:
        arrays = _valid_arrays()
        shard = {
            "feature_schema_version": AV2_FEATURE_SCHEMA_VERSION,
            "post_feat": arrays[0], "post_mask": arrays[1], "meas_feat": arrays[2],
            "meas_mask": arrays[3], "evidence_feat": arrays[4], "evidence_mask": arrays[5],
            "mp_pair_feat": arrays[6],
        }
        self.assertEqual(validate_av2_feature_shard(shard).num_steps, 4)
        shard["feature_schema_version"] = "unknown"
        with self.assertRaisesRegex(AV2FeatureSchemaError, "feature_schema_version"):
            validate_av2_feature_shard(shard)


if __name__ == "__main__":
    unittest.main()
