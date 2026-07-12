from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from configs.av2_config import AV2_PILOT_V1, PROTOCOL_VERSION
from data.av2.manifest import (
    Av2ManifestError,
    Av2PilotManifest,
    build_pilot_manifest,
    manifest_overlap_count,
    write_manifest,
)


def _record(index: int, *, split: str = "train", eligible: bool = True) -> dict[str, object]:
    return {
        "scenario_id": f"scenario-{index:04d}",
        "official_split": split,
        "focal_track_id": f"focal-{index:04d}",
        "is_focal_track": True,
        "object_type": "VEHICLE",
        "valid_state_count": 105 if eligible else 104,
        "max_internal_missing_gap_steps": 2,
        "displacement_m": 15.0,
        "mean_speed_mps": 2.0,
        "local_coordinates_from_time_zero_only": True,
        "causal_processing": True,
        "future_state_as_input": False,
        "motion_type": ("straight", "turning", "accelerating", "high_dynamic")[index % 4],
        "city": ("PIT", "MIA", "ATX")[index % 3],
    }


class Av2ManifestTests(unittest.TestCase):
    def test_smoke_is_deterministic_stratified_and_excludes_official_validation(self) -> None:
        inventory = [_record(index) for index in range(120)]
        inventory.extend(_record(1000 + index, split="validation") for index in range(40))
        first = build_pilot_manifest(inventory, stage="smoke")
        second = build_pilot_manifest(reversed(inventory), stage="smoke")

        self.assertEqual(first, second)
        self.assertEqual(first.assignment_counts, {"train": 50, "validation": 20, "development_holdout": 20})
        self.assertEqual(first.protocol_version, PROTOCOL_VERSION)
        self.assertEqual(first.protocol_document_sha256, first.to_dict()["protocol_document_sha256"])
        self.assertEqual(first.manifest_spec["source_split"], "train")
        self.assertTrue(first.manifest_spec["split_by_scenario_before_variants"])
        self.assertEqual({entry.official_split for entry in first.entries}, {"train"})
        self.assertFalse({entry.scenario_id for entry in first.entries} & {f"scenario-{1000 + x:04d}" for x in range(40)})
        self.assertEqual(len({entry.scenario_id for entry in first.entries}), 90)
        self.assertGreater(len({(entry.motion_type, entry.city) for entry in first.entries}), 1)

    def test_decision_uses_frozen_counts_and_rejects_ineligible_records(self) -> None:
        inventory = [_record(index) for index in range(900)] + [_record(2000, eligible=False)]
        manifest = build_pilot_manifest(inventory, stage="decision")
        self.assertEqual(manifest.assignment_counts, {"train": 500, "validation": 100, "development_holdout": 200})
        self.assertNotIn("scenario-2000", {entry.scenario_id for entry in manifest.entries})
        self.assertEqual(manifest.eligibility_spec["minimum_valid_states"], AV2_PILOT_V1.trajectory.minimum_valid_states)

    def test_duplicate_and_overlap_detection_are_strict(self) -> None:
        duplicate = [_record(index) for index in range(90)]
        duplicate.append(_record(0))
        with self.assertRaisesRegex(Av2ManifestError, "duplicate scenario_id"):
            build_pilot_manifest(duplicate, stage="smoke")

        smoke = build_pilot_manifest([_record(index) for index in range(100)], stage="smoke")
        with self.assertRaisesRegex(Av2ManifestError, "overlapping Pilot splits"):
            Av2PilotManifest(
                stage="smoke", source_split="train", protocol_version=smoke.protocol_version,
                protocol_document_sha256=smoke.protocol_document_sha256, manifest_spec=smoke.manifest_spec,
                eligibility_spec=smoke.eligibility_spec,
                entries=(smoke.entries[0], smoke.entries[0].__class__(
                    scenario_id=smoke.entries[0].scenario_id, assignment="validation",
                    motion_type="straight", city="PIT", focal_track_id="other",
                )),
            )
        self.assertEqual(manifest_overlap_count((smoke, smoke)), 90)

    def test_json_csv_round_trip_and_tamper_evidence(self) -> None:
        manifest = build_pilot_manifest([_record(index) for index in range(100)], stage="smoke")
        self.assertEqual(Av2PilotManifest.from_json(manifest.to_json()), manifest)
        tampered = manifest.to_json().replace("scenario-0000", "scenario-tampered", 1)
        with self.assertRaisesRegex(Av2ManifestError, "sha256"):
            Av2PilotManifest.from_json(tampered)
        with tempfile.TemporaryDirectory() as directory:
            artifacts = write_manifest(manifest, Path(directory), stem="smoke_manifest")
            self.assertEqual(artifacts.json_path.read_text(encoding="utf-8"), manifest.to_json())
            with artifacts.csv_path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 90)
            self.assertEqual({row["manifest_sha256"] for row in rows}, {manifest.manifest_sha256})
            self.assertEqual({row["source_split"] for row in rows}, {"train"})


if __name__ == "__main__":
    unittest.main()
