from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data.av2.paths import AV2_RAW_RELATIVE_DIR, resolve_av2_paths
from data.av2.shard_io import (
    SHARD_SCHEMA_VERSION,
    Av2ShardError,
    ShardMetadata,
    build_shard_metadata,
    load_mock_shard,
    write_mock_shard,
)


_MANIFEST_HASH = "1" * 64
_CONFIG_HASH = "2" * 64


def _paths(tmp_path: Path):
    root = tmp_path / "mounted_av2"
    (root / AV2_RAW_RELATIVE_DIR).mkdir(parents=True)
    return root, resolve_av2_paths(root, environ={})


def _arrays() -> dict[str, np.ndarray]:
    return {
        "post_feat": np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3),
        "target": np.arange(2 * 4 * 2, dtype=np.float64).reshape(2, 4, 2),
    }


def _metadata(arrays: dict[str, np.ndarray]):
    return build_shard_metadata(
        shard_id="smoke_000",
        stage="pilot",
        manifest_sha256=_MANIFEST_HASH,
        config_sha256=_CONFIG_HASH,
        sequence_ids=("scenario-a", "scenario-b"),
        arrays=arrays,
        created_utc="2026-07-10T00:00:00Z",
    )


class Av2ShardIoTests(unittest.TestCase):
    def test_round_trip_uses_allowlisted_feature_shard_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root, paths = _paths(Path(temp_dir))
            arrays = _arrays()
            metadata = _metadata(arrays)

            payload_path, metadata_path = write_mock_shard(paths, metadata, arrays)
            loaded = load_mock_shard(
                paths,
                stage="pilot",
                shard_id="smoke_000",
                expected_manifest_sha256=_MANIFEST_HASH,
                expected_config_sha256=_CONFIG_HASH,
            )

            expected_dir = root / "data" / "feature_shards" / "pilot"
            self.assertEqual(payload_path.parent, expected_dir)
            self.assertEqual(metadata_path.parent, expected_dir)
            self.assertEqual(loaded.metadata, metadata)
            np.testing.assert_array_equal(loaded.arrays["post_feat"], arrays["post_feat"])
            np.testing.assert_array_equal(loaded.arrays["target"], arrays["target"])
            self.assertFalse((root / "runs").exists())
            self.assertEqual(list((root / AV2_RAW_RELATIVE_DIR).iterdir()), [])

    def test_load_rejects_schema_or_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _, paths = _paths(Path(temp_dir))
            arrays = _arrays()
            write_mock_shard(paths, _metadata(arrays), arrays)

            with self.assertRaisesRegex(Av2ShardError, "manifest hash"):
                load_mock_shard(
                    paths,
                    stage="pilot",
                    shard_id="smoke_000",
                    expected_manifest_sha256="3" * 64,
                )

            metadata_path = paths.root / "data" / "feature_shards" / "pilot" / "smoke_000.metadata.json"
            document = json.loads(metadata_path.read_text(encoding="utf-8"))
            document["schema_version"] = "AV2_FEATURE_SHARD_V999"
            metadata_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(Av2ShardError, "Unsupported shard schema"):
                load_mock_shard(paths, stage="pilot", shard_id="smoke_000")

    def test_write_rejects_inconsistent_sequence_dimensions_before_creating_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root, paths = _paths(Path(temp_dir))
            arrays = _arrays()
            metadata = _metadata(arrays)
            malformed = {**arrays, "mask": np.ones((2, 3, 1), dtype=np.float32)}

            with self.assertRaisesRegex(Av2ShardError, "identical \[sequence, time\] dimensions"):
                write_mock_shard(paths, metadata, malformed)

            output_dir = root / "data" / "feature_shards" / "pilot"
            self.assertFalse(output_dir.exists())

    def test_metadata_is_strictly_serializable_and_validates_payload_dtypes(self) -> None:
        arrays = _arrays()
        metadata = _metadata(arrays)
        decoded = ShardMetadata.from_dict(metadata.to_dict())

        self.assertEqual(decoded.schema_version, SHARD_SCHEMA_VERSION)
        altered = {"post_feat": arrays["post_feat"].astype(np.float64), "target": arrays["target"]}
        with tempfile.TemporaryDirectory() as temp_dir:
            _, paths = _paths(Path(temp_dir))
            with self.assertRaisesRegex(Av2ShardError, "dtype"):
                write_mock_shard(paths, metadata, altered)

    def test_stage_and_shard_id_cannot_escape_allowlisted_paths(self) -> None:
        arrays = _arrays()
        with self.assertRaisesRegex(Av2ShardError, "stage must"):
            build_shard_metadata(
                shard_id="smoke_000",
                stage="../raw",
                manifest_sha256=_MANIFEST_HASH,
                config_sha256=_CONFIG_HASH,
                sequence_ids=("scenario-a", "scenario-b"),
                arrays=arrays,
                created_utc="2026-07-10T00:00:00Z",
            )
        with self.assertRaisesRegex(Av2ShardError, "safe"):
            build_shard_metadata(
                shard_id="../escape",
                stage="pilot",
                manifest_sha256=_MANIFEST_HASH,
                config_sha256=_CONFIG_HASH,
                sequence_ids=("scenario-a", "scenario-b"),
                arrays=arrays,
                created_utc="2026-07-10T00:00:00Z",
            )


if __name__ == "__main__":
    unittest.main()
