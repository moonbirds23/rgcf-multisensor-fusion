"""Portable, schema-checked I/O for small AV2 feature-shard smoke tests.

This module deliberately does *not* read AV2 data, create features, or depend
on PyTorch.  It is the persistence boundary for already-created, complete
sequence arrays.  Both the payload and its JSON sidecar are confined to the
allow-listed ``data/feature_shards/{pilot,formal}`` directories through
``data.av2.paths``.

The format is intentionally simple during the pre-data phase:

* ``<shard_id>.npz`` contains named numeric NumPy arrays;
* ``<shard_id>.metadata.json`` is a versioned, human-inspectable contract;
* every array has the same leading ``[sequence, time]`` dimensions.

The generic array mapping keeps this I/O layer independent from the future
sensor/EKF/feature builder while still making malformed shards fail early.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np

from .paths import Av2Paths, prepare_av2_output_dir


SHARD_SCHEMA_VERSION = "AV2_FEATURE_SHARD_V1"
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SHARD_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_STAGES = frozenset(("pilot", "formal"))


class Av2ShardError(ValueError):
    """Raised when a shard violates its versioned storage contract."""


@dataclass(frozen=True)
class ShardFieldMetadata:
    """Static shape and dtype details for one payload array."""

    name: str
    trailing_shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class ShardMetadata:
    """Serializable contract for a complete-sequence feature shard."""

    schema_version: str
    shard_id: str
    stage: str
    manifest_sha256: str
    config_sha256: str
    sequence_ids: tuple[str, ...]
    sequence_count: int
    sequence_length: int
    fields: tuple[ShardFieldMetadata, ...]
    created_utc: str

    def validate(self) -> None:
        if not all(
            isinstance(value, str)
            for value in (
                self.schema_version,
                self.shard_id,
                self.stage,
                self.manifest_sha256,
                self.config_sha256,
                self.created_utc,
            )
        ):
            raise Av2ShardError("core metadata values must be strings")
        if isinstance(self.sequence_count, bool) or isinstance(self.sequence_length, bool):
            raise Av2ShardError("sequence_count and sequence_length must be integers")
        if not isinstance(self.sequence_count, int) or not isinstance(self.sequence_length, int):
            raise Av2ShardError("sequence_count and sequence_length must be integers")
        if self.schema_version != SHARD_SCHEMA_VERSION:
            raise Av2ShardError(
                f"Unsupported shard schema {self.schema_version!r}; expected {SHARD_SCHEMA_VERSION!r}"
            )
        _validate_shard_id(self.shard_id)
        _validate_stage(self.stage)
        _validate_hash(self.manifest_sha256, "manifest_sha256")
        _validate_hash(self.config_sha256, "config_sha256")
        if self.sequence_count <= 0 or self.sequence_length <= 0:
            raise Av2ShardError("sequence_count and sequence_length must be positive")
        if len(self.sequence_ids) != self.sequence_count:
            raise Av2ShardError("sequence_ids count must equal sequence_count")
        if not all(isinstance(item, str) and item for item in self.sequence_ids):
            raise Av2ShardError("sequence_ids must contain non-empty strings")
        if len(set(self.sequence_ids)) != len(self.sequence_ids):
            raise Av2ShardError("sequence_ids must be unique within a shard")
        if not self.fields:
            raise Av2ShardError("a shard must contain at least one array field")
        field_names = [field.name for field in self.fields]
        if len(set(field_names)) != len(field_names):
            raise Av2ShardError("field metadata names must be unique")
        for field in self.fields:
            if not isinstance(field.name, str) or not field.name or field.name.startswith("_"):
                raise Av2ShardError(f"invalid field name: {field.name!r}")
            if not isinstance(field.dtype, str):
                raise Av2ShardError(f"{field.name}: dtype must be a string")
            if any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in field.trailing_shape):
                raise Av2ShardError(f"{field.name}: trailing dimensions must be positive")
            try:
                dtype = np.dtype(field.dtype)
            except TypeError as exc:
                raise Av2ShardError(f"{field.name}: invalid dtype {field.dtype!r}") from exc
            if dtype.hasobject or dtype.kind not in "biufc":
                raise Av2ShardError(f"{field.name}: object/string arrays are not supported")
        try:
            datetime.fromisoformat(self.created_utc.replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise Av2ShardError("created_utc must be an ISO-8601 timestamp") from exc

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe metadata with stable field ordering."""

        return {
            "schema_version": self.schema_version,
            "shard_id": self.shard_id,
            "stage": self.stage,
            "manifest_sha256": self.manifest_sha256,
            "config_sha256": self.config_sha256,
            "sequence_ids": list(self.sequence_ids),
            "sequence_count": self.sequence_count,
            "sequence_length": self.sequence_length,
            "fields": [
                {
                    "name": field.name,
                    "trailing_shape": list(field.trailing_shape),
                    "dtype": field.dtype,
                }
                for field in self.fields
            ],
            "created_utc": self.created_utc,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ShardMetadata":
        """Deserialize and validate a metadata document without coercion."""

        required = {
            "schema_version",
            "shard_id",
            "stage",
            "manifest_sha256",
            "config_sha256",
            "sequence_ids",
            "sequence_count",
            "sequence_length",
            "fields",
            "created_utc",
        }
        if set(value) != required:
            missing = sorted(required - set(value))
            unknown = sorted(set(value) - required)
            raise Av2ShardError(f"metadata keys mismatch; missing={missing}, unknown={unknown}")
        try:
            fields_value = value["fields"]
            if not isinstance(fields_value, list):
                raise TypeError("fields is not a list")
            fields: list[ShardFieldMetadata] = []
            for item in fields_value:
                if not isinstance(item, dict) or set(item) != {"name", "trailing_shape", "dtype"}:
                    raise TypeError("field entry is not a valid object")
                trailing_shape = item["trailing_shape"]
                if not isinstance(trailing_shape, list):
                    raise TypeError("field trailing_shape is not a list")
                fields.append(
                    ShardFieldMetadata(
                        name=item["name"],
                        trailing_shape=tuple(trailing_shape),
                        dtype=item["dtype"],
                    )
                )
            sequence_ids_value = value["sequence_ids"]
            if not isinstance(sequence_ids_value, list):
                raise TypeError("sequence_ids is not a list")
            metadata = cls(
                schema_version=value["schema_version"],
                shard_id=value["shard_id"],
                stage=value["stage"],
                manifest_sha256=value["manifest_sha256"],
                config_sha256=value["config_sha256"],
                sequence_ids=tuple(sequence_ids_value),
                sequence_count=value["sequence_count"],
                sequence_length=value["sequence_length"],
                fields=tuple(fields),
                created_utc=value["created_utc"],
            )
        except (KeyError, TypeError) as exc:
            raise Av2ShardError("metadata has invalid value types") from exc
        metadata.validate()
        return metadata


@dataclass(frozen=True)
class LoadedShard:
    """A validated shard payload with its immutable metadata contract."""

    metadata: ShardMetadata
    arrays: dict[str, np.ndarray]


def build_shard_metadata(
    *,
    shard_id: str,
    stage: str,
    manifest_sha256: str,
    config_sha256: str,
    sequence_ids: tuple[str, ...] | list[str],
    arrays: Mapping[str, np.ndarray],
    created_utc: str | None = None,
) -> ShardMetadata:
    """Build metadata from in-memory complete sequence arrays.

    This helper is useful for synthetic smoke-test shards.  Production feature
    building may use the same function only after all arrays are finalized.
    """

    normalized = _normalize_arrays(arrays)
    sequence_count, sequence_length = _shared_sequence_dimensions(normalized)
    metadata = ShardMetadata(
        schema_version=SHARD_SCHEMA_VERSION,
        shard_id=shard_id,
        stage=stage,
        manifest_sha256=manifest_sha256,
        config_sha256=config_sha256,
        sequence_ids=tuple(sequence_ids),
        sequence_count=sequence_count,
        sequence_length=sequence_length,
        fields=tuple(
            ShardFieldMetadata(name=name, trailing_shape=tuple(array.shape[2:]), dtype=array.dtype.str)
            for name, array in sorted(normalized.items())
        ),
        created_utc=created_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )
    metadata.validate()
    _validate_arrays_against_metadata(normalized, metadata)
    return metadata


def write_mock_shard(paths: Av2Paths, metadata: ShardMetadata, arrays: Mapping[str, np.ndarray]) -> tuple[Path, Path]:
    """Write one validated synthetic complete-sequence shard under AV2 storage.

    No arbitrary output path is accepted.  The target is derived solely from
    the validated ``Av2Paths`` and metadata's frozen stage and shard identifier.
    """

    metadata.validate()
    normalized = _normalize_arrays(arrays)
    _validate_arrays_against_metadata(normalized, metadata)
    shard_dir = _feature_shard_dir(paths, metadata.stage)
    payload_path, metadata_path = _shard_paths(shard_dir, metadata.shard_id)
    _atomic_write_npz(payload_path, normalized)
    try:
        _atomic_write_json(metadata_path, metadata.to_dict())
    except Exception:
        payload_path.unlink(missing_ok=True)
        raise
    return payload_path, metadata_path


def load_mock_shard(
    paths: Av2Paths,
    *,
    stage: str,
    shard_id: str,
    expected_manifest_sha256: str | None = None,
    expected_config_sha256: str | None = None,
) -> LoadedShard:
    """Load a shard only after checking metadata version, hashes, and shapes."""

    _validate_stage(stage)
    _validate_shard_id(shard_id)
    if expected_manifest_sha256 is not None:
        _validate_hash(expected_manifest_sha256, "expected_manifest_sha256")
    if expected_config_sha256 is not None:
        _validate_hash(expected_config_sha256, "expected_config_sha256")
    shard_dir = _feature_shard_dir(paths, stage)
    payload_path, metadata_path = _shard_paths(shard_dir, shard_id)
    if not payload_path.is_file() or not metadata_path.is_file():
        raise Av2ShardError(f"shard payload and metadata must both exist: {shard_id}")
    try:
        raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Av2ShardError(f"cannot read shard metadata: {metadata_path}") from exc
    if not isinstance(raw_metadata, dict):
        raise Av2ShardError("shard metadata must be a JSON object")
    metadata = ShardMetadata.from_dict(raw_metadata)
    if metadata.stage != stage or metadata.shard_id != shard_id:
        raise Av2ShardError("metadata stage or shard_id does not match the requested shard")
    if expected_manifest_sha256 is not None and metadata.manifest_sha256 != expected_manifest_sha256:
        raise Av2ShardError("shard manifest hash does not match the expected manifest")
    if expected_config_sha256 is not None and metadata.config_sha256 != expected_config_sha256:
        raise Av2ShardError("shard config hash does not match the expected config")
    try:
        with np.load(payload_path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
    except (OSError, ValueError) as exc:
        raise Av2ShardError(f"cannot load shard payload: {payload_path}") from exc
    normalized = _normalize_arrays(arrays)
    _validate_arrays_against_metadata(normalized, metadata)
    return LoadedShard(metadata=metadata, arrays=normalized)


def _feature_shard_dir(paths: Av2Paths, stage: str) -> Path:
    _validate_stage(stage)
    return prepare_av2_output_dir(paths, Path("data") / "feature_shards" / stage)


def _shard_paths(shard_dir: Path, shard_id: str) -> tuple[Path, Path]:
    _validate_shard_id(shard_id)
    payload_path = shard_dir / f"{shard_id}.npz"
    metadata_path = shard_dir / f"{shard_id}.metadata.json"
    if payload_path.parent != shard_dir or metadata_path.parent != shard_dir:
        raise Av2ShardError("shard identifier must not create nested output paths")
    return payload_path, metadata_path


def _normalize_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not arrays:
        raise Av2ShardError("a shard must contain at least one array")
    normalized: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        if not isinstance(name, str) or not name or name.startswith("_"):
            raise Av2ShardError(f"invalid array field name: {name!r}")
        if not isinstance(value, np.ndarray):
            raise Av2ShardError(f"{name}: shard arrays must be NumPy arrays")
        if value.ndim < 2:
            raise Av2ShardError(f"{name}: arrays must have [sequence, time, ...] dimensions")
        if value.dtype.hasobject or value.dtype.kind not in "biufc":
            raise Av2ShardError(f"{name}: object/string arrays are not supported")
        if not np.isfinite(value).all():
            raise Av2ShardError(f"{name}: arrays must contain only finite values")
        normalized[name] = np.ascontiguousarray(value)
    return normalized


def _shared_sequence_dimensions(arrays: Mapping[str, np.ndarray]) -> tuple[int, int]:
    dimensions = {(array.shape[0], array.shape[1]) for array in arrays.values()}
    if len(dimensions) != 1:
        raise Av2ShardError("all fields must share identical [sequence, time] dimensions")
    sequence_count, sequence_length = dimensions.pop()
    if sequence_count <= 0 or sequence_length <= 0:
        raise Av2ShardError("sequence and time dimensions must be positive")
    return sequence_count, sequence_length


def _validate_arrays_against_metadata(arrays: Mapping[str, np.ndarray], metadata: ShardMetadata) -> None:
    sequence_count, sequence_length = _shared_sequence_dimensions(arrays)
    if (sequence_count, sequence_length) != (metadata.sequence_count, metadata.sequence_length):
        raise Av2ShardError("payload sequence dimensions do not match metadata")
    expected_fields = {field.name: field for field in metadata.fields}
    if set(arrays) != set(expected_fields):
        raise Av2ShardError("payload array names do not match metadata fields")
    for name, array in arrays.items():
        field = expected_fields[name]
        if tuple(array.shape[2:]) != field.trailing_shape:
            raise Av2ShardError(f"{name}: payload trailing shape does not match metadata")
        if array.dtype.str != field.dtype:
            raise Av2ShardError(f"{name}: payload dtype does not match metadata")


def _atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temp_name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".json", dir=path.parent)
    temporary = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_stage(stage: str) -> None:
    if stage not in _STAGES:
        raise Av2ShardError(f"stage must be one of {sorted(_STAGES)}, got {stage!r}")


def _validate_shard_id(shard_id: str) -> None:
    if not isinstance(shard_id, str) or not _SHARD_ID_RE.fullmatch(shard_id):
        raise Av2ShardError("shard_id must be a safe 1-128 character filename stem")


def _validate_hash(value: str, label: str) -> None:
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise Av2ShardError(f"{label} must be a lowercase SHA-256 hex digest")
