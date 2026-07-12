"""Frozen, data-independent AV2 Pilot manifest construction.

This module deliberately accepts *inventory records*, rather than paths or AV2
objects.  Producing those records is a later, data-mounted step.  Keeping the
selection layer here means that the Pilot split can be tested and frozen before
the AV2 download is available.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from configs.av2_config import (
    AV2_PILOT_V1,
    PROTOCOL_DOCUMENT_SHA256,
    PROTOCOL_VERSION,
    ManifestSpec,
)


MANIFEST_SCHEMA_VERSION = "AV2_PILOT_MANIFEST_V1"
_ASSIGNMENT_ORDER = ("train", "validation", "development_holdout")
_REQUIRED_INVENTORY_FIELDS = (
    "scenario_id",
    "official_split",
    "focal_track_id",
    "is_focal_track",
    "object_type",
    "valid_state_count",
    "max_internal_missing_gap_steps",
    "displacement_m",
    "mean_speed_mps",
    "local_coordinates_from_time_zero_only",
    "causal_processing",
    "future_state_as_input",
    "motion_type",
    "city",
)


class Av2ManifestError(ValueError):
    """Raised when inventory or a manifest violates the frozen Pilot protocol."""


def _nonempty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Av2ManifestError(f"{name} must be a non-empty string")
    return value.strip()


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Av2ManifestError(f"{name} must be a finite number")
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise Av2ManifestError(f"{name} must be a finite number")
    return result


@dataclass(frozen=True)
class Av2InventoryRecord:
    """One focal-track candidate produced by a future AV2 inventory adapter.

    The boolean provenance flags are intentional: a manifest builder must not
    infer causal/local-frame compliance from the shape of a trajectory.
    """

    scenario_id: str
    official_split: str
    focal_track_id: str
    is_focal_track: bool
    object_type: str
    valid_state_count: int
    max_internal_missing_gap_steps: int
    displacement_m: float
    mean_speed_mps: float
    local_coordinates_from_time_zero_only: bool
    causal_processing: bool
    future_state_as_input: bool
    motion_type: str
    city: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "Av2InventoryRecord":
        missing = [field for field in _REQUIRED_INVENTORY_FIELDS if field not in raw]
        if missing:
            raise Av2ManifestError("inventory record is missing required fields: " + ", ".join(missing))
        for field in (
            "is_focal_track",
            "local_coordinates_from_time_zero_only",
            "causal_processing",
            "future_state_as_input",
        ):
            if not isinstance(raw[field], bool):
                raise Av2ManifestError(f"{field} must be a bool")
        valid_count = raw["valid_state_count"]
        gap = raw["max_internal_missing_gap_steps"]
        if isinstance(valid_count, bool) or not isinstance(valid_count, int) or valid_count < 0:
            raise Av2ManifestError("valid_state_count must be a non-negative integer")
        if isinstance(gap, bool) or not isinstance(gap, int) or gap < 0:
            raise Av2ManifestError("max_internal_missing_gap_steps must be a non-negative integer")
        return cls(
            scenario_id=_nonempty_string("scenario_id", raw["scenario_id"]),
            official_split=_nonempty_string("official_split", raw["official_split"]),
            focal_track_id=_nonempty_string("focal_track_id", raw["focal_track_id"]),
            is_focal_track=raw["is_focal_track"],
            object_type=_nonempty_string("object_type", raw["object_type"]),
            valid_state_count=valid_count,
            max_internal_missing_gap_steps=gap,
            displacement_m=_finite_number("displacement_m", raw["displacement_m"]),
            mean_speed_mps=_finite_number("mean_speed_mps", raw["mean_speed_mps"]),
            local_coordinates_from_time_zero_only=raw["local_coordinates_from_time_zero_only"],
            causal_processing=raw["causal_processing"],
            future_state_as_input=raw["future_state_as_input"],
            motion_type=_nonempty_string("motion_type", raw["motion_type"]),
            city=_nonempty_string("city", raw["city"]),
        )

    def eligible_for_protocol(self) -> bool:
        """Whether this record satisfies every frozen trajectory requirement."""
        spec = AV2_PILOT_V1.trajectory
        return (
            self.is_focal_track == spec.focal_vehicle_only
            and self.object_type == spec.required_object_type
            and self.valid_state_count >= spec.minimum_valid_states
            and self.max_internal_missing_gap_steps <= spec.maximum_internal_missing_gap_steps
            and self.displacement_m >= spec.minimum_displacement_m
            and self.mean_speed_mps >= spec.minimum_mean_speed_mps
            and self.local_coordinates_from_time_zero_only == spec.local_coordinates_from_time_zero_only
            and self.causal_processing == spec.causal_processing
            and self.future_state_as_input == spec.future_state_as_input
        )

    @property
    def stratum(self) -> tuple[str, str]:
        return (self.motion_type, self.city)


def _record_from_input(record: Av2InventoryRecord | Mapping[str, Any]) -> Av2InventoryRecord:
    if isinstance(record, Av2InventoryRecord):
        return record
    if isinstance(record, Mapping):
        return Av2InventoryRecord.from_mapping(record)
    raise TypeError("inventory records must be Av2InventoryRecord instances or mappings")


@dataclass(frozen=True)
class ManifestEntry:
    scenario_id: str
    assignment: str
    motion_type: str
    city: str
    focal_track_id: str
    official_split: str = "train"

    def __post_init__(self) -> None:
        if self.assignment not in _ASSIGNMENT_ORDER:
            raise Av2ManifestError(f"unknown Pilot assignment {self.assignment!r}")
        if self.official_split != "train":
            raise Av2ManifestError("Pilot manifest entries must come from official train only")


@dataclass(frozen=True)
class Av2PilotManifest:
    """Fully self-describing, hashable Pilot selection."""

    stage: str
    source_split: str
    protocol_version: str
    protocol_document_sha256: str
    manifest_spec: Mapping[str, Any]
    eligibility_spec: Mapping[str, Any]
    entries: tuple[ManifestEntry, ...]
    manifest_sha256: str = ""

    def __post_init__(self) -> None:
        if self.stage not in {"smoke", "decision"}:
            raise Av2ManifestError("stage must be 'smoke' or 'decision'")
        if self.source_split != "train":
            raise Av2ManifestError("AV2 Pilot supports official train only")
        if self.protocol_version != PROTOCOL_VERSION or self.protocol_document_sha256 != PROTOCOL_DOCUMENT_SHA256:
            raise Av2ManifestError("manifest does not identify the frozen AV2 Pilot v1.0 protocol")
        validate_split_isolation(self.entries)
        expected = _sha256(self._payload_dict())
        if self.manifest_sha256 and self.manifest_sha256 != expected:
            raise Av2ManifestError("manifest_sha256 does not match canonical manifest content")
        object.__setattr__(self, "manifest_sha256", expected)

    @property
    def assignment_counts(self) -> dict[str, int]:
        return {assignment: sum(entry.assignment == assignment for entry in self.entries) for assignment in _ASSIGNMENT_ORDER}

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "stage": self.stage,
            "source_split": self.source_split,
            "protocol_version": self.protocol_version,
            "protocol_document_sha256": self.protocol_document_sha256,
            "manifest_spec": dict(self.manifest_spec),
            "eligibility_spec": dict(self.eligibility_spec),
            "entries": [asdict(entry) for entry in self.entries],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_dict(), "manifest_sha256": self.manifest_sha256}

    def to_json(self) -> str:
        return _canonical_json(self.to_dict()) + "\n"

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "Av2PilotManifest":
        required = {
            "schema_version", "stage", "source_split", "protocol_version", "protocol_document_sha256",
            "manifest_spec", "eligibility_spec", "entries", "manifest_sha256",
        }
        missing = required.difference(document)
        extra = set(document).difference(required)
        if missing or extra:
            raise Av2ManifestError(f"manifest keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
        if document["schema_version"] != MANIFEST_SCHEMA_VERSION:
            raise Av2ManifestError("unsupported manifest schema version")
        entries_raw = document["entries"]
        if not isinstance(entries_raw, list):
            raise Av2ManifestError("manifest entries must be a list")
        try:
            entries = tuple(ManifestEntry(**entry) for entry in entries_raw)
        except (TypeError, ValueError) as exc:
            raise Av2ManifestError("manifest contains an invalid entry") from exc
        if not isinstance(document["manifest_spec"], Mapping) or not isinstance(document["eligibility_spec"], Mapping):
            raise Av2ManifestError("manifest protocol specifications must be mappings")
        return cls(
            stage=document["stage"], source_split=document["source_split"],
            protocol_version=document["protocol_version"],
            protocol_document_sha256=document["protocol_document_sha256"],
            manifest_spec=dict(document["manifest_spec"]), eligibility_spec=dict(document["eligibility_spec"]),
            entries=entries, manifest_sha256=document["manifest_sha256"],
        )

    @classmethod
    def from_json(cls, value: str) -> "Av2PilotManifest":
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise Av2ManifestError("manifest JSON is invalid") from exc
        if not isinstance(parsed, Mapping):
            raise Av2ManifestError("manifest JSON must be an object")
        return cls.from_dict(parsed)


@dataclass(frozen=True)
class ManifestArtifacts:
    json_path: Path
    csv_path: Path
    manifest_sha256: str


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _manifest_spec_dict(spec: ManifestSpec) -> dict[str, Any]:
    return asdict(spec)


def _eligibility_spec_dict() -> dict[str, Any]:
    return asdict(AV2_PILOT_V1.trajectory)


def _stage_spec(stage: str) -> ManifestSpec:
    if stage == "smoke":
        return AV2_PILOT_V1.smoke_manifest
    if stage == "decision":
        return AV2_PILOT_V1.decision_manifest
    raise Av2ManifestError("stage must be 'smoke' or 'decision'")


def _rank(record: Av2InventoryRecord, *, stage: str, assignment: str) -> tuple[str, str]:
    # A digest gives a reproducible permutation without global RNG state or a
    # result-tunable random seed.  scenario_id remains a total-order tie-break.
    digest = hashlib.sha256(
        f"{PROTOCOL_VERSION}|{stage}|{assignment}|{record.scenario_id}".encode("utf-8")
    ).hexdigest()
    return digest, record.scenario_id


def _apportion(total: int, population_by_stratum: Mapping[tuple[str, str], int]) -> dict[tuple[str, str], int]:
    available = sum(population_by_stratum.values())
    if total > available:
        raise Av2ManifestError(f"need {total} eligible official-train scenarios, but only {available} are available")
    if total < 0:
        raise Av2ManifestError("selection total must not be negative")
    quotas = {stratum: total * count / available for stratum, count in population_by_stratum.items()} if available else {}
    result = {stratum: int(quota) for stratum, quota in quotas.items()}
    remainder = total - sum(result.values())
    ranked = sorted(
        population_by_stratum,
        key=lambda stratum: (-(quotas[stratum] - result[stratum]), stratum[0], stratum[1]),
    )
    for stratum in ranked[:remainder]:
        result[stratum] += 1
    return result


def validate_split_isolation(entries: Iterable[ManifestEntry]) -> None:
    """Reject duplicate scenario IDs and any scenario assigned to two Pilot splits."""
    assignments: dict[str, str] = {}
    for entry in entries:
        previous = assignments.get(entry.scenario_id)
        if previous is not None:
            if previous == entry.assignment:
                raise Av2ManifestError(f"duplicate scenario_id in manifest: {entry.scenario_id}")
            raise Av2ManifestError(
                f"scenario_id appears in overlapping Pilot splits: {entry.scenario_id} ({previous}, {entry.assignment})"
            )
        assignments[entry.scenario_id] = entry.assignment


def manifest_overlap_count(manifests: Iterable[Av2PilotManifest]) -> int:
    """Return the number of scenario IDs reused by more than one manifest."""
    seen: set[str] = set()
    overlaps: set[str] = set()
    for manifest in manifests:
        validate_split_isolation(manifest.entries)
        for entry in manifest.entries:
            if entry.scenario_id in seen:
                overlaps.add(entry.scenario_id)
            seen.add(entry.scenario_id)
    return len(overlaps)


def build_pilot_manifest(
    inventory: Iterable[Av2InventoryRecord | Mapping[str, Any]], *, stage: str
) -> Av2PilotManifest:
    """Build the frozen Smoke or Decision manifest from metadata-only inventory.

    Official validation rows are never candidates.  They may be present in the
    inventory so one inventory file can describe a complete AV2 download, but
    no row outside official train can enter the resulting manifest.
    """
    spec = _stage_spec(stage)
    spec.validate()
    records = tuple(_record_from_input(record) for record in inventory)
    all_ids: set[str] = set()
    for record in records:
        if record.official_split not in {"train", "validation"}:
            raise Av2ManifestError(f"unsupported AV2 official_split {record.official_split!r}")
        if record.scenario_id in all_ids:
            raise Av2ManifestError(f"duplicate scenario_id in inventory: {record.scenario_id}")
        all_ids.add(record.scenario_id)

    eligible = [record for record in records if record.official_split == "train" and record.eligible_for_protocol()]
    by_stratum: dict[tuple[str, str], list[Av2InventoryRecord]] = {}
    for record in eligible:
        by_stratum.setdefault(record.stratum, []).append(record)

    requested = {
        "train": spec.train_scenarios,
        "validation": spec.validation_scenarios,
        "development_holdout": spec.development_holdout_scenarios,
    }
    remaining = {stratum: list(values) for stratum, values in by_stratum.items()}
    entries: list[ManifestEntry] = []
    for assignment in _ASSIGNMENT_ORDER:
        quotas = _apportion(requested[assignment], {key: len(value) for key, value in remaining.items()})
        for stratum in sorted(quotas):
            count = quotas[stratum]
            selected = sorted(remaining[stratum], key=lambda value: _rank(value, stage=stage, assignment=assignment))[:count]
            entries.extend(
                ManifestEntry(
                    scenario_id=record.scenario_id,
                    assignment=assignment,
                    motion_type=record.motion_type,
                    city=record.city,
                    focal_track_id=record.focal_track_id,
                )
                for record in selected
            )
            selected_ids = {record.scenario_id for record in selected}
            remaining[stratum] = [record for record in remaining[stratum] if record.scenario_id not in selected_ids]

    manifest = Av2PilotManifest(
        stage=stage,
        source_split=spec.source_split,
        protocol_version=PROTOCOL_VERSION,
        protocol_document_sha256=PROTOCOL_DOCUMENT_SHA256,
        manifest_spec=_manifest_spec_dict(spec),
        eligibility_spec=_eligibility_spec_dict(),
        entries=tuple(sorted(entries, key=lambda entry: (_ASSIGNMENT_ORDER.index(entry.assignment), entry.motion_type, entry.city, entry.scenario_id))),
    )
    if manifest.assignment_counts != requested:
        raise Av2ManifestError("internal error: manifest assignment counts do not match frozen protocol")
    return manifest


def write_manifest(manifest: Av2PilotManifest, directory: str | Path, *, stem: str | None = None) -> ManifestArtifacts:
    """Write canonical JSON and a flat CSV view; does not touch raw AV2 data."""
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    file_stem = stem or f"{manifest.stage}_{manifest.manifest_sha256[:12]}"
    if not file_stem or Path(file_stem).name != file_stem:
        raise Av2ManifestError("stem must be a plain file name")
    json_path = output / f"{file_stem}.json"
    csv_path = output / f"{file_stem}.csv"
    # ``Path.write_text(..., newline=...)`` was added after the project's
    # baseline Python 3.8 runtime, so use ``open`` for stable LF output.
    with json_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(manifest.to_json())
    fields = (
        "schema_version", "manifest_sha256", "protocol_version", "protocol_document_sha256", "stage",
        "source_split", "assignment", "scenario_id", "focal_track_id", "official_split", "motion_type", "city",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for entry in manifest.entries:
            writer.writerow({
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "manifest_sha256": manifest.manifest_sha256,
                "protocol_version": manifest.protocol_version,
                "protocol_document_sha256": manifest.protocol_document_sha256,
                "stage": manifest.stage,
                "source_split": manifest.source_split,
                **asdict(entry),
            })
    return ManifestArtifacts(json_path=json_path, csv_path=csv_path, manifest_sha256=manifest.manifest_sha256)
