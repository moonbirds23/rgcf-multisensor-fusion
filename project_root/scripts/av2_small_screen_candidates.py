from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
from av2.datasets.motion_forecasting.scenario_serialization import (
    load_argoverse_scenario_parquet,
)


STAT_FIELDS = (
    "scenario_id",
    "source_split",
    "city_name",
    "focal_track_id",
    "focal_object_type",
    "focal_category",
    "num_focal_states",
    "missing_timestep_count",
    "total_displacement_m",
    "mean_speed_mps",
    "position_finite",
    "velocity_finite",
    "heading_finite",
    "eligible",
    "exclusion_reason",
    "parquet_path",
    "map_path",
)


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value)).lower()


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def inspect_candidate(candidate_dir: Path) -> dict[str, Any]:
    scenario_id = candidate_dir.name
    parquet_path = candidate_dir / f"scenario_{scenario_id}.parquet"
    map_path = candidate_dir / f"log_map_archive_{scenario_id}.json"
    if not parquet_path.is_file() or not map_path.is_file():
        return {
            "scenario_id": scenario_id,
            "source_split": "train",
            "eligible": False,
            "exclusion_reason": "file_missing_or_download_failed",
            "parquet_path": str(parquet_path),
            "map_path": str(map_path),
        }
    try:
        scenario = load_argoverse_scenario_parquet(parquet_path)
        focal_track = next(
            track for track in scenario.tracks if track.track_id == scenario.focal_track_id
        )
    except Exception as exc:
        return {
            "scenario_id": scenario_id,
            "source_split": "train",
            "eligible": False,
            "exclusion_reason": f"official_api_read_failed:{type(exc).__name__}",
            "parquet_path": str(parquet_path),
            "map_path": str(map_path),
        }

    states = sorted(focal_track.object_states, key=lambda state: state.timestep)
    positions = np.asarray([state.position for state in states], dtype=np.float64)
    velocities = np.asarray([state.velocity for state in states], dtype=np.float64)
    headings = np.asarray([state.heading for state in states], dtype=np.float64)
    timesteps = np.asarray([state.timestep for state in states], dtype=np.int64)
    object_type = _enum_value(focal_track.object_type)
    category = _enum_value(focal_track.category)
    missing = int(timesteps[-1] - timesteps[0] + 1 - np.unique(timesteps).size)
    displacement = float(np.linalg.norm(positions[-1] - positions[0]))
    speed = np.linalg.norm(velocities, axis=1)
    reasons: list[str] = []
    if object_type != "vehicle":
        reasons.append("focal_object_not_vehicle")
    # AV2 0.2.1 exposes the enum value as "3" for FOCAL_TRACK.
    if category not in {"3", "focal_track"}:
        reasons.append("category_not_focal_track")
    if len(states) < 100:
        reasons.append("insufficient_states")
    if missing > 5:
        reasons.append("too_many_missing_timesteps")
    if displacement < 10.0:
        reasons.append("displacement_too_small")
    if not np.isfinite(positions).all():
        reasons.append("nonfinite_position")
    if not np.isfinite(velocities).all():
        reasons.append("nonfinite_velocity")
    if not np.isfinite(headings).all():
        reasons.append("nonfinite_heading")
    return {
        "scenario_id": scenario.scenario_id,
        "source_split": "train",
        "city_name": scenario.city_name,
        "focal_track_id": scenario.focal_track_id,
        "focal_object_type": object_type,
        "focal_category": category,
        "num_focal_states": len(states),
        "missing_timestep_count": missing,
        "total_displacement_m": f"{displacement:.6f}",
        "mean_speed_mps": f"{float(np.mean(speed)):.6f}",
        "position_finite": bool(np.isfinite(positions).all()),
        "velocity_finite": bool(np.isfinite(velocities).all()),
        "heading_finite": bool(np.isfinite(headings).all()),
        "eligible": not reasons,
        "exclusion_reason": ";".join(reasons),
        "parquet_path": str(parquet_path),
        "map_path": str(map_path),
    }


def run(root: Path) -> None:
    candidate_root = root / "raw" / "train_candidates"
    manifest_dir = root / "manifests"
    metadata_dir = root / "metadata"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        inspect_candidate(path)
        for path in sorted(candidate_root.iterdir(), key=lambda item: item.name)
        if path.is_dir()
    ]
    eligible = [row for row in rows if row.get("eligible") is True]
    excluded = [row for row in rows if row.get("eligible") is not True]
    _write_csv(metadata_dir / "trajectory_statistics.csv", rows, STAT_FIELDS)
    _write_csv(
        manifest_dir / "eligible_scenarios.csv",
        eligible,
        STAT_FIELDS,
    )
    _write_csv(
        metadata_dir / "exclusion_log.csv",
        excluded,
        STAT_FIELDS,
    )
    print(f"candidates={len(rows)} eligible={len(eligible)} excluded={len(excluded)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen downloaded AV2 small-scene candidates.")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
