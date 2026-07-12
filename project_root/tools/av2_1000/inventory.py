"""Read candidate AV2 parquet files and produce a deterministic inventory."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .common import PROTOCOL_NAME, atomic_json, git_commit, sha256_file, write_csv

FIELDS = ["scenario_id", "official_split", "parquet_path", "file_size_bytes", "file_sha256", "city_name", "focal_track_id", "focal_object_type", "expected_timesteps", "state_count", "observed_timesteps", "missing_timesteps", "all_finite", "total_displacement_m", "mean_speed_mps", "speed_range_mps", "heading_change_deg", "max_yaw_rate_deg_s", "motion_type", "eligible", "rejection_code", "rejection_detail", "inventory_schema_version", "source_reader_version"]


def _motion(positions: np.ndarray, velocities: np.ndarray, headings: np.ndarray, elapsed: np.ndarray) -> tuple[float, float, float, float, str]:
    speed = np.linalg.norm(velocities, axis=1)
    displacement = float(np.linalg.norm(positions[-1] - positions[0]))
    duration = max(float(elapsed[-1] - elapsed[0]), 1e-9)
    mean_speed = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum() / duration)
    speed_range = float(speed.max() - speed.min())
    heading_change = float(abs(np.rad2deg(np.unwrap(headings)[-1] - np.unwrap(headings)[0])))
    dt = np.diff(elapsed); yaw = np.abs(np.rad2deg(np.diff(np.unwrap(headings)) / np.maximum(dt, 1e-9))) if dt.size else np.zeros(1)
    max_yaw = float(yaw.max())
    if heading_change >= 45.0 or (speed_range >= 6.0 and max_yaw >= 12.0): label = "high_dynamic"
    elif heading_change >= 15.0: label = "turning"
    elif speed_range >= 3.0: label = "longitudinal"
    else: label = "stable_straight"
    return displacement, mean_speed, speed_range, max_yaw, label


def inventory_one(parquet: Path, official_split: str, expected_steps: int = 110) -> dict[str, Any]:
    from av2.datasets.motion_forecasting.scenario_serialization import load_argoverse_scenario_parquet
    base = {"parquet_path": str(parquet.resolve()), "file_size_bytes": parquet.stat().st_size, "file_sha256": sha256_file(parquet), "official_split": official_split, "expected_timesteps": expected_steps, "inventory_schema_version": 1, "source_reader_version": "av2-0.2.1"}
    scenario = load_argoverse_scenario_parquet(parquet)
    track = next((item for item in scenario.tracks if item.track_id == scenario.focal_track_id), None)
    if track is None: raise ValueError("focal track missing")
    states = sorted(track.object_states, key=lambda item: item.timestep)
    timestep = np.asarray([item.timestep for item in states], dtype=np.int64)
    positions = np.asarray([item.position for item in states], dtype=np.float64)
    velocities = np.asarray([item.velocity for item in states], dtype=np.float64)
    headings = np.asarray([item.heading for item in states], dtype=np.float64)
    expected = np.arange(expected_steps, dtype=np.int64)
    finite = bool(np.isfinite(positions).all() and np.isfinite(velocities).all() and np.isfinite(headings).all())
    complete = bool(timestep.shape == expected.shape and np.array_equal(timestep, expected))
    elapsed = np.asarray(scenario.timestamps_ns, dtype=np.int64)[timestep].astype(np.float64) * 1e-9
    displacement, mean_speed, speed_range, max_yaw, motion_type = _motion(positions, velocities, headings, elapsed)
    object_type = getattr(track.object_type, "name", str(track.object_type))
    category = getattr(track.category, "name", str(track.category))
    code = ""
    if object_type != "VEHICLE": code = "wrong_object_type"
    elif category != "FOCAL_TRACK": code = "wrong_category"
    elif not complete: code = "incomplete_focal_track"
    elif not finite: code = "non_finite"
    elif displacement < 10.0: code = "displacement_too_small"
    return {**base, "scenario_id": scenario.scenario_id, "city_name": str(getattr(scenario, "city_name", getattr(scenario, "city", "UNKNOWN"))), "focal_track_id": scenario.focal_track_id, "focal_object_type": object_type, "state_count": int(timestep.size), "observed_timesteps": int(sum(bool(item.observed) for item in states)), "missing_timesteps": int(expected_steps - timestep.size), "all_finite": finite, "total_displacement_m": f"{displacement:.9g}", "mean_speed_mps": f"{mean_speed:.9g}", "speed_range_mps": f"{speed_range:.9g}", "heading_change_deg": f"{abs(np.rad2deg(np.unwrap(headings)[-1] - np.unwrap(headings)[0])):.9g}", "max_yaw_rate_deg_s": f"{max_yaw:.9g}", "motion_type": motion_type, "eligible": str(not code).lower(), "rejection_code": code, "rejection_detail": "" if not code else code}


def build_inventory(input_root: Path, official_split: str, output_csv: Path, repo_root: Path) -> dict[str, Any]:
    rows, failures = [], []
    for folder in sorted(item for item in input_root.iterdir() if item.is_dir()):
        parquet = folder / ("scenario_" + folder.name + ".parquet")
        try: rows.append(inventory_one(parquet, official_split))
        except Exception as exc: failures.append({"path": str(parquet), "error": repr(exc)})
    write_csv(output_csv, rows, FIELDS)
    summary = {"protocol_name": PROTOCOL_NAME, "official_split": official_split, "total_files": len(rows) + len(failures), "read_success": len(rows), "read_failed": len(failures), "eligible_count": sum(row["eligible"] == "true" for row in rows), "ineligible_count": sum(row["eligible"] != "true" for row in rows), "git_commit": git_commit(repo_root), "failures": failures}
    atomic_json(output_csv.with_name(output_csv.stem + "_summary.json"), summary)
    if failures: raise RuntimeError("inventory completed with read failures; inspect summary before continuing")
    return summary
