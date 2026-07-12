from __future__ import annotations

import argparse
import csv
from importlib import metadata
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from av2.datasets.motion_forecasting.scenario_serialization import (
    load_argoverse_scenario_parquet,
)
from av2.map.map_api import ArgoverseStaticMap


AUDIT_FIELDS = (
    "scenario_id",
    "city_name",
    "focal_track_id",
    "num_scenario_timestamps",
    "num_tracks",
    "focal_object_type",
    "focal_category",
    "num_focal_states",
    "first_focal_timestep",
    "last_focal_timestep",
    "observed_state_count",
    "unobserved_state_count",
    "missing_timestep_count",
    "position_has_nan",
    "velocity_has_nan",
    "heading_has_nan",
    "map_file_exists",
    "parquet_path",
    "map_path",
)


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _scenario_paths(candidate_dir: Path) -> tuple[Path, Path]:
    scenario_id = candidate_dir.name
    return (
        candidate_dir / f"scenario_{scenario_id}.parquet",
        candidate_dir / f"log_map_archive_{scenario_id}.json",
    )


def _plot_map(ax: plt.Axes, map_path: Path) -> None:
    static_map = ArgoverseStaticMap.from_json(map_path)
    for lane in static_map.vector_lane_segments.values():
        left = np.asarray(lane.left_lane_boundary.xyz, dtype=np.float64)
        right = np.asarray(lane.right_lane_boundary.xyz, dtype=np.float64)
        ax.plot(left[:, 0], left[:, 1], color="0.72", linewidth=0.7, zorder=1)
        ax.plot(right[:, 0], right[:, 1], color="0.72", linewidth=0.7, zorder=1)


def _plot_audit(
    output_path: Path,
    scenario_id: str,
    positions: np.ndarray,
    velocities: np.ndarray,
    headings: np.ndarray,
    timesteps: np.ndarray,
    observed: np.ndarray,
    map_path: Path,
    include_map: bool,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    trajectory_ax = axes[0, 0]
    if include_map:
        _plot_map(trajectory_ax, map_path)
    trajectory_ax.plot(positions[:, 0], positions[:, 1], "-", color="#1f77b4", label="focal")
    trajectory_ax.scatter(
        positions[observed, 0], positions[observed, 1], s=10, color="#2ca02c", label="observed"
    )
    trajectory_ax.scatter(
        positions[~observed, 0], positions[~observed, 1], s=10, color="#d62728", label="unobserved"
    )
    trajectory_ax.set_title("Focal trajectory" + (" with AV2 map" if include_map else ""))
    trajectory_ax.set_aspect("equal", adjustable="box")
    trajectory_ax.legend(fontsize=8)

    speed = np.linalg.norm(velocities, axis=1)
    axes[0, 1].plot(timesteps, speed, color="#1f77b4")
    axes[0, 1].set_title("Speed magnitude")
    axes[0, 1].set_xlabel("timestep")
    axes[0, 1].set_ylabel("m/s")

    axes[1, 0].plot(timesteps, np.unwrap(headings), color="#9467bd")
    axes[1, 0].set_title("Unwrapped heading")
    axes[1, 0].set_xlabel("timestep")
    axes[1, 0].set_ylabel("rad")

    axes[1, 1].step(timesteps, observed.astype(int), where="mid", color="#ff7f0e")
    axes[1, 1].set_ylim(-0.1, 1.1)
    axes[1, 1].set_yticks((0, 1), ("False", "True"))
    axes[1, 1].set_title("AV2 observed flag")
    axes[1, 1].set_xlabel("timestep")

    fig.suptitle(scenario_id)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def audit_candidate(candidate_dir: Path, plot_dir: Path, include_map: bool) -> dict[str, Any]:
    parquet_path, map_path = _scenario_paths(candidate_dir)
    if not parquet_path.is_file() or not map_path.is_file():
        raise FileNotFoundError(f"Incomplete candidate directory: {candidate_dir}")
    scenario = load_argoverse_scenario_parquet(parquet_path)
    focal_track = next(
        track for track in scenario.tracks if track.track_id == scenario.focal_track_id
    )
    states = sorted(focal_track.object_states, key=lambda state: state.timestep)
    positions = np.asarray([state.position for state in states], dtype=np.float64)
    velocities = np.asarray([state.velocity for state in states], dtype=np.float64)
    headings = np.asarray([state.heading for state in states], dtype=np.float64)
    timesteps = np.asarray([state.timestep for state in states], dtype=np.int64)
    observed = np.asarray([state.observed for state in states], dtype=bool)
    missing = int(timesteps[-1] - timesteps[0] + 1 - np.unique(timesteps).size)

    _plot_audit(
        plot_dir / f"{scenario.scenario_id}_level_a.png",
        scenario.scenario_id,
        positions,
        velocities,
        headings,
        timesteps,
        observed,
        map_path,
        include_map,
    )
    return {
        "scenario_id": scenario.scenario_id,
        "city_name": scenario.city_name,
        "focal_track_id": scenario.focal_track_id,
        "num_scenario_timestamps": len(scenario.timestamps_ns),
        "num_tracks": len(scenario.tracks),
        "focal_object_type": _enum_value(focal_track.object_type),
        "focal_category": _enum_value(focal_track.category),
        "num_focal_states": len(states),
        "first_focal_timestep": int(timesteps[0]),
        "last_focal_timestep": int(timesteps[-1]),
        "observed_state_count": int(observed.sum()),
        "unobserved_state_count": int((~observed).sum()),
        "missing_timestep_count": missing,
        "position_has_nan": not bool(np.isfinite(positions).all()),
        "velocity_has_nan": not bool(np.isfinite(velocities).all()),
        "heading_has_nan": not bool(np.isfinite(headings).all()),
        "map_file_exists": map_path.is_file(),
        "parquet_path": str(parquet_path),
        "map_path": str(map_path),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(root: Path, level_a_count: int = 5) -> None:
    candidate_root = root / "raw" / "train_candidates"
    metadata_dir = root / "metadata"
    manifest_dir = root / "manifests"
    plot_dir = root / "plots" / "level_a_truth"
    report_dir = root / "reports"
    for directory in (metadata_dir, manifest_dir, plot_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for candidate_dir in sorted(candidate_root.iterdir(), key=lambda path: path.name):
        if not candidate_dir.is_dir():
            continue
        try:
            row = audit_candidate(candidate_dir, plot_dir, include_map=not rows)
        except Exception as exc:  # preserve the failure and continue to the next candidate
            failures.append(f"{candidate_dir.name}: {type(exc).__name__}: {exc}")
            continue
        rows.append(row)
        if len(rows) == level_a_count:
            break

    if len(rows) != level_a_count:
        raise RuntimeError(f"Only {len(rows)} of {level_a_count} Level A scenes were readable")
    _write_csv(metadata_dir / "av2_schema_audit.csv", rows, AUDIT_FIELDS)
    _write_csv(
        manifest_dir / "level_a_5.csv",
        [{"scenario_id": row["scenario_id"]} for row in rows],
        ("scenario_id",),
    )

    checks = {
        "read_success_rate": len(rows) / level_a_count,
        "focal_track_found_rate": sum(bool(row["focal_track_id"]) for row in rows) / level_a_count,
        "finite_position_rate": sum(not row["position_has_nan"] for row in rows) / level_a_count,
        "finite_velocity_rate": sum(not row["velocity_has_nan"] for row in rows) / level_a_count,
        "finite_heading_rate": sum(not row["heading_has_nan"] for row in rows) / level_a_count,
        "map_file_found_rate": sum(row["map_file_exists"] for row in rows) / level_a_count,
    }
    passed = all(value == 1.0 for value in checks.values())
    report_lines = [
        "# AV2 small-scene Level A report",
        "",
        f"- Result: {'PASS' if passed else 'FAIL'}",
        f"- AV2 API package: {metadata.version('av2')}",
        f"- Scenes audited: {len(rows)}",
        "- Source split: official train only",
        "",
        "## Acceptance checks",
        "",
    ]
    report_lines.extend(f"- {name}: {value:.1%}" for name, value in checks.items())
    report_lines.extend(["", "## Selected scenarios", ""])
    report_lines.extend(f"- {row['scenario_id']} ({row['city_name']}, {row['focal_object_type']})" for row in rows)
    if failures:
        report_lines.extend(["", "## Read failures skipped before selection", ""])
        report_lines.extend(f"- {failure}" for failure in failures)
    report_lines.extend(
        [
            "",
            "## Environment note",
            "",
            "The repository baseline is Python 3.8. The newest PyPI AV2 package usable without changing the Python runtime was pinned to av2==0.2.1; av2==0.3.1 imports Python 3.9-only NumPy typing syntax.",
            "",
        ]
    )
    (report_dir / "level_a_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    if not passed:
        raise RuntimeError("Level A acceptance checks failed; inspect the report")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AV2 small-scene Level A audit.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    run(args.root.resolve(), level_a_count=args.count)


if __name__ == "__main__":
    main()
