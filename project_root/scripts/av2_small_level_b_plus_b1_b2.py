"""Run the nominal Level B+ B1/B2 diagnostics on the frozen Level-B scenes.

This script intentionally covers only motion statistics (B1) and posterior
complementarity (B2).  It reads ``manifests/level_b_20.csv``, uses the frozen
nominal measurement seeds, aggregates repeated seeds within each scene first,
and never injects faults or invokes a GPU.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from av2.datasets.motion_forecasting.scenario_serialization import (
    load_argoverse_scenario_parquet,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.av2.trajectory_transform import AV2Trajectory, compute_eval_mask, to_initial_local_frame
from configs.av2_level_b_plus_v11 import AGGRESSIVE_PROTOCOL, LEVEL_B_PLUS_PROTOCOL_VERSION, MEASUREMENT_SEEDS
from simulation.av2_sensor_ekf import run_av2_sensor_ekfs


PROTOCOL = LEVEL_B_PLUS_PROTOCOL_VERSION
SOURCE_NAMES = ("T1", "T2", "T3")
PAIR_SPECS = ((0, 1, "T1-T2"), (0, 2, "T1-T3"), (1, 2, "T2-T3"))
DEFAULT_SEEDS = MEASUREMENT_SEEDS


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _percentile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.percentile(values, q)) if values.size else float("nan")


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, including deterministic tie handling."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    start = 0
    while start < x.size:
        end = start + 1
        while end < x.size and x[order[end]] == x[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Finite-pair Spearman correlation; NaN for fewer than two varying pairs."""
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("Spearman inputs must have equal shapes")
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 2:
        return float("nan")
    xr, yr = _rankdata(x[valid]), _rankdata(y[valid])
    if np.ptp(xr) == 0.0 or np.ptp(yr) == 0.0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def compute_motion_statistics(local: AV2Trajectory) -> dict[str, float]:
    """Compute timestamp-aware B1 statistics for one local-frame trajectory."""
    if local.headings_rad is None:
        raise ValueError("B1 requires headings")
    speed = np.linalg.norm(local.velocities_mps, axis=1)
    dt = local.dt_seconds
    acceleration = np.linalg.norm(np.diff(local.velocities_mps, axis=0) / dt[:, None], axis=1)
    heading = np.unwrap(np.asarray(local.headings_rad, dtype=np.float64))
    heading_delta = np.diff(heading)
    yaw_rate = np.abs(heading_delta / dt)
    cv_prediction = local.positions_m[:-1] + local.velocities_mps[:-1] * dt[:, None]
    cv_residual = np.linalg.norm(local.positions_m[1:] - cv_prediction, axis=1)
    return {
        "duration": float(local.elapsed_seconds[-1]),
        "total_displacement": float(np.linalg.norm(local.positions_m[-1] - local.positions_m[0])),
        "mean_speed": float(np.mean(speed)),
        "speed_range": float(np.ptp(speed)),
        "acceleration_mean": float(np.mean(acceleration)),
        "acceleration_p95": _percentile(acceleration, 95),
        "acceleration_max": float(np.max(acceleration)),
        "heading_change_total": float(np.sum(np.abs(heading_delta))),
        "yaw_rate_p95": _percentile(yaw_rate, 95),
        "yaw_rate_max": float(np.max(yaw_rate)),
        "cv_residual_mean": float(np.mean(cv_residual)),
        "cv_residual_p95": _percentile(cv_residual, 95),
        "cv_residual_max": float(np.max(cv_residual)),
    }


def _load_local_truth(parquet_path: Path) -> tuple[Any, AV2Trajectory, np.ndarray]:
    scenario = load_argoverse_scenario_parquet(parquet_path)
    focal = next(track for track in scenario.tracks if track.track_id == scenario.focal_track_id)
    states = sorted(focal.object_states, key=lambda state: state.timestep)
    timesteps = np.asarray([state.timestep for state in states], dtype=np.int64)
    timestamps = np.asarray(scenario.timestamps_ns, dtype=np.int64)[timesteps]
    trajectory = AV2Trajectory(
        timestamps_ns=timestamps,
        positions_m=np.asarray([state.position for state in states], dtype=np.float64),
        velocities_mps=np.asarray([state.velocity for state in states], dtype=np.float64),
        observed=np.asarray([state.observed for state in states], dtype=bool),
        headings_rad=np.asarray([state.heading for state in states], dtype=np.float64),
        scenario_id=scenario.scenario_id,
        focal_track_id=scenario.focal_track_id,
    )
    local = to_initial_local_frame(trajectory)
    truth = np.column_stack((local.positions_m, local.velocities_mps))
    return scenario, local, truth


def _ecdf(values: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(list(values), dtype=np.float64))
    return x, np.arange(1, x.size + 1, dtype=np.float64) / x.size


def _aggregate(values: Sequence[float]) -> float:
    return float(np.nanmean(np.asarray(values, dtype=np.float64)))


def run(
    root: Path,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    *,
    output_metadata_dir: Path | None = None,
    output_plot_dir: Path | None = None,
    protocol_name: str = PROTOCOL,
) -> None:
    manifest_path = root / "manifests" / "level_b_20.csv"
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if len(manifest) != 20:
        raise RuntimeError(f"Frozen Level B manifest must contain exactly 20 scenes, found {len(manifest)}")
    if tuple(seeds) != DEFAULT_SEEDS:
        raise ValueError(f"{protocol_name} freezes measurement seeds to {list(DEFAULT_SEEDS)}")

    metadata_dir = output_metadata_dir or root / "metadata" / "level_b_plus"
    plot_dir = output_plot_dir or root / "plots" / "level_b_plus"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    motion_rows: list[dict[str, Any]] = []
    posterior_rows: list[dict[str, Any]] = []
    switching_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    scene_best: list[str] = []
    speed_profiles: list[tuple[str, np.ndarray, np.ndarray]] = []
    heading_profiles: list[tuple[str, np.ndarray, np.ndarray]] = []
    cv_values: list[float] = []
    scene_rmse_values: list[list[float]] = [[], [], []]
    covariance_error_points: list[list[tuple[float, float]]] = [[], [], []]
    timeline_example: tuple[str, np.ndarray, np.ndarray] | None = None

    for scene_index, row in enumerate(manifest):
        scenario, local, truth = _load_local_truth(Path(row["parquet_path"]))
        stats = compute_motion_statistics(local)
        motion_rows.append({"protocol": protocol_name, "scenario_id": scenario.scenario_id, **{k: f"{v:.9g}" for k, v in stats.items()}})
        elapsed = local.elapsed_seconds
        speed_profiles.append((scenario.scenario_id, elapsed, np.linalg.norm(local.velocities_mps, axis=1)))
        heading_profiles.append((scenario.scenario_id, elapsed, np.rad2deg(np.unwrap(local.headings_rad))))
        cv_values.append(stats["cv_residual_mean"])

        eval_mask = compute_eval_mask(local.timestamps_ns, warmup_seconds=1.0)
        per_seed_source: list[list[dict[str, float]]] = []
        per_seed_switching: list[dict[str, Any]] = []
        per_seed_pair: list[dict[str, dict[str, float]]] = []
        per_seed_corr: list[list[float]] = []
        for seed in seeds:
            outputs = run_av2_sensor_ekfs(
                local.timestamps_ns,
                truth,
                rng=np.random.default_rng(seed),
                protocol=AGGRESSIVE_PROTOCOL,
            )
            available = np.asarray(outputs.posterior_available, dtype=bool)
            valid = eval_mask[:, None] & available & np.isfinite(outputs.posterior_mean).all(axis=2)
            common = np.all(valid, axis=1)
            if not np.any(common):
                raise RuntimeError(f"No common post-warmup posterior steps for {scenario.scenario_id}, seed {seed}")
            posterior = np.asarray(outputs.posterior_mean, dtype=np.float64)
            covariance = np.asarray(outputs.posterior_covariance_internal, dtype=np.float64)
            error = np.linalg.norm(posterior[:, :, :2] - truth[:, None, :2], axis=2)
            velocity_error = np.linalg.norm(posterior[:, :, 2:] - truth[:, None, 2:], axis=2)
            pos_trace = np.trace(covariance[:, :, :2, :2], axis1=2, axis2=3)
            source_metrics: list[dict[str, float]] = []
            correlations: list[float] = []
            for source in range(3):
                mask = valid[:, source]
                e, ve, trace = error[mask, source], velocity_error[mask, source], pos_trace[mask, source]
                source_metrics.append({
                    "position_rmse": float(np.sqrt(np.mean(e ** 2))),
                    "velocity_rmse": float(np.sqrt(np.mean(ve ** 2))),
                    "position_p95": _percentile(e, 95),
                    "position_max": float(np.max(e)),
                    "mean_trace_p_pos": float(np.mean(trace)),
                    "p95_trace_p_pos": _percentile(trace, 95),
                })
                correlations.append(spearman_correlation(trace, e ** 2))
            per_seed_source.append(source_metrics)
            per_seed_corr.append(correlations)

            best_timeline = np.argmin(error[common], axis=1)
            switches = int(np.count_nonzero(np.diff(best_timeline)))
            fractions = np.bincount(best_timeline, minlength=3) / best_timeline.size
            per_seed_switching.append({"switch_count": switches, "fractions": fractions})
            if timeline_example is None and scene_index == 0 and seed == seeds[0]:
                timeline_example = (scenario.scenario_id, elapsed[common], best_timeline)

            pair_metrics: dict[str, dict[str, float]] = {}
            for left, right, name in PAIR_SPECS:
                distance = np.linalg.norm(posterior[common, left, :2] - posterior[common, right, :2], axis=1)
                pair_metrics[name] = {
                    "mean": float(np.mean(distance)), "p95": _percentile(distance, 95), "max": float(np.max(distance))
                }
            per_seed_pair.append(pair_metrics)

        # Seed repetition is reduced here, before any cross-scene result.
        scene_rmse = []
        for source, source_name in enumerate(SOURCE_NAMES):
            metrics = {key: _aggregate([seed_metrics[source][key] for seed_metrics in per_seed_source]) for key in per_seed_source[0][source]}
            scene_rmse.append(metrics["position_rmse"])
            scene_rmse_values[source].append(metrics["position_rmse"])
            covariance_error_points[source].append(
                (metrics["mean_trace_p_pos"], metrics["position_rmse"] ** 2)
            )
            pair_columns: dict[str, float] = {}
            for _, _, pair_name in PAIR_SPECS:
                pair_columns[f"{pair_name}_distance_mean"] = _aggregate([item[pair_name]["mean"] for item in per_seed_pair])
                pair_columns[f"{pair_name}_distance_p95"] = _aggregate([item[pair_name]["p95"] for item in per_seed_pair])
                pair_columns[f"{pair_name}_distance_max"] = _aggregate([item[pair_name]["max"] for item in per_seed_pair])
            posterior_rows.append({
                "protocol": protocol_name, "scenario_id": scenario.scenario_id, "source": source_name, "measurement_seed_count": len(seeds),
                **{key: f"{value:.9g}" for key, value in metrics.items()},
                **{key: f"{value:.9g}" for key, value in pair_columns.items()},
            })
            rhos = [seed_corr[source] for seed_corr in per_seed_corr]
            correlation_rows.append({
                "protocol": protocol_name, "scenario_id": scenario.scenario_id, "source": source_name,
                "spearman_trace_p_pos_vs_squared_position_error": f"{_aggregate(rhos):.9g}",
                "valid_seed_correlations": int(np.isfinite(rhos).sum()),
            })
        scene_best.append(SOURCE_NAMES[int(np.argmin(scene_rmse))])
        switching_rows.append({
            "protocol": protocol_name, "scenario_id": scenario.scenario_id,
            "switch_count_mean": f"{_aggregate([x['switch_count'] for x in per_seed_switching]):.9g}",
            "switch_count_min": min(x["switch_count"] for x in per_seed_switching),
            "switch_count_max": max(x["switch_count"] for x in per_seed_switching),
            **{f"{name}_best_step_fraction": f"{_aggregate([x['fractions'][i] for x in per_seed_switching]):.9g}" for i, name in enumerate(SOURCE_NAMES)},
        })

    motion_fields = ["protocol", "scenario_id", "duration", "total_displacement", "mean_speed", "speed_range", "acceleration_mean", "acceleration_p95", "acceleration_max", "heading_change_total", "yaw_rate_p95", "yaw_rate_max", "cv_residual_mean", "cv_residual_p95", "cv_residual_max"]
    pair_fields = [f"{name}_distance_{metric}" for _, _, name in PAIR_SPECS for metric in ("mean", "p95", "max")]
    posterior_fields = ["protocol", "scenario_id", "source", "measurement_seed_count", "position_rmse", "velocity_rmse", "position_p95", "position_max", "mean_trace_p_pos", "p95_trace_p_pos", *pair_fields]
    _write_csv(metadata_dir / "av2_motion_statistics.csv", motion_rows, motion_fields)
    _write_csv(metadata_dir / "posterior_scene_metrics.csv", posterior_rows, posterior_fields)
    counts = {name: scene_best.count(name) for name in SOURCE_NAMES}
    _write_csv(metadata_dir / "posterior_best_source_counts.csv", [
        {"protocol": protocol_name, "source": name, "best_scene_count": counts[name], "best_scene_fraction": f"{counts[name] / len(manifest):.9g}"} for name in SOURCE_NAMES
    ], ["protocol", "source", "best_scene_count", "best_scene_fraction"])
    _write_csv(metadata_dir / "posterior_switching.csv", switching_rows, ["protocol", "scenario_id", "switch_count_mean", "switch_count_min", "switch_count_max", *[f"{name}_best_step_fraction" for name in SOURCE_NAMES]])
    _write_csv(metadata_dir / "posterior_covariance_error_correlation.csv", correlation_rows, ["protocol", "scenario_id", "source", "spearman_trace_p_pos_vs_squared_position_error", "valid_seed_correlations"])

    _plot_motion(plot_dir, speed_profiles, heading_profiles, cv_values)
    _plot_posterior(plot_dir, scene_rmse_values, covariance_error_points, timeline_example)
    print(f"protocol={protocol_name} scenes={len(manifest)} seeds={list(seeds)}")
    print(f"outputs={metadata_dir} plots={plot_dir}")


def _plot_motion(plot_dir: Path, speed_profiles: Sequence[tuple[str, np.ndarray, np.ndarray]], heading_profiles: Sequence[tuple[str, np.ndarray, np.ndarray]], cv_values: Sequence[float]) -> None:
    for filename, profiles, ylabel in (("speed_profiles.pdf", speed_profiles, "speed (m/s)"), ("heading_profiles.pdf", heading_profiles, "unwrapped heading (deg)")):
        fig, ax = plt.subplots(figsize=(8, 5))
        for scenario_id, elapsed, values in profiles:
            ax.plot(elapsed, values, linewidth=0.9, alpha=0.7, label=scenario_id[:8])
        ax.set(xlabel="elapsed time (s)", ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=5, ncol=4)
        fig.tight_layout(); fig.savefig(plot_dir / filename); plt.close(fig)
    x, y = _ecdf(cv_values)
    fig, ax = plt.subplots(figsize=(7, 5)); ax.plot(x, y); ax.set(xlabel="scene-level mean CV residual (m)", ylabel="ECDF"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(plot_dir / "cv_residual_ecdf.pdf"); plt.close(fig)


def _plot_posterior(plot_dir: Path, scene_rmse_values: Sequence[Sequence[float]], points: Sequence[Sequence[tuple[float, float]]], timeline: tuple[str, np.ndarray, np.ndarray] | None) -> None:
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c")
    fig, ax = plt.subplots(figsize=(7, 5))
    for source, name in enumerate(SOURCE_NAMES):
        x, y = _ecdf(scene_rmse_values[source]); ax.plot(x, y, label=name, color=colors[source])
    ax.set(xlabel="scene-level position RMSE (m)", ylabel="ECDF"); ax.grid(alpha=0.25); ax.legend(); fig.tight_layout(); fig.savefig(plot_dir / "posterior_error_ecdf.pdf"); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for source, name in enumerate(SOURCE_NAMES):
        values = np.asarray(points[source], dtype=np.float64)
        axes[source].scatter(values[:, 0], values[:, 1], s=3, alpha=0.15, color=colors[source])
        axes[source].set(title=name, xlabel="trace(P_pos)", ylabel="squared position error")
        axes[source].grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(plot_dir / "covariance_vs_error.pdf"); plt.close(fig)
    if timeline is not None:
        scenario_id, elapsed, best = timeline
        fig, ax = plt.subplots(figsize=(9, 3)); ax.step(elapsed, best + 1, where="post"); ax.set_yticks((1, 2, 3), SOURCE_NAMES); ax.set(xlabel="elapsed time (s)", title=f"Best-source timeline: {scenario_id}"); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(plot_dir / "posterior_best_source_timeline.pdf"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AV2 Level B+ B1/B2 nominal diagnostics.")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
