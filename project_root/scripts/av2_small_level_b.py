from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from av2.datasets.motion_forecasting.scenario_serialization import (
    load_argoverse_scenario_parquet,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.av2_config import AV2_PILOT_V1
from data.av2.feature_builder import AV2FeatureArrays, build_av2_feature_arrays
from data.av2.trajectory_transform import AV2Trajectory, to_initial_local_frame
from simulation.av2_sensor_ekf import run_av2_sensor_ekfs


METRIC_FIELDS = (
    "scenario_id",
    "city_name",
    "measurement_seed",
    "num_steps",
    "eval_steps",
    "scene_success",
    "finite_measurements",
    "finite_ekf",
    "symmetric_covariance",
    "positive_covariance_count",
    "covariance_count",
    "positive_covariance_rate",
    "posteriors_not_identical",
    "all_posteriors_divergent",
    "t1_position_rmse_m",
    "t2_position_rmse_m",
    "t3_position_rmse_m",
    "e1_pair_residual_std",
    "e2_pair_residual_std",
    "evidence_residual_has_variation",
)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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
    target = np.column_stack((local.positions_m, local.velocities_mps)).astype(np.float64)
    return scenario, local, target


def _measurements_are_finite(outputs: Any) -> bool:
    for row in outputs.measurements:
        for measurement in row:
            if not measurement.valid:
                continue
            if not all(
                value is not None and np.isfinite(np.asarray(value)).all()
                for value in (measurement.z, measurement.R_actual, measurement.R_reported)
            ):
                return False
    return True


def _covariance_checks(outputs: Any) -> tuple[bool, bool, int, int]:
    finite = True
    symmetric = True
    positive = 0
    count = 0
    covariance = np.asarray(outputs.posterior_covariance_internal, dtype=np.float64)
    available = np.asarray(outputs.posterior_available, dtype=bool)
    for matrix in covariance[available]:
        count += 1
        finite = finite and bool(np.isfinite(matrix).all())
        symmetric = symmetric and bool(np.allclose(matrix, matrix.T, atol=1e-9, rtol=0.0))
        if finite and float(np.linalg.eigvalsh((matrix + matrix.T) / 2.0).min()) > 0.0:
            positive += 1
    return finite, symmetric, positive, count


def _residual_std(features: AV2FeatureArrays, evidence_index: int) -> float:
    values = features.diagnostics.postfit_residual_norm[:, 3 + evidence_index, :]
    valid = np.asarray(features.eval_mask, dtype=bool)[:, None] & np.isfinite(values)
    selected = values[valid]
    return float(np.std(selected)) if selected.size else 0.0


def _plot_level_b(
    output_path: Path,
    scenario_id: str,
    local: AV2Trajectory,
    target: np.ndarray,
    outputs: Any,
    features: AV2FeatureArrays,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    posterior = np.asarray(outputs.posterior_mean)
    eval_mask = np.asarray(features.eval_mask, dtype=bool)
    axes[0, 0].plot(target[:, 0], target[:, 1], color="black", linewidth=2, label="reference")
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c")
    for index, color in enumerate(colors):
        axes[0, 0].plot(posterior[:, index, 0], posterior[:, index, 1], color=color, label=f"T{index+1}")
    for spec in AV2_PILOT_V1.sensors[1:]:
        if spec.local_position_xy_m is not None:
            axes[0, 0].scatter(*spec.local_position_xy_m, marker="x", s=45, label=spec.name)
    axes[0, 0].set_aspect("equal", adjustable="box")
    axes[0, 0].set_title("Local layout and posterior tracks")
    axes[0, 0].legend(fontsize=7, ncol=2)

    elapsed = local.elapsed_seconds
    for index, color in enumerate(colors):
        error = np.linalg.norm(posterior[:, index, :2] - target[:, None, :2][:, 0], axis=1)
        axes[0, 1].plot(elapsed[eval_mask], error[eval_mask], color=color, label=f"T{index+1}")
    axes[0, 1].set_title("Posterior position errors")
    axes[0, 1].set_xlabel("time (s)")
    axes[0, 1].set_ylabel("m")
    axes[0, 1].legend(fontsize=8)

    covariance = np.asarray(outputs.posterior_covariance_internal)
    for index, color in enumerate(colors):
        trace = np.trace(covariance[:, index], axis1=1, axis2=2)
        axes[0, 2].plot(elapsed, trace, color=color, label=f"T{index+1}")
    axes[0, 2].set_title("Posterior covariance trace")
    axes[0, 2].set_xlabel("time (s)")
    axes[0, 2].legend(fontsize=8)

    for evidence_index, label in enumerate(("E1 AOA", "E2 UWB")):
        ax = axes[1, evidence_index]
        residual = features.diagnostics.postfit_residual_norm[:, 3 + evidence_index, :]
        for posterior_index, color in enumerate(colors):
            ax.plot(elapsed, residual[:, posterior_index], color=color, label=f"to T{posterior_index+1}")
        ax.set_title(f"{label} normalized residual")
        ax.set_xlabel("time (s)")
        ax.legend(fontsize=8)

    axes[1, 2].plot(local.positions_m[:, 0], local.positions_m[:, 1], color="black")
    axes[1, 2].quiver(
        local.positions_m[::10, 0],
        local.positions_m[::10, 1],
        local.velocities_mps[::10, 0],
        local.velocities_mps[::10, 1],
        color="#9467bd",
        scale_units="xy",
        scale=1,
    )
    axes[1, 2].set_aspect("equal", adjustable="box")
    axes[1, 2].set_title("Reference trajectory and velocity")
    fig.suptitle(scenario_id)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run(root: Path, count: int = 20, measurement_seed: int = 100) -> None:
    eligible_path = root / "manifests" / "eligible_scenarios.csv"
    with eligible_path.open(encoding="utf-8", newline="") as handle:
        eligible = sorted(csv.DictReader(handle), key=lambda row: row["scenario_id"])
    selected = eligible[:count]
    if len(selected) < count:
        raise RuntimeError(f"Need {count} eligible scenes, found {len(selected)}")
    plot_dir = root / "plots" / "level_b_posteriors"
    truth_dir = root / "cache" / "truth"
    report_dir = root / "reports"
    for directory in (plot_dir, truth_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    metrics: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        scenario, local, target = _load_local_truth(Path(row["parquet_path"]))
        seed = measurement_seed + index
        outputs = run_av2_sensor_ekfs(
            local.timestamps_ns,
            target,
            rng=np.random.default_rng(seed),
        )
        features = build_av2_feature_arrays(outputs, target, warmup_seconds=1.0)
        finite_measurements = _measurements_are_finite(outputs)
        finite_ekf, symmetric, positive, covariance_count = _covariance_checks(outputs)
        rate = positive / covariance_count if covariance_count else 0.0
        posterior = np.asarray(outputs.posterior_mean)
        pair_distances = [
            float(np.nanmean(np.linalg.norm(posterior[:, left, :2] - posterior[:, right, :2], axis=1)))
            for left, right in ((0, 1), (0, 2), (1, 2))
        ]
        not_identical = all(distance > 1e-6 for distance in pair_distances)
        eval_mask = np.asarray(features.eval_mask, dtype=bool)
        rmses = [
            float(np.sqrt(np.mean(np.sum((posterior[eval_mask, source, :2] - target[eval_mask, :2]) ** 2, axis=1))))
            for source in range(3)
        ]
        all_divergent = all(value > 100.0 for value in rmses)
        e1_std = _residual_std(features, 0)
        e2_std = _residual_std(features, 1)
        residual_variation = e1_std > 1e-6 and e2_std > 1e-6
        success = all(
            (
                finite_measurements,
                finite_ekf,
                symmetric,
                rate >= 0.999,
                not_identical,
                not all_divergent,
                residual_variation,
            )
        )
        metrics.append(
            {
                "scenario_id": scenario.scenario_id,
                "city_name": scenario.city_name,
                "measurement_seed": seed,
                "num_steps": local.num_steps,
                "eval_steps": int(eval_mask.sum()),
                "scene_success": success,
                "finite_measurements": finite_measurements,
                "finite_ekf": finite_ekf,
                "symmetric_covariance": symmetric,
                "positive_covariance_count": positive,
                "covariance_count": covariance_count,
                "positive_covariance_rate": f"{rate:.9f}",
                "posteriors_not_identical": not_identical,
                "all_posteriors_divergent": all_divergent,
                "t1_position_rmse_m": f"{rmses[0]:.6f}",
                "t2_position_rmse_m": f"{rmses[1]:.6f}",
                "t3_position_rmse_m": f"{rmses[2]:.6f}",
                "e1_pair_residual_std": f"{e1_std:.6f}",
                "e2_pair_residual_std": f"{e2_std:.6f}",
                "evidence_residual_has_variation": residual_variation,
            }
        )
        np.savez_compressed(
            truth_dir / f"{scenario.scenario_id}.npz",
            timestamps_ns=local.timestamps_ns,
            target=target,
            eval_mask=eval_mask,
        )
        if index < 3:
            _plot_level_b(
                plot_dir / f"{scenario.scenario_id}_level_b.png",
                scenario.scenario_id,
                local,
                target,
                outputs,
                features,
            )

    _write_csv(root / "manifests" / "level_b_20.csv", selected, tuple(selected[0].keys()))
    _write_csv(root / "metadata" / "level_b_metrics.csv", metrics, METRIC_FIELDS)
    success_count = sum(bool(row["scene_success"]) for row in metrics)
    scene_success_rate = success_count / len(metrics)
    positive_total = sum(int(row["positive_covariance_count"]) for row in metrics)
    covariance_total = sum(int(row["covariance_count"]) for row in metrics)
    covariance_rate = positive_total / covariance_total
    passed = scene_success_rate >= 0.95 and covariance_rate >= 0.999
    report = [
        "# AV2 small-scene Level B report",
        "",
        f"- Result: {'PASS' if passed else 'FAIL'}",
        "- Condition: nominal only",
        f"- Scenes: {len(metrics)}",
        f"- Scene success rate: {scene_success_rate:.1%}",
        f"- Finite EKF output rate: {sum(bool(row['finite_ekf']) for row in metrics)/len(metrics):.1%}",
        f"- Positive-definite covariance rate: {covariance_rate:.6%}",
        f"- Posterior sources non-identical: {all(bool(row['posteriors_not_identical']) for row in metrics)}",
        f"- Evidence residual variation: {all(bool(row['evidence_residual_has_variation']) for row in metrics)}",
        "",
        "This Level B run uses only official-train focal VEHICLE scenes and does not inject faults.",
        "",
    ]
    (report_dir / "level_b_report.md").write_text("\n".join(report), encoding="utf-8")
    if not passed:
        raise RuntimeError("Level B acceptance checks failed; inspect level_b_metrics.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run nominal AV2 small-scene Level B.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--measurement-seed", type=int, default=100)
    args = parser.parse_args()
    run(args.root.resolve(), count=args.count, measurement_seed=args.measurement_seed)


if __name__ == "__main__":
    main()
