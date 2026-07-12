"""Level B+ B3: nominal evidence discriminability diagnostics.

This is deliberately a diagnostic-only program: it replays the frozen Level B
scenes with nominal measurements, never trains a model, and scores only frames
after the timestamp-derived one-second warmup.  Metrics are computed inside a
scene/measurement-seed run before any across-scene summaries are formed.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.av2.feature_builder import build_av2_feature_arrays
from configs.av2_level_b_plus_v11 import AGGRESSIVE_PROTOCOL, LEVEL_B_PLUS_PROTOCOL_VERSION, MEASUREMENT_SEEDS
from scripts.av2_small_level_b import _load_local_truth
from simulation.av2_sensor_ekf import run_av2_sensor_ekfs


PROTOCOL = LEVEL_B_PLUS_PROTOCOL_VERSION
SOURCE_NAMES = ("T1", "T2", "T3")
EVIDENCE_NAMES = ("E1", "E2")
PAIRS = ((0, 1), (0, 2), (1, 2))
EPS = 1e-12


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _pairwise_concordance(score: np.ndarray, error: np.ndarray) -> tuple[float, int, int]:
    """Return ordering agreement across the three source pairs.

    Exact ties in either quantity carry no ordering information and are not
    included in the denominator.
    """
    concordant = 0
    valid = 0
    for left, right in PAIRS:
        score_delta = score[:, left] - score[:, right]
        error_delta = error[:, left] - error[:, right]
        keep = (
            np.isfinite(score_delta)
            & np.isfinite(error_delta)
            & (np.abs(score_delta) > EPS)
            & (np.abs(error_delta) > EPS)
        )
        valid += int(keep.sum())
        concordant += int(np.count_nonzero(score_delta[keep] * error_delta[keep] > 0.0))
    return (float(concordant / valid) if valid else float("nan"), concordant, valid)


def _worst_accuracy(score: np.ndarray, error: np.ndarray) -> tuple[float, np.ndarray, int]:
    finite = np.isfinite(score).all(axis=1) & np.isfinite(error).all(axis=1)
    predicted = np.argmax(score[finite], axis=1)
    actual = np.argmax(error[finite], axis=1)
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (actual, predicted), 1)
    count = int(finite.sum())
    accuracy = float(np.trace(confusion) / count) if count else float("nan")
    return accuracy, confusion, count


def _standardize_evidence(residual: np.ndarray) -> np.ndarray:
    """Standardize each evidence stream within one scene/seed run.

    ``residual`` is [time, evidence, posterior].  Each evidence source gets
    one mean and scale across its post-warmup time/posterior values, retaining
    every evidence-posterior pair rather than broadcasting a pooled score.
    """
    standardized = np.empty_like(residual, dtype=np.float64)
    for evidence in range(residual.shape[1]):
        values = residual[:, evidence, :]
        finite = values[np.isfinite(values)]
        if not finite.size:
            raise ValueError(f"evidence index {evidence} has no valid post-warmup residuals")
        mean = float(np.mean(finite))
        scale = float(np.std(finite))
        standardized[:, evidence, :] = (values - mean) / (scale + EPS)
    return standardized


def _summary_rows(
    rows: list[dict[str, Any]],
    group_keys: tuple[str, ...],
    metrics: tuple[str, ...],
    protocol_name: str = PROTOCOL,
) -> list[dict[str, Any]]:
    """Average measurement seeds per scene, then summarize the 20 scenes."""
    scene_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[group_key] for group_key in group_keys) + (row["scenario_id"],)
        scene_groups.setdefault(key, []).append(row)
    scene_rows: list[dict[str, Any]] = []
    for key, group_rows in scene_groups.items():
        scene_row = {group_key: value for group_key, value in zip(group_keys, key[:-1])}
        scene_row["scenario_id"] = key[-1]
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group_rows], dtype=np.float64)
            values = values[np.isfinite(values)]
            scene_row[metric] = float(np.mean(values)) if values.size else float("nan")
        scene_rows.append(scene_row)

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in scene_rows:
        groups.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    output: list[dict[str, Any]] = []
    for group, group_rows in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        base = {key: value for key, value in zip(group_keys, group)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group_rows], dtype=np.float64)
            values = values[np.isfinite(values)]
            if not values.size:
                continue
            output.append(
                {
                    "protocol": protocol_name,
                    "row_type": "across_scene_summary",
                    **base,
                    "metric": metric,
                    "scene_count": int(values.size),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "p25": float(np.percentile(values, 25)),
                    "p75": float(np.percentile(values, 75)),
                }
            )
    return output


def _ecdf(ax: Any, values: Iterable[float], label: str) -> None:
    array = np.sort(np.asarray(list(values), dtype=np.float64))
    array = array[np.isfinite(array)]
    if array.size:
        ax.step(array, np.arange(1, array.size + 1) / array.size, where="post", label=label)


def _scene_means(rows: list[dict[str, Any]], metric: str, **selectors: Any) -> list[float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if all(row.get(key) == value for key, value in selectors.items()):
            value = float(row[metric])
            if np.isfinite(value):
                grouped.setdefault(str(row["scenario_id"]), []).append(value)
    return [float(np.mean(values)) for _, values in sorted(grouped.items())]


def run(
    root: Path,
    count: int = 20,
    measurement_seeds: tuple[int, ...] = (100, 101, 102),
    *,
    output_metadata_dir: Path | None = None,
    output_plot_dir: Path | None = None,
    output_log_dir: Path | None = None,
    protocol_name: str = PROTOCOL,
) -> dict[str, Any]:
    manifest_path = root / "manifests" / "level_b_20.csv"
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        manifest = sorted(csv.DictReader(handle), key=lambda row: row["scenario_id"])
    if count != 20:
        raise ValueError(f"{protocol_name} freezes scene count to 20, got {count}")
    if tuple(measurement_seeds) != MEASUREMENT_SEEDS:
        raise ValueError(f"{protocol_name} freezes measurement seeds to {list(MEASUREMENT_SEEDS)}")
    if len(manifest) != count:
        raise RuntimeError(f"Frozen Level B manifest must contain exactly {count} scenes, found {len(manifest)}")
    selected = manifest[:count]
    metadata_dir = output_metadata_dir or root / "metadata" / "level_b_plus"
    plot_dir = output_plot_dir or root / "plots" / "level_b_plus"
    log_dir = output_log_dir or root / "logs" / "level_b_plus"
    for directory in (metadata_dir, plot_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    residual_rows: list[dict[str, Any]] = []
    concordance_rows: list[dict[str, Any]] = []
    accuracy_rows: list[dict[str, Any]] = []
    plot_error: dict[str, list[np.ndarray]] = {"E1": [], "E2": [], "combined": []}
    plot_scores: dict[str, list[np.ndarray]] = {"E1": [], "E2": [], "combined": []}
    spread_values: dict[str, list[np.ndarray]] = {"E1": [], "E2": []}
    combined_confusion = np.zeros((3, 3), dtype=np.int64)

    for row in selected:
        scenario, local, target = _load_local_truth(Path(row["parquet_path"]))
        for seed in measurement_seeds:
            outputs = run_av2_sensor_ekfs(
                local.timestamps_ns,
                target,
                rng=np.random.default_rng(seed),
                protocol=AGGRESSIVE_PROTOCOL,
            )
            features = build_av2_feature_arrays(
                outputs,
                target,
                warmup_seconds=1.0,
                protocol=AGGRESSIVE_PROTOCOL,
            )
            eval_mask = np.asarray(features.eval_mask, dtype=bool)
            posterior = np.asarray(outputs.posterior_mean, dtype=np.float64)[eval_mask]
            position_error = np.linalg.norm(posterior[:, :, :2] - target[eval_mask, None, :2], axis=2)
            # Diagnostics store the whitened norm sqrt(r'R^-1r); B3's d is its square.
            whitened_norm = np.asarray(features.diagnostics.postfit_residual_norm, dtype=np.float64)
            residual = np.square(whitened_norm[eval_mask, 3:5, :])
            if not np.isfinite(position_error).all():
                raise RuntimeError(f"non-finite B3 inputs for {scenario.scenario_id}, seed {seed}")
            standardized = _standardize_evidence(residual)
            combined = np.mean(standardized, axis=1)

            for evidence_index, name in enumerate(EVIDENCE_NAMES):
                evidence_valid = np.isfinite(residual[:, evidence_index, :]).all(axis=1)
                scores = residual[evidence_valid, evidence_index, :]
                errors = position_error[evidence_valid]
                if not scores.size:
                    raise RuntimeError(f"no valid {name} residuals for {scenario.scenario_id}, seed {seed}")
                spread = np.std(scores, axis=1)
                relative = spread / (np.mean(scores, axis=1) + EPS)
                spread_values[name].append(relative)
                residual_rows.append(
                    {
                        "protocol": protocol_name,
                        "row_type": "scene_seed",
                        "scenario_id": scenario.scenario_id,
                        "measurement_seed": seed,
                        "evidence": name,
                        "post_warmup_frames": int(eval_mask.sum()),
                        "residual_definition": "squared_whitened_postfit_norm",
                        "residual_mean": float(np.mean(scores)),
                        "residual_std": float(np.std(scores)),
                        "spread_mean": float(np.mean(spread)),
                        "spread_median": float(np.median(spread)),
                        "relative_spread_mean": float(np.mean(relative)),
                        "relative_spread_median": float(np.median(relative)),
                        "relative_spread_p25": float(np.percentile(relative, 25)),
                        "relative_spread_p75": float(np.percentile(relative, 75)),
                    }
                )
                concordance, concordant, valid_pairs = _pairwise_concordance(scores, errors)
                accuracy, _, valid_frames = _worst_accuracy(scores, errors)
                plot_scores[name].append(scores.reshape(-1))
                plot_error[name].append(errors.reshape(-1))
                concordance_rows.append(
                    {
                        "protocol": protocol_name,
                        "row_type": "scene_seed",
                        "scenario_id": scenario.scenario_id,
                        "measurement_seed": seed,
                        "score": name,
                        "pairwise_concordance": concordance,
                        "concordant_pairs": concordant,
                        "valid_pairs": valid_pairs,
                        "spearman_position_error": float(spearmanr(scores.reshape(-1), errors.reshape(-1)).statistic),
                    }
                )
                accuracy_rows.append(
                    {
                        "protocol": protocol_name,
                        "row_type": "scene_seed",
                        "scenario_id": scenario.scenario_id,
                        "measurement_seed": seed,
                        "score": name,
                        "worst_source_accuracy": accuracy,
                        "valid_frames": valid_frames,
                    }
                )

            combined_valid = np.isfinite(combined).all(axis=1)
            combined_scores = combined[combined_valid]
            combined_errors = position_error[combined_valid]
            if not combined_scores.size:
                raise RuntimeError(f"no jointly valid evidence frames for {scenario.scenario_id}, seed {seed}")
            concordance, concordant, valid_pairs = _pairwise_concordance(combined_scores, combined_errors)
            accuracy, confusion, valid_frames = _worst_accuracy(combined_scores, combined_errors)
            combined_confusion += confusion
            plot_scores["combined"].append(combined_scores.reshape(-1))
            plot_error["combined"].append(combined_errors.reshape(-1))
            concordance_rows.append(
                {
                    "protocol": protocol_name,
                    "row_type": "scene_seed",
                    "scenario_id": scenario.scenario_id,
                    "measurement_seed": seed,
                    "score": "combined",
                    "pairwise_concordance": concordance,
                    "concordant_pairs": concordant,
                    "valid_pairs": valid_pairs,
                    "spearman_position_error": float(spearmanr(combined_scores.reshape(-1), combined_errors.reshape(-1)).statistic),
                }
            )
            accuracy_rows.append(
                {
                    "protocol": protocol_name,
                    "row_type": "scene_seed",
                    "scenario_id": scenario.scenario_id,
                    "measurement_seed": seed,
                    "score": "combined",
                    "worst_source_accuracy": accuracy,
                    "valid_frames": valid_frames,
                }
            )

    _write_csv(metadata_dir / "evidence_residual_statistics.csv", residual_rows)
    _write_csv(metadata_dir / "evidence_concordance.csv", concordance_rows)
    _write_csv(metadata_dir / "evidence_worst_source_accuracy.csv", accuracy_rows)
    summary_rows = []
    summary_rows += _summary_rows(
        residual_rows,
        ("evidence",),
        ("residual_mean", "spread_mean", "spread_median", "relative_spread_mean", "relative_spread_median"),
        protocol_name,
    )
    summary_rows += _summary_rows(
        concordance_rows, ("score",), ("pairwise_concordance", "spearman_position_error"), protocol_name
    )
    summary_rows += _summary_rows(accuracy_rows, ("score",), ("worst_source_accuracy",), protocol_name)
    _write_csv(metadata_dir / "evidence_b3_across_scene_summary.csv", summary_rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, name in zip(axes, ("E1", "E2", "combined")):
        errors = np.concatenate(plot_error[name])
        scores = np.concatenate(plot_scores[name])
        ax.hexbin(errors, scores, gridsize=45, bins="log", mincnt=1, cmap="viridis")
        ax.set_xlabel("Posterior position error (m)")
        ax.set_ylabel(f"{name} evidence score")
        ax.set_title(name)
    fig.suptitle("Nominal evidence score vs posterior error (post-warmup)")
    fig.tight_layout()
    fig.savefig(plot_dir / "residual_vs_posterior_error.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name in EVIDENCE_NAMES:
        _ecdf(ax, np.concatenate(spread_values[name]), name)
    ax.set_xlabel("Relative residual spread across T1/T2/T3")
    ax.set_ylabel("ECDF")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "residual_spread_ecdf.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    row_sum = combined_confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(combined_confusion, row_sum, out=np.zeros_like(combined_confusion, dtype=float), where=row_sum > 0)
    image = ax.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
    for actual in range(3):
        for predicted in range(3):
            ax.text(predicted, actual, f"{normalized[actual, predicted]:.2f}\n(n={combined_confusion[actual, predicted]})", ha="center", va="center")
    ax.set_xticks(range(3), SOURCE_NAMES)
    ax.set_yticks(range(3), SOURCE_NAMES)
    ax.set_xlabel("Predicted worst source")
    ax.set_ylabel("Actual worst source")
    ax.set_title("Combined evidence worst-source confusion")
    fig.colorbar(image, ax=ax, label="Row-normalized rate")
    fig.tight_layout()
    fig.savefig(plot_dir / "evidence_confusion_matrix.pdf")
    plt.close(fig)

    combined_concordance = _scene_means(
        concordance_rows, "pairwise_concordance", score="combined"
    )
    combined_accuracy = _scene_means(
        accuracy_rows, "worst_source_accuracy", score="combined"
    )
    relative_spread = _scene_means(residual_rows, "relative_spread_median")
    result = {
        "protocol": protocol_name,
        "condition": "nominal",
        "scenes": count,
        "measurement_seeds": list(measurement_seeds),
        "scene_seed_runs": count * len(measurement_seeds),
        "combined_pairwise_concordance_median": float(np.median(combined_concordance)),
        "combined_worst_source_accuracy_median": float(np.median(combined_accuracy)),
        "median_relative_residual_spread": float(np.median(relative_spread)),
    }
    result["pass"] = bool(
        result["combined_pairwise_concordance_median"] >= 0.55
        and result["combined_worst_source_accuracy_median"] >= 0.40
        and result["median_relative_residual_spread"] > 0.0
    )
    (log_dir / "b3_completed.txt").write_text("\n".join(f"{k}={v}" for k, v in result.items()) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AV2 Level B+ B3 evidence discriminability diagnostics.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--measurement-seeds", type=int, nargs="+", default=[100, 101, 102])
    args = parser.parse_args()
    result = run(args.root.resolve(), count=args.count, measurement_seeds=tuple(args.measurement_seeds))
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
