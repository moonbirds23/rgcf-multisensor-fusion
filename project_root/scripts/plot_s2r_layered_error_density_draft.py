"""Plot a compact four-layer longitudinal/lateral error-density figure.

The source is the frozen representative S2R trajectory used by the paper.  The
global position residual at each timestamp is projected onto the local tangent
and left-normal directions of the ground-truth trajectory. Gaussian probability
ellipses are mean-centered at the shared zero-error axis for covariance-only
comparison. Four algorithms are shown; Track-1 is intentionally excluded.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd

from paper_figure_style import COLORS, apply_paper_style, save_figure


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    WORKSPACE_ROOT
    / "workshop_paper"
    / "05_已完成实验结果"
    / "数据"
    / "第5章时间序列与轨迹"
    / "trajectory_exports"
    / "s2r_representative_trajectory.csv"
)

METHODS = [
    ("Mean Fusion", "Mean", "#CCBB44", 0.00),
    ("Covariance Intersection", "CI", "#228833", 0.78),
    ("Covariance-Weighted Moment Matching", "Cov.-weighted", "#4477AA", 1.56),
    ("PEFNet", "PEFNet", "#EE6677", 2.34),
]


def _require_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise RuntimeError("Trajectory input is missing columns: {}".format(", ".join(missing)))


def compute_directional_errors(frame: pd.DataFrame) -> pd.DataFrame:
    """Project prediction residuals into the truth trajectory's local frame."""
    _require_columns(frame, ["method", "sim_seed", "k", "t", "x", "y"])
    truth = frame[frame["method"] == "Ground Truth"].sort_values("k").reset_index(drop=True)
    if len(truth) < 3 or truth["k"].duplicated().any():
        raise RuntimeError("Ground-truth trajectory must contain unique ordered timestamps.")

    time = truth["t"].to_numpy(float)
    truth_xy = truth[["x", "y"]].to_numpy(float)
    velocity = np.gradient(truth_xy, time, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    if np.any(speed < 1e-9):
        raise RuntimeError("Cannot define the local trajectory frame at zero-speed timestamps.")
    tangent = velocity / speed[:, None]
    left_normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))

    records = []
    for source_name, display_name, _, layer in METHODS:
        prediction = frame[frame["method"] == source_name].sort_values("k").reset_index(drop=True)
        if len(prediction) != len(truth) or not np.array_equal(
            prediction["k"].to_numpy(), truth["k"].to_numpy()
        ):
            raise RuntimeError("{} does not align with ground truth.".format(source_name))
        residual = prediction[["x", "y"]].to_numpy(float) - truth_xy
        longitudinal = np.sum(residual * tangent, axis=1)
        lateral = np.sum(residual * left_normal, axis=1)
        radial = np.linalg.norm(residual, axis=1)
        for index in range(len(truth)):
            records.append(
                {
                    "method": display_name,
                    "source_method": source_name,
                    "layer": float(layer),
                    "sim_seed": int(prediction.loc[index, "sim_seed"]),
                    "k": int(truth.loc[index, "k"]),
                    "time_seconds": float(time[index]),
                    "longitudinal_error_m": float(longitudinal[index]),
                    "lateral_error_m": float(lateral[index]),
                    "position_error_m": float(radial[index]),
                }
            )
    return pd.DataFrame.from_records(records)


def _gaussian_kernel(sigma_bins: float) -> np.ndarray:
    sigma = max(float(sigma_bins), 0.75)
    radius = max(2, int(np.ceil(4.0 * sigma)))
    coordinates = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (coordinates / sigma) ** 2)
    return kernel / kernel.sum()


def _smooth_histogram(histogram: np.ndarray, sigma_x: float, sigma_y: float) -> np.ndarray:
    kernel_x = _gaussian_kernel(sigma_x)
    kernel_y = _gaussian_kernel(sigma_y)
    smoothed = np.apply_along_axis(
        lambda values: np.convolve(values, kernel_x, mode="same"), 0, histogram
    )
    return np.apply_along_axis(
        lambda values: np.convolve(values, kernel_y, mode="same"), 1, smoothed
    )


def estimate_density(
    longitudinal: np.ndarray,
    lateral: np.ndarray,
    x_limits: Tuple[float, float],
    y_limits: Tuple[float, float],
    bins: int = 150,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Estimate a 2D KDE with a Gaussian-smoothed histogram."""
    histogram, x_edges, y_edges = np.histogram2d(
        longitudinal,
        lateral,
        bins=(bins, bins),
        range=(x_limits, y_limits),
    )
    dx = float(x_edges[1] - x_edges[0])
    dy = float(y_edges[1] - y_edges[0])
    sample_count = max(len(longitudinal), 2)
    scott = sample_count ** (-1.0 / 6.0)
    sigma_x = max(scott * float(np.std(longitudinal, ddof=1)) / dx, 1.0)
    sigma_y = max(scott * float(np.std(lateral, ddof=1)) / dy, 1.0)
    density = _smooth_histogram(histogram, sigma_x, sigma_y)
    density /= max(float(density.sum()) * dx * dy, 1e-15)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    grid_x, grid_y = np.meshgrid(x_centers, y_centers, indexing="ij")
    return grid_x, grid_y, density, dx, dy


def _hdr_threshold(density: np.ndarray, dx: float, dy: float, probability: float) -> float:
    flat = density.ravel()
    order = np.argsort(flat)[::-1]
    cumulative = np.cumsum(flat[order]) * dx * dy
    index = min(int(np.searchsorted(cumulative, probability, side="left")), len(order) - 1)
    return float(flat[order[index]])


def _common_limits(errors: pd.DataFrame) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    # Keep a shared robust range for direct comparison while preventing a few
    # tail samples from creating oversized, mostly empty planes.
    x_extent = float(np.quantile(np.abs(errors["longitudinal_error_m"]), 0.99)) * 1.04
    y_extent = float(np.quantile(np.abs(errors["lateral_error_m"]), 0.99)) * 1.04
    x_extent = max(np.ceil(x_extent * 4.0) / 4.0, 0.75)
    y_extent = max(np.ceil(y_extent * 4.0) / 4.0, 0.75)
    return (-x_extent, x_extent), (-y_extent, y_extent)


def _add_plane(axis, x_limits, y_limits, layer: float) -> None:
    vertices = [
        (x_limits[0], y_limits[0], layer),
        (x_limits[1], y_limits[0], layer),
        (x_limits[1], y_limits[1], layer),
        (x_limits[0], y_limits[1], layer),
    ]
    plane = Poly3DCollection(
        [vertices],
        facecolor=(0.95, 0.96, 0.97, 0.065),
        edgecolor=mpl_colors.to_rgba(COLORS["axis"], 0.70),
        linewidth=0.55,
    )
    axis.add_collection3d(plane)
    axis.plot(
        [x_limits[0], x_limits[1]],
        [0.0, 0.0],
        [layer, layer],
        color=COLORS["reference"],
        linewidth=0.42,
        linestyle=(0, (2.0, 2.5)),
        alpha=0.45,
        zorder=1,
    )
    axis.plot(
        [0.0, 0.0],
        [y_limits[0], y_limits[1]],
        [layer, layer],
        color=COLORS["reference"],
        linewidth=0.42,
        linestyle=(0, (2.0, 2.5)),
        alpha=0.45,
        zorder=1,
    )


def _plane_limits_from_hdr(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    density: np.ndarray,
    threshold_90: float,
    global_x_limits: Tuple[float, float],
    global_y_limits: Tuple[float, float],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Tightly frame a layer around its 90% HDR while retaining the origin."""
    mask = density >= threshold_90
    if not np.any(mask):
        return global_x_limits, global_y_limits
    x_values = grid_x[mask]
    y_values = grid_y[mask]
    x_span = max(float(x_values.max() - x_values.min()), 0.25)
    y_span = max(float(y_values.max() - y_values.min()), 0.25)
    x_limits = (
        max(global_x_limits[0], min(float(x_values.min() - 0.08 * x_span), 0.0)),
        min(global_x_limits[1], max(float(x_values.max() + 0.08 * x_span), 0.0)),
    )
    y_limits = (
        max(global_y_limits[0], min(float(y_values.min() - 0.08 * y_span), 0.0)),
        min(global_y_limits[1], max(float(y_values.max() + 0.08 * y_span), 0.0)),
    )
    return x_limits, y_limits


def _gaussian_probability_ellipse(
    longitudinal: np.ndarray,
    lateral: np.ndarray,
    probability: float,
    points: int = 361,
    center_on_origin: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a fitted 2D-Gaussian probability ellipse and its moments."""
    values = np.column_stack((longitudinal, lateral))
    mean = values.mean(axis=0)
    covariance = np.cov(values, rowvar=False, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 1e-12)
    eigenvectors = eigenvectors[:, order]
    # For two degrees of freedom, chi2.ppf(p, 2) = -2 log(1-p).
    radius = np.sqrt(-2.0 * np.log(1.0 - probability))
    theta = np.linspace(0.0, 2.0 * np.pi, points)
    unit_circle = np.vstack((np.cos(theta), np.sin(theta)))
    transform = eigenvectors @ np.diag(np.sqrt(eigenvalues) * radius)
    plotted_center = np.zeros(2, dtype=float) if center_on_origin else mean
    ellipse = plotted_center[:, None] + transform @ unit_circle
    semi_axes = np.sqrt(eigenvalues) * radius
    return ellipse[0], ellipse[1], mean, covariance, semi_axes, eigenvectors


def _limits_from_ellipses(errors: pd.DataFrame) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    x_values = [0.0]
    y_values = [0.0]
    for _, display_name, _, _ in METHODS:
        part = errors[errors["method"] == display_name]
        ellipse_x, ellipse_y, *_ = _gaussian_probability_ellipse(
            part["longitudinal_error_m"].to_numpy(float),
            part["lateral_error_m"].to_numpy(float),
            0.90,
            center_on_origin=True,
        )
        x_values.extend(ellipse_x.tolist())
        y_values.extend(ellipse_y.tolist())
    x_extent = max(abs(float(np.min(x_values))), abs(float(np.max(x_values)))) * 1.06
    y_extent = max(abs(float(np.min(y_values))), abs(float(np.max(y_values)))) * 1.06
    x_extent = max(np.ceil(x_extent * 4.0) / 4.0, 0.75)
    y_extent = max(np.ceil(y_extent * 4.0) / 4.0, 0.75)
    return (-x_extent, x_extent), (-y_extent, y_extent)


def _plane_limits_from_ellipse(
    ellipse_x: np.ndarray,
    ellipse_y: np.ndarray,
    global_x_limits: Tuple[float, float],
    global_y_limits: Tuple[float, float],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    x_span = max(float(np.ptp(ellipse_x)), 0.25)
    y_span = max(float(np.ptp(ellipse_y)), 0.25)
    x_limits = (
        max(global_x_limits[0], min(float(ellipse_x.min() - 0.10 * x_span), 0.0)),
        min(global_x_limits[1], max(float(ellipse_x.max() + 0.10 * x_span), 0.0)),
    )
    y_limits = (
        max(global_y_limits[0], min(float(ellipse_y.min() - 0.10 * y_span), 0.0)),
        min(global_y_limits[1], max(float(ellipse_y.max() + 0.10 * y_span), 0.0)),
    )
    return x_limits, y_limits


def _add_filled_ellipse(axis, x: np.ndarray, y: np.ndarray, z: float, color, alpha: float) -> None:
    vertices = list(zip(x, y, np.full_like(x, z)))
    patch = Poly3DCollection(
        [vertices],
        facecolor=mpl_colors.to_rgba(color, alpha),
        edgecolor=mpl_colors.to_rgba(color, 0.98),
        linewidth=0.72,
        antialiased=True,
    )
    axis.add_collection3d(patch)


def draw_figure(errors: pd.DataFrame, output: Path) -> Dict[str, Path]:
    x_limits, y_limits = _limits_from_ellipses(errors)
    apply_paper_style()
    fig = plt.figure(figsize=(7.08, 4.55))
    axis = fig.add_subplot(111, projection="3d")
    axis.set_proj_type("persp", focal_length=0.95)

    for _, display_name, color, layer in METHODS:
        part = errors[errors["method"] == display_name]
        outer_x, outer_y, *_ = _gaussian_probability_ellipse(
            part["longitudinal_error_m"].to_numpy(float),
            part["lateral_error_m"].to_numpy(float),
            0.90,
            center_on_origin=True,
        )
        inner_x, inner_y, *_ = _gaussian_probability_ellipse(
            part["longitudinal_error_m"].to_numpy(float),
            part["lateral_error_m"].to_numpy(float),
            0.50,
            center_on_origin=True,
        )
        _add_plane(axis, x_limits, y_limits, layer)
        offset = layer + 0.018
        _add_filled_ellipse(axis, outer_x, outer_y, offset, color, 0.30)
        _add_filled_ellipse(axis, inner_x, inner_y, offset + 0.004, color, 0.66)
        axis.scatter(
            [0.0],
            [0.0],
            [offset + 0.008],
            marker="+",
            s=22.0,
            color=COLORS["text"],
            linewidths=0.75,
            depthshade=False,
            zorder=8,
        )

    top_layer = max(layer for _, _, _, layer in METHODS)
    # Internal coordinate axes intersect at (0, 0, 0).  The ellipses are
    # mean-centered for this view, so every layer shares the same zero-error
    # reference while covariance size and orientation remain unchanged.
    axis.plot(
        [x_limits[0], x_limits[1]],
        [0.0, 0.0],
        [0.0, 0.0],
        color=COLORS["text"],
        linewidth=0.85,
        alpha=0.92,
        zorder=10,
    )
    axis.plot(
        [0.0, 0.0],
        [y_limits[0], y_limits[1]],
        [0.0, 0.0],
        color=COLORS["text"],
        linewidth=0.85,
        alpha=0.92,
        zorder=10,
    )
    axis.plot(
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, top_layer + 0.12],
        color=COLORS["text"],
        linewidth=0.90,
        alpha=0.92,
        zorder=10,
    )
    axis.scatter(
        [0.0],
        [0.0],
        [0.0],
        marker="o",
        s=7.0,
        facecolor=COLORS["text"],
        edgecolor="none",
        depthshade=False,
        zorder=12,
    )
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.set_zlim(-0.06, top_layer + 0.16)
    axis.set_xlabel("Longitudinal position error (m)", labelpad=5.0)
    axis.set_ylabel("Lateral position error (m)", labelpad=7.0)
    axis.set_zlabel("")
    axis.set_zticks([])
    axis.tick_params(axis="x", pad=1.0)
    axis.tick_params(axis="y", pad=1.0)
    axis.view_init(elev=23.0, azim=-57.0)
    axis.set_box_aspect((1.48, 1.62, 1.16))
    axis.grid(False)
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    axis.xaxis._axinfo["grid"]["linewidth"] = 0.0
    axis.yaxis._axinfo["grid"]["linewidth"] = 0.0
    axis.zaxis._axinfo["grid"]["linewidth"] = 0.0
    axis.zaxis.line.set_color((0.0, 0.0, 0.0, 0.0))
    axis.zaxis.line.set_linewidth(0.0)
    method_handles = [
        Patch(
            facecolor=mpl_colors.to_rgba(color, 0.55),
            edgecolor=color,
            linewidth=0.6,
            label=display_name,
        )
        for _, display_name, color, _ in reversed(METHODS)
    ]
    region_handles = [
        Patch(facecolor="#687887", edgecolor="none", label="50% Gaussian region"),
        Patch(facecolor="#C4CDD5", edgecolor="none", label="90% Gaussian region"),
    ]
    axis.legend(
        handles=method_handles + region_handles,
        loc="upper left",
        bbox_to_anchor=(0.035, 0.915),
        borderaxespad=0.0,
        ncol=3,
        frameon=False,
        fontsize=6.35,
        handlelength=1.35,
        labelspacing=0.28,
        columnspacing=0.68,
    )
    fig.subplots_adjust(left=0.015, right=0.985, top=0.995, bottom=0.12)
    outputs = save_figure(fig, output.parent, output.stem, pad_inches=0.02)
    plt.close(fig)
    return outputs


def _write_summary(errors: pd.DataFrame, output: Path) -> None:
    rows = []
    for _, display_name, _, layer in METHODS:
        part = errors[errors["method"] == display_name]
        _, _, mean, covariance, axes_90, eigenvectors = _gaussian_probability_ellipse(
            part["longitudinal_error_m"].to_numpy(float),
            part["lateral_error_m"].to_numpy(float),
            0.90,
        )
        major_angle = float(np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])))
        rows.append(
            {
                "method": display_name,
                "layer": layer,
                "samples": len(part),
                "mean_longitudinal_error_m": part["longitudinal_error_m"].mean(),
                "std_longitudinal_error_m": part["longitudinal_error_m"].std(ddof=1),
                "mean_lateral_error_m": part["lateral_error_m"].mean(),
                "std_lateral_error_m": part["lateral_error_m"].std(ddof=1),
                "position_error_p95_m": part["position_error_m"].quantile(0.95),
                "gaussian_center_longitudinal_m": mean[0],
                "gaussian_center_lateral_m": mean[1],
                "gaussian_cov_long_long_m2": covariance[0, 0],
                "gaussian_cov_long_lat_m2": covariance[0, 1],
                "gaussian_cov_lat_lat_m2": covariance[1, 1],
                "gaussian_90_major_radius_m": axes_90[0],
                "gaussian_90_minor_radius_m": axes_90[1],
                "gaussian_major_axis_angle_deg": major_angle,
            }
        )
    pd.DataFrame(rows).to_csv(output.with_name(output.stem + "_summary.csv"), index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True, help="PDF path; SVG and PNG are also written.")
    parser.add_argument("--errors-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    errors = compute_directional_errors(frame)
    errors_output = args.errors_output or args.output.with_name(
        args.output.stem + "_directional_errors.csv"
    )
    errors_output.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(errors_output, index=False)
    _write_summary(errors, args.output)
    draw_figure(errors, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
