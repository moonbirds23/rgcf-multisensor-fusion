"""Generate five aligned AV2 method profiles for the paper.

Each panel combines a smoothed time-wise position RMSE curve (left axis) with
the unsmoothed inter-scene IQR of instantaneous position error (right axis).
The five panels share the time coordinate but use tight method-specific RMSE
limits so temporal variation remains legible.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from paper_figure_style import COLORS, apply_paper_style, method_style, save_figure


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
METHODS = [
    "full_pefnet",
    "covariance_intersection",
    "posterior_only",
    "pefnet_no_external_evidence",
    "t1",
]
METHOD_TITLES = {
    "full_pefnet": "Full PEFNet",
    "covariance_intersection": "CI",
    "posterior_only": "Posterior-only",
    "pefnet_no_external_evidence": "No external evidence",
    "t1": "T1",
}
RIGHT_AXIS_MAX = 3.0
BAR_ALPHA = 0.46
LEFT_AXIS_MIN = 1.6
LEFT_AXIS_MAX = 6.4
LEFT_AXIS_TICKS = [1.6, 2.4, 3.2, 4.0, 4.8, 5.6, 6.4]
ANNOTATION_TIMES = [2.0, 5.0, 8.0]


def _find_timestep_data() -> Path:
    matches = [path for path in WORKSPACE_ROOT.rglob("timestep_scene_level.csv") if path.is_file()]
    formal_matches = [
        path
        for path in matches
        if (path.parent / "AV2_1000_FORMAL_RESULTS_20260714.zip").is_file()
    ]
    if len(formal_matches) == 1:
        return formal_matches[0]
    if len(matches) != 1:
        raise FileNotFoundError(
            "Expected one timestep_scene_level.csv under {}, found {}".format(
                WORKSPACE_ROOT, len(matches)
            )
        )
    return matches[0]


def _local_polynomial_smooth(
    values: Iterable[float], window: int = 9, polynomial_order: int = 2
) -> np.ndarray:
    """Smooth a one-dimensional series with edge-aware local polynomial fits."""
    values_array = np.asarray(list(values), dtype=float)
    if window % 2 != 1 or window < 5:
        raise ValueError("The smoothing window must be an odd integer of at least five.")
    if window > len(values_array):
        raise ValueError("The smoothing window cannot exceed the series length.")
    if polynomial_order < 1 or polynomial_order >= window:
        raise ValueError("The polynomial order must be between one and window - 1.")

    half = window // 2
    smoothed = np.empty_like(values_array)
    for index in range(len(values_array)):
        start = max(0, index - half)
        stop = min(len(values_array), index + half + 1)
        if stop - start < window:
            if start == 0:
                stop = window
            else:
                start = len(values_array) - window
        offsets = np.arange(start, stop, dtype=float) - float(index)
        coefficients = np.polyfit(offsets, values_array[start:stop], polynomial_order)
        smoothed[index] = np.polyval(coefficients, 0.0)
    return smoothed


def compute_profiles(
    frame: pd.DataFrame, smooth_window: int = 9
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-step RMSE and inter-scene IQR for all five methods."""
    records = []
    summaries = []
    common_times = None
    common_scenarios = None

    for method in METHODS:
        subset = frame[frame["method"] == method]
        pivot = subset.pivot(
            index="scenario_id", columns="eval_step", values="position_squared_error"
        ).sort_index(axis=0).sort_index(axis=1)
        if pivot.shape != (200, 100) or pivot.isna().any().any():
            raise RuntimeError("Incomplete timestep coverage for {}: {}".format(method, pivot.shape))

        times = (
            subset.groupby("eval_step")["time_seconds"]
            .mean()
            .reindex(pivot.columns)
            .to_numpy(float)
        )
        if common_times is None:
            common_times = times
            common_scenarios = pivot.index
        elif not np.allclose(times, common_times, rtol=0.0, atol=1e-12):
            raise RuntimeError("Methods do not share identical time coordinates.")
        elif not pivot.index.equals(common_scenarios):
            raise RuntimeError("Methods do not share identical paired scenes.")

        squared_error = pivot.to_numpy(float)
        scene_error = np.sqrt(np.clip(squared_error, 0.0, None))
        rmse = np.sqrt(squared_error.mean(axis=0))
        rmse_smooth = _local_polynomial_smooth(rmse, window=smooth_window)
        q25 = np.quantile(scene_error, 0.25, axis=0)
        q75 = np.quantile(scene_error, 0.75, axis=0)
        iqr = q75 - q25

        for step, time_seconds, raw_value, smooth_value, iqr_value in zip(
            pivot.columns, times, rmse, rmse_smooth, iqr
        ):
            records.append(
                {
                    "method": method,
                    "eval_step": int(step),
                    "time_seconds": float(time_seconds),
                    "position_rmse": float(raw_value),
                    "position_rmse_smooth": float(smooth_value),
                    "position_error_iqr": float(iqr_value),
                    "scenarios": int(len(pivot)),
                    "smooth_window": int(smooth_window),
                    "smooth_polynomial_order": 2,
                }
            )
        summaries.append(
            {
                "method": method,
                "display_name": METHOD_TITLES[method],
                "mean_position_rmse": float(rmse.mean()),
                "mean_position_error_iqr": float(iqr.mean()),
                "min_smoothed_rmse": float(rmse_smooth.min()),
                "max_smoothed_rmse": float(rmse_smooth.max()),
            }
        )

    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(summaries)


def _style_box_axes(axis: plt.Axes, *, right: bool = False) -> None:
    axis.set_facecolor("none")
    axis.grid(False)
    axis.tick_params(direction="out", pad=2.0)
    if right:
        axis.spines["right"].set_visible(True)
        axis.spines["right"].set_color(COLORS["axis"])
        axis.spines["right"].set_linewidth(0.62)
        for side in ("left", "top", "bottom"):
            axis.spines[side].set_visible(False)
    else:
        for side in ("left", "right", "top", "bottom"):
            axis.spines[side].set_visible(True)
            axis.spines[side].set_color(COLORS["axis"])
            axis.spines[side].set_linewidth(0.62)
        axis.grid(
            True,
            axis="both",
            color=COLORS["grid"],
            linewidth=0.44,
            alpha=0.76,
            linestyle=(0, (2.0, 2.4)),
            zorder=0,
        )


def _plot_profile(
    axis: plt.Axes,
    profile: pd.DataFrame,
    summary: pd.Series,
    method: str,
    *,
    show_legend: bool,
) -> None:
    color = str(method_style(method)["color"])
    times = profile["time_seconds"].to_numpy(float)
    smoothed = profile["position_rmse_smooth"].to_numpy(float)
    iqr = profile["position_error_iqr"].to_numpy(float)
    twin = axis.twinx()

    time_step = float(np.median(np.diff(times)))
    twin.bar(
        times,
        iqr,
        width=time_step * 0.76,
        color=color,
        edgecolor=color,
        linewidth=0.20,
        alpha=BAR_ALPHA,
        zorder=1,
        align="center",
    )
    twin.set_ylim(0.0, RIGHT_AXIS_MAX)
    twin.set_yticks([0.0, 1.0, 2.0, 3.0])
    twin.set_ylabel("IQR (m)", labelpad=5.0)
    _style_box_axes(twin, right=True)

    axis.plot(
        times,
        smoothed,
        color=color,
        linewidth=1.25 if method == "full_pefnet" else 1.08,
        linestyle="-",
        zorder=5,
    )
    axis.set_xlim(float(times.min()), float(times.max()))
    axis.set_ylim(LEFT_AXIS_MIN, LEFT_AXIS_MAX)
    axis.set_yticks(LEFT_AXIS_TICKS)
    axis.set_ylabel("RMSE (m)", labelpad=4.5)
    _style_box_axes(axis)
    axis.set_zorder(2)
    twin.set_zorder(1)
    axis.patch.set_visible(False)
    twin.patch.set_visible(False)

    for annotation_time in ANNOTATION_TIMES:
        time_index = int(np.argmin(np.abs(times - annotation_time)))
        time_value = float(times[time_index])
        rmse_value = float(smoothed[time_index])
        axis.scatter(
            [time_value],
            [rmse_value],
            s=10.0,
            marker="o",
            facecolors="white",
            edgecolors=color,
            linewidths=0.72,
            zorder=7,
        )
        axis.annotate(
            "{:.2f}".format(rmse_value),
            xy=(time_value, rmse_value),
            xytext=(0.0, 5.0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.0,
            color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.45},
            zorder=8,
        )

    text_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.8}
    axis.text(
        0.015,
        0.94,
        METHOD_TITLES[method],
        transform=axis.transAxes,
        ha="left",
        va="top",
        color=color,
        fontsize=8.2,
        fontweight="bold",
        bbox=text_box,
        zorder=8,
    )
    axis.text(
        0.985,
        0.94,
        "Mean RMSE: {:.2f} m\nMean IQR: {:.2f} m".format(
            float(summary["mean_position_rmse"]),
            float(summary["mean_position_error_iqr"]),
        ),
        transform=axis.transAxes,
        ha="right",
        va="top",
        color=color,
        fontsize=6.6,
        linespacing=1.25,
        bbox=text_box,
        zorder=8,
    )

    if show_legend:
        line = Line2D([0], [0], color=color, linewidth=1.25, linestyle="-")
        bars = Patch(facecolor=color, edgecolor=color, linewidth=0.25, alpha=BAR_ALPHA)
        legend = axis.legend(
            [line, bars],
            ["Smoothed RMSE", "Inter-scene error IQR"],
            loc="upper center",
            bbox_to_anchor=(0.52, 0.995),
            ncol=2,
            borderaxespad=0.0,
            frameon=True,
            fancybox=False,
            framealpha=0.82,
            facecolor="white",
            edgecolor="none",
            fontsize=6.5,
            handlelength=2.0,
            columnspacing=0.9,
            handletextpad=0.4,
        )
        legend.set_zorder(9)


def _write_summary(output_dir: Path, stem: str, summaries: pd.DataFrame) -> None:
    lines = [
        "# AV2 Method Profile Figure Summary",
        "",
        "- Left axis: position RMSE over 200 fixed AV2 test scenes.",
        "- Right axis: unsmoothed inter-scene IQR of instantaneous position error.",
        "- RMSE display smoothing: second-order local polynomial fit over a nine-step window.",
        "- All panels use the same 1.6--6.4 m RMSE scale with the 0.8 m PEFNet tick interval.",
        "- Smoothed RMSE values are annotated at the shared 2.0 s, 5.0 s, and 8.0 s timestamps.",
        "- All panels share the 0.0--9.9 s time coordinate and a common 0--3 m IQR scale.",
        "",
        "| Method | Mean RMSE (m) | Mean IQR (m) | Smoothed RMSE range (m) |",
        "|---|---:|---:|---:|",
    ]
    for _, row in summaries.iterrows():
        lines.append(
            "| {} | {:.3f} | {:.3f} | {:.3f}--{:.3f} |".format(
                row["display_name"],
                row["mean_position_rmse"],
                row["mean_position_error_iqr"],
                row["min_smoothed_rmse"],
                row["max_smoothed_rmse"],
            )
        )
    lines.append("")
    (output_dir / "{}_summary.md".format(stem)).write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True, help="PDF path; SVG/PNG siblings are written.")
    parser.add_argument("--statistics-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--smooth-window", type=int, default=9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data or _find_timestep_data()
    frame = pd.read_csv(data_path)
    required = {"method", "scenario_id", "eval_step", "time_seconds", "position_squared_error"}
    if not required.issubset(frame.columns):
        raise RuntimeError("Missing timestep columns: {}".format(sorted(required - set(frame.columns))))
    if set(frame["method"]) != set(METHODS) or len(frame) != 5 * 200 * 100:
        raise RuntimeError("Input must contain five methods x 200 scenes x 100 evaluation steps.")

    profiles, summaries = compute_profiles(frame, smooth_window=args.smooth_window)
    statistics_output = args.statistics_output or args.output.with_name(
        "AV2_1000_method_profiles_statistics.csv"
    )
    summary_output = args.summary_output or args.output.with_name(
        "AV2_1000_method_profiles_method_summary.csv"
    )
    statistics_output.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(statistics_output, index=False)
    summaries.to_csv(summary_output, index=False)

    apply_paper_style()
    fig, axes = plt.subplots(5, 1, figsize=(7.08, 7.15), sharex=True)
    summary_by_method: Dict[str, pd.Series] = {
        row["method"]: row for _, row in summaries.iterrows()
    }
    for index, (axis, method) in enumerate(zip(axes, METHODS)):
        profile = profiles[profiles["method"] == method].sort_values("eval_step")
        _plot_profile(
            axis,
            profile,
            summary_by_method[method],
            method,
            show_legend=index == 0,
        )
        if index < len(axes) - 1:
            axis.tick_params(labelbottom=False)

    ticks = [0.0, 2.0, 4.0, 6.0, 8.0, 9.9]
    axes[-1].set_xticks(ticks)
    axes[-1].set_xlabel("Time after evaluation start (s)", labelpad=4.5)
    fig.subplots_adjust(left=0.085, right=0.915, top=0.992, bottom=0.075, hspace=0.075)
    save_figure(fig, args.output.parent, args.output.stem, pad_inches=0.018)
    _write_summary(args.output.parent, args.output.stem, summaries)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
