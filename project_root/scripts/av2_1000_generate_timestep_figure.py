"""Generate a two-panel AV2 time-wise RMSE and relative-increase figure."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_figure_style import (
    COLORS,
    DISPLAY_LABELS,
    apply_paper_style,
    method_style,
    save_figure,
    style_axes,
)


METHODS = [
    "full_pefnet",
    "covariance_intersection",
    "posterior_only",
    "pefnet_no_external_evidence",
    "t1",
]
COMPARISONS = [method for method in METHODS if method != "full_pefnet"]
DIFFERENCE_STYLES = {
    "covariance_intersection": {"color": COLORS["ci"], "linewidth": 1.35, "linestyle": (0, (5.0, 2.0)), "alpha": 1.0},
    "posterior_only": {"color": "#928995", "linewidth": 0.95, "linestyle": "-.", "alpha": 0.62},
    "pefnet_no_external_evidence": {"color": "#7B8794", "linewidth": 0.98, "linestyle": "--", "alpha": 0.66},
    "t1": {"color": "#4C566A", "linewidth": 1.00, "linestyle": (0, (6.0, 2.2)), "alpha": 0.68},
}


def _pivot_method(frame: pd.DataFrame, method: str) -> Tuple[pd.DataFrame, np.ndarray]:
    subset = frame[frame["method"] == method]
    pivot = subset.pivot(
        index="scenario_id", columns="eval_step", values="position_squared_error"
    ).sort_index(axis=0).sort_index(axis=1)
    if pivot.shape != (200, 100) or pivot.isna().any().any():
        raise RuntimeError("Incomplete timestep coverage for {}: {}".format(method, pivot.shape))
    times = subset.groupby("eval_step")["time_seconds"].mean().reindex(pivot.columns).to_numpy(float)
    return pivot, times


def bootstrap_statistics(
    frame: pd.DataFrame,
    repetitions: int = 4000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return absolute curves and paired method-minus-PEFNet differences."""
    rng = np.random.default_rng(20260713)
    values_by_method: Dict[str, np.ndarray] = {}
    columns = None
    times = None
    scenario_index = None
    for method in METHODS:
        pivot, method_times = _pivot_method(frame, method)
        if scenario_index is None:
            scenario_index = pivot.index
            columns = pivot.columns
            times = method_times
        elif not pivot.index.equals(scenario_index) or not pivot.columns.equals(columns):
            raise RuntimeError("Timestep methods do not share identical paired scenario/step coverage.")
        elif not np.allclose(method_times, times, rtol=0.0, atol=1e-12):
            raise RuntimeError("Timestep methods do not share identical time coordinates.")
        values_by_method[method] = pivot.to_numpy(float)

    assert columns is not None and times is not None
    absolute_samples: Dict[str, List[np.ndarray]] = {method: [] for method in METHODS}
    difference_samples: Dict[str, List[np.ndarray]] = {method: [] for method in COMPARISONS}
    n_scenes = len(scenario_index)

    for start in range(0, repetitions, 250):
        count = min(250, repetitions - start)
        indices = rng.integers(0, n_scenes, size=(count, n_scenes))
        bootstrap_batch = {
            method: np.sqrt(values[indices].mean(axis=1))
            for method, values in values_by_method.items()
        }
        for method in METHODS:
            absolute_samples[method].append(bootstrap_batch[method])
        pefnet_batch = bootstrap_batch["full_pefnet"]
        for method in COMPARISONS:
            difference_samples[method].append(bootstrap_batch[method] - pefnet_batch)

    curve_rows = []
    centers = {
        method: np.sqrt(values.mean(axis=0))
        for method, values in values_by_method.items()
    }
    for method in METHODS:
        bootstrap = np.concatenate(absolute_samples[method], axis=0)
        low, high = np.quantile(bootstrap, [0.025, 0.975], axis=0)
        for step, elapsed, center, lower, upper in zip(columns, times, centers[method], low, high):
            curve_rows.append(
                {
                    "method": method,
                    "eval_step": int(step),
                    "time_seconds": float(elapsed),
                    "position_rmse": float(center),
                    "ci95_low": float(lower),
                    "ci95_high": float(upper),
                    "scenarios": n_scenes,
                }
            )

    difference_rows = []
    pefnet_center = centers["full_pefnet"]
    for method in COMPARISONS:
        bootstrap = np.concatenate(difference_samples[method], axis=0)
        low, high = np.quantile(bootstrap, [0.025, 0.975], axis=0)
        center = centers[method] - pefnet_center
        for step, elapsed, value, lower, upper in zip(columns, times, center, low, high):
            difference_rows.append(
                {
                    "method": method,
                    "reference_method": "full_pefnet",
                    "eval_step": int(step),
                    "time_seconds": float(elapsed),
                    "rmse_increase": float(value),
                    "ci95_low": float(lower),
                    "ci95_high": float(upper),
                    "scenarios": n_scenes,
                }
            )
    return pd.DataFrame.from_records(curve_rows), pd.DataFrame.from_records(difference_rows)


def _plot_absolute(ax: plt.Axes, curve: pd.DataFrame) -> None:
    handles: Dict[str, object] = {}
    for method, alpha in (("full_pefnet", 0.14), ("covariance_intersection", 0.10)):
        part = curve[curve["method"] == method].sort_values("eval_step")
        ax.fill_between(
            part["time_seconds"].to_numpy(),
            part["ci95_low"].to_numpy(),
            part["ci95_high"].to_numpy(),
            color=method_style(method)["color"],
            alpha=alpha,
            linewidth=0.0,
            zorder=1,
        )

    pefnet = curve[curve["method"] == "full_pefnet"].sort_values("eval_step")
    (handles["full_pefnet"],) = ax.plot(
        pefnet["time_seconds"],
        pefnet["position_rmse"],
        color=COLORS["pefnet"],
        linewidth=2.05,
        linestyle="-",
        alpha=1.0,
        zorder=5,
        label=DISPLAY_LABELS["full_pefnet"],
    )
    ci = curve[curve["method"] == "covariance_intersection"].sort_values("eval_step")
    (handles["covariance_intersection"],) = ax.plot(
        ci["time_seconds"],
        ci["position_rmse"],
        color=COLORS["ci"],
        linewidth=1.38,
        linestyle=(0, (5.0, 2.0)),
        alpha=1.0,
        zorder=4,
        label=DISPLAY_LABELS["covariance_intersection"],
    )

    xmax = float(curve["time_seconds"].max())
    focus = curve[curve["method"].isin(["full_pefnet", "covariance_intersection"])]
    ymax = math.ceil(float(focus["ci95_high"].max()) * 1.08 * 2.0) / 2.0
    ax.set_xlim(0.0, xmax)
    ax.set_ylim(0.0, ymax)
    ax.set_ylabel("Position RMSE (m)")
    ax.set_title("(a) Absolute position RMSE", pad=3.5)
    ax.margins(x=0.0)
    style_axes(ax)
    ax.legend(
        [handles["full_pefnet"], handles["covariance_intersection"]],
        [DISPLAY_LABELS["full_pefnet"], DISPLAY_LABELS["covariance_intersection"]],
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        ncol=2,
        borderaxespad=0.0,
        frameon=False,
        columnspacing=0.85,
        handlelength=2.0,
    )


def _plot_differences(ax: plt.Axes, difference: pd.DataFrame) -> None:
    ci = difference[difference["method"] == "covariance_intersection"].sort_values("eval_step")
    ax.fill_between(
        ci["time_seconds"].to_numpy(),
        ci["ci95_low"].to_numpy(),
        ci["ci95_high"].to_numpy(),
        color=COLORS["ci"],
        alpha=0.12,
        linewidth=0.0,
        zorder=1,
    )

    handles: Dict[str, object] = {}
    for method in reversed(COMPARISONS):
        part = difference[difference["method"] == method].sort_values("eval_step")
        style = DIFFERENCE_STYLES[method]
        (line,) = ax.plot(
            part["time_seconds"],
            part["rmse_increase"],
            color=style["color"],
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            alpha=style["alpha"],
            label=DISPLAY_LABELS[method],
            zorder=5 if method == "covariance_intersection" else 3,
        )
        handles[method] = line

    lower = float(min(difference["ci95_low"].min(), difference["rmse_increase"].min(), 0.0))
    upper = float(max(difference["ci95_high"].max(), difference["rmse_increase"].max(), 0.0))
    padding = max((upper - lower) * 0.10, 0.18)
    ymin = math.floor((lower - padding) * 2.0) / 2.0
    ymax = math.ceil((upper + padding) * 2.0) / 2.0
    xmax = float(difference["time_seconds"].max())
    ax.set_xlim(0.0, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axhline(0.0, color=COLORS["axis"], linewidth=0.72, zorder=0)
    ax.set_xlabel("Time after evaluation start (s)")
    ax.set_ylabel(r"$\Delta$RMSE (m)")
    ax.set_title("(b) RMSE increase relative to PEFNet", pad=3.5)
    ax.margins(x=0.0)
    style_axes(ax)

    legend_order = [
        "covariance_intersection",
        "posterior_only",
        "pefnet_no_external_evidence",
        "t1",
    ]
    labels = [
        "CI + paired 95% band",
        "Post-only",
        "No external",
        DISPLAY_LABELS["t1"],
    ]
    ax.legend(
        [handles[method] for method in legend_order],
        labels,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        ncol=2,
        borderaxespad=0.0,
        frameon=False,
        fontsize=5.9,
        columnspacing=0.70,
        handlelength=2.0,
        handletextpad=0.38,
        labelspacing=0.35,
    )


def _write_summary(output_dir: Path, stem: str, difference: pd.DataFrame) -> None:
    ci = difference[difference["method"] == "covariance_intersection"]
    significant = int((ci["ci95_low"] > 0.0).sum())
    lines = [
        "# AV2 Timestep RMSE Figure Summary",
        "",
        "- Panel (a) shows only PEFNet and CI absolute RMSE with separate scene-bootstrap 95% bands.",
        "- Panel (b) shows comparison-method RMSE minus PEFNet RMSE; positive values favor PEFNet.",
        "- The CI difference band uses paired scene bootstrap resampling.",
        "- CI's paired 95% band is entirely above zero at {}/100 evaluation steps.".format(significant),
        "- Continuous markers are removed; secondary methods are shown with lower-contrast grey lines.",
        "",
    ]
    (output_dir / "{}_summary.md".format(stem)).write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="PDF path; SVG/PNG siblings are also written.")
    parser.add_argument("--curve-output", type=Path, default=None)
    parser.add_argument("--difference-output", type=Path, default=None)
    parser.add_argument("--bootstrap-repetitions", type=int, default=4000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.data)
    required = {"method", "scenario_id", "eval_step", "time_seconds", "position_squared_error"}
    if not required.issubset(frame.columns):
        raise RuntimeError("Missing timestep columns: {}".format(sorted(required - set(frame.columns))))
    if set(frame["method"]) != set(METHODS) or len(frame) != 5 * 200 * 100:
        raise RuntimeError("Timestep input must contain five methods x 200 scenes x 100 steps.")
    if args.bootstrap_repetitions < 100:
        raise ValueError("--bootstrap-repetitions must be at least 100")

    curve, difference = bootstrap_statistics(frame, repetitions=args.bootstrap_repetitions)
    curve_output = args.curve_output or args.output.with_name("timestep_curve.csv")
    difference_output = args.difference_output or args.output.with_name("timestep_difference_curve.csv")
    curve_output.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(curve_output, index=False)
    difference.to_csv(difference_output, index=False)

    apply_paper_style()
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.10), sharex=True)
    _plot_absolute(axes[0], curve)
    _plot_differences(axes[1], difference)
    axes[0].tick_params(labelbottom=False)
    xmax = float(curve["time_seconds"].max())
    ticks = [0.0, 2.0, 4.0, 6.0, 8.0, xmax]
    axes[0].set_xticks(ticks)
    axes[1].set_xticks(ticks)
    fig.subplots_adjust(left=0.17, right=0.985, top=0.965, bottom=0.12, hspace=0.22)
    save_figure(fig, args.output.parent, args.output.stem)
    _write_summary(args.output.parent, args.output.stem, difference)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
