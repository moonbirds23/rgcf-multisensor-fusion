"""Generate the AV2 paired-scene figure with a grouped improvement panel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple
from zipfile import ZipFile

import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
import numpy as np
import pandas as pd

from paper_figure_style import (
    COLORS,
    MOTION_STYLES,
    apply_paper_style,
    save_figure,
    style_axes,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MOTIONS = ["high_dynamic", "longitudinal", "stable_straight", "turning"]
MOTION_LABELS = {
    "high_dynamic": "High dynamic",
    "longitudinal": "Longitudinal",
    "stable_straight": "Stable straight",
    "turning": "Turning",
}
DISTRIBUTION_ORDER = ["stable_straight", "longitudinal", "turning", "high_dynamic"]
DISTRIBUTION_LABELS = ["Stable\nstraight", "Longitudinal", "Turning", "High\ndynamic"]
SCATTER_LOW = 2.2
SCATTER_HIGH = 5.5
DISTRIBUTION_LOW = -0.18
DISTRIBUTION_HIGH = 3.0


def _blend_with_white(color: str, strength: float = 0.28) -> Tuple[float, float, float]:
    rgb = np.asarray(mpl_colors.to_rgb(color), dtype=np.float64)
    return tuple(1.0 - (1.0 - rgb) * strength)


def _find_unique(filename: str) -> Path:
    matches = list(WORKSPACE_ROOT.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError("Expected one {} under {}, found {}".format(filename, WORKSPACE_ROOT, len(matches)))
    return matches[0]


def _read_scenario_data(args: argparse.Namespace) -> pd.DataFrame:
    if args.scenario_data is not None:
        return pd.read_csv(args.scenario_data)
    if args.formal_archive is not None:
        with ZipFile(args.formal_archive) as archive:
            with archive.open("av2_1000/scenario_level.csv") as stream:
                return pd.read_csv(stream)
    if args.root is not None:
        return pd.read_csv(args.root / "results" / "av2_1000" / "scenario_level.csv")
    archive_path = _find_unique("AV2_1000_FORMAL_RESULTS_20260714.zip")
    with ZipFile(archive_path) as archive:
        with archive.open("av2_1000/scenario_level.csv") as stream:
            return pd.read_csv(stream)


def _read_scene_metadata(args: argparse.Namespace) -> pd.DataFrame:
    if args.metadata_csv is not None:
        metadata = pd.read_csv(args.metadata_csv)
    elif args.metadata_json is not None:
        payload = json.loads(args.metadata_json.read_text(encoding="utf-8"))
        metadata = pd.DataFrame(payload["scatter"])
    elif args.root is not None:
        metadata = pd.read_csv(args.root / "data" / "manifests" / "av2_1000_v2" / "test_200.csv")
    else:
        metadata_path = _find_unique("av2_figure_data.json")
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata = pd.DataFrame(payload["scatter"])

    required = {"scenario_id", "motion_type"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise RuntimeError("Scene metadata is missing columns: {}".format(", ".join(missing)))
    metadata = metadata[["scenario_id", "motion_type"]].drop_duplicates()
    if len(metadata) != 200 or metadata["scenario_id"].duplicated().any():
        raise RuntimeError("Scene metadata must contain exactly 200 unique scenarios.")
    return metadata


def load_paired(args: argparse.Namespace) -> pd.DataFrame:
    raw = _read_scenario_data(args)
    metadata = _read_scene_metadata(args)
    required = {"method", "model_seed", "scenario_id", "position_rmse"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise RuntimeError("Scenario results are missing columns: {}".format(", ".join(missing)))

    scenario = (
        raw.groupby(["method", "model_seed", "scenario_id"], as_index=False)["position_rmse"]
        .mean()
        .groupby(["method", "scenario_id"], as_index=False)["position_rmse"]
        .mean()
        .merge(metadata, on="scenario_id", validate="many_to_one")
    )
    paired = scenario[scenario["method"].isin(["full_pefnet", "covariance_intersection"])]
    paired = paired.pivot(
        index=["scenario_id", "motion_type"], columns="method", values="position_rmse"
    ).reset_index()
    if len(paired) != 200 or paired[["full_pefnet", "covariance_intersection"]].isna().any().any():
        raise RuntimeError("Paired AV2 comparison must contain exactly 200 complete scenes.")
    if set(paired["motion_type"]) != set(MOTIONS):
        raise RuntimeError("Unexpected AV2 motion categories: {}".format(sorted(paired["motion_type"].unique())))
    paired["rmse_reduction"] = paired["covariance_intersection"] - paired["full_pefnet"]
    return paired


def _scatter_by_motion(ax: plt.Axes, frame: pd.DataFrame) -> Dict[str, object]:
    handles: Dict[str, object] = {}
    for motion in MOTIONS:
        part = frame[frame["motion_type"] == motion]
        style = MOTION_STYLES[motion]
        color = str(style["color"])
        visible = part[
            part["covariance_intersection"].between(SCATTER_LOW, SCATTER_HIGH)
            & part["full_pefnet"].between(SCATTER_LOW, SCATTER_HIGH)
        ]
        collection = ax.scatter(
            visible["covariance_intersection"],
            visible["full_pefnet"],
            s=10.5,
            marker=style["marker"],
            facecolors=[_blend_with_white(color)],
            edgecolors=color,
            linewidths=0.52,
            alpha=0.72,
            label=MOTION_LABELS[motion],
            zorder=3,
        )
        handles[motion] = collection

        clipped = part.drop(index=visible.index)
        for _, row in clipped.iterrows():
            x = float(np.clip(row["covariance_intersection"], SCATTER_LOW + 0.025, SCATTER_HIGH - 0.025))
            y = float(np.clip(row["full_pefnet"], SCATTER_LOW + 0.025, SCATTER_HIGH - 0.025))
            if row["covariance_intersection"] > SCATTER_HIGH:
                marker = ">"
            elif row["covariance_intersection"] < SCATTER_LOW:
                marker = "<"
            elif row["full_pefnet"] > SCATTER_HIGH:
                marker = "^"
            else:
                marker = "v"
            ax.scatter(
                [x],
                [y],
                s=19.0,
                marker=marker,
                facecolors=[_blend_with_white(color, 0.38)],
                edgecolors=color,
                linewidths=0.62,
                alpha=0.90,
                zorder=4,
                clip_on=True,
            )
    return handles


def _plot_paired_scatter(ax: plt.Axes, paired: pd.DataFrame) -> None:
    handles = _scatter_by_motion(ax, paired)
    ax.plot(
        [SCATTER_LOW, SCATTER_HIGH],
        [SCATTER_LOW, SCATTER_HIGH],
        color=COLORS["reference"],
        linewidth=0.72,
        linestyle=(0, (4.0, 2.4)),
        zorder=1,
    )
    ax.set_xlim(SCATTER_LOW, SCATTER_HIGH)
    ax.set_ylim(SCATTER_LOW, SCATTER_HIGH)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("CI scene RMSE (m)")
    ax.set_ylabel("PEFNet scene RMSE (m)")
    ax.set_title("(a) Paired scene RMSE", pad=5.0)
    style_axes(ax)

    wins = int((paired["rmse_reduction"] > 0.0).sum())
    ax.text(
        0.975,
        0.025,
        "{}/200 scenes favor PEFNet".format(wins),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.4,
        color=COLORS["muted"],
    )
    legend_order = ["stable_straight", "longitudinal", "turning", "high_dynamic"]
    ax.legend(
        [handles[motion] for motion in legend_order],
        [MOTION_LABELS[motion] for motion in legend_order],
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.985),
        borderaxespad=0.0,
        columnspacing=0.75,
        handletextpad=0.35,
        labelspacing=0.38,
        scatterpoints=1,
        frameon=False,
        fontsize=6.4,
    )


def _plot_reduction_distribution(ax: plt.Axes, paired: pd.DataFrame) -> None:
    rng = np.random.default_rng(20260716)
    positions = np.arange(1, len(DISTRIBUTION_ORDER) + 1, dtype=float)
    values_by_motion = [
        paired.loc[paired["motion_type"] == motion, "rmse_reduction"].to_numpy(float)
        for motion in DISTRIBUTION_ORDER
    ]
    visible_values = [values[(values >= DISTRIBUTION_LOW) & (values <= DISTRIBUTION_HIGH)] for values in values_by_motion]

    violins = ax.violinplot(
        visible_values,
        positions=positions,
        widths=0.74,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        points=80,
    )
    for body, motion in zip(violins["bodies"], DISTRIBUTION_ORDER):
        color = str(MOTION_STYLES[motion]["color"])
        body.set_facecolor(_blend_with_white(color, 0.48))
        body.set_edgecolor(color)
        body.set_linewidth(0.70)
        body.set_alpha(0.82)
        body.set_zorder(1)

    boxes = ax.boxplot(
        values_by_motion,
        positions=positions,
        widths=0.18,
        whis=(5, 95),
        showfliers=False,
        patch_artist=True,
        manage_ticks=False,
        zorder=3,
    )
    for index, motion in enumerate(DISTRIBUTION_ORDER):
        color = str(MOTION_STYLES[motion]["color"])
        boxes["boxes"][index].set_facecolor("white")
        boxes["boxes"][index].set_edgecolor(color)
        boxes["boxes"][index].set_linewidth(0.82)
        boxes["medians"][index].set_color(COLORS["text"])
        boxes["medians"][index].set_linewidth(0.92)
        for item in boxes["whiskers"][2 * index : 2 * index + 2]:
            item.set_color(color)
            item.set_linewidth(0.72)
        for item in boxes["caps"][2 * index : 2 * index + 2]:
            item.set_color(color)
            item.set_linewidth(0.72)

        values = values_by_motion[index]
        x = positions[index] + np.clip(rng.normal(0.0, 0.055, size=len(values)), -0.14, 0.14)
        clipped_y = np.clip(values, DISTRIBUTION_LOW + 0.025, DISTRIBUTION_HIGH - 0.025)
        regular = values <= DISTRIBUTION_HIGH
        ax.scatter(
            x[regular],
            clipped_y[regular],
            s=4.8,
            color=color,
            alpha=0.32,
            linewidths=0.0,
            zorder=2,
        )
        if (~regular).any():
            ax.scatter(
                x[~regular],
                clipped_y[~regular],
                s=16.0,
                marker="^",
                facecolors=[_blend_with_white(color, 0.42)],
                edgecolors=color,
                linewidths=0.55,
                zorder=4,
            )

    ax.axhline(0.0, color=COLORS["axis"], linewidth=0.72, zorder=0)
    ax.set_xlim(0.5, len(DISTRIBUTION_ORDER) + 0.5)
    ax.set_ylim(DISTRIBUTION_LOW, DISTRIBUTION_HIGH)
    ax.set_xticks(positions)
    ax.set_xticklabels(DISTRIBUTION_LABELS)
    ax.set_ylabel("RMSE reduction vs CI (m)")
    ax.set_title("(b) Reduction distribution by motion", pad=5.0)
    style_axes(ax, grid_axis="y")


def _write_summary(output_dir: Path, stem: str, paired: pd.DataFrame) -> None:
    grouped = paired.groupby("motion_type")["rmse_reduction"].agg(["count", "mean", "median", "min", "max"])
    lines = [
        "# AV2 Paired Scene Figure Summary",
        "",
        "- Panel (a) shows paired PEFNet-versus-CI scene RMSE for 200 test scenes.",
        "- {}/200 scenes favor PEFNet.".format(int((paired["rmse_reduction"] > 0.0).sum())),
        "- Five scenes outside the 2.2--5.5 m scatter window are indicated by boundary arrows.",
        "- Panel (b) shows CI minus PEFNet RMSE; positive values favor PEFNet.",
        "- Violin density and jittered points use the visible range; the box/whisker statistics use all scenes.",
        "",
        "| Motion type | Scenes | Mean reduction (m) | Median reduction (m) | Min | Max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for motion in DISTRIBUTION_ORDER:
        row = grouped.loc[motion]
        lines.append(
            "| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                MOTION_LABELS[motion], int(row["count"]), row["mean"], row["median"], row["min"], row["max"]
            )
        )
    (output_dir / "{}_summary.md".format(stem)).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--scenario-data", type=Path, default=None)
    parser.add_argument("--formal-archive", type=Path, default=None)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True, help="PDF path; SVG/PNG siblings are also written.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paired = load_paired(args)
    apply_paper_style()

    fig, axes = plt.subplots(1, 2, figsize=(7.08, 3.05), gridspec_kw={"width_ratios": [1.0, 1.08]})
    _plot_paired_scatter(axes[0], paired)
    _plot_reduction_distribution(axes[1], paired)
    fig.subplots_adjust(left=0.082, right=0.992, top=0.91, bottom=0.17, wspace=0.28)
    save_figure(fig, args.output.parent, args.output.stem)
    _write_summary(args.output.parent, args.output.stem, paired)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
