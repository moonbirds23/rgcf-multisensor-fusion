"""Shared publication style for PEFNet experiment figures.

The module keeps method identity stable across synthetic and AV2 experiments,
uses a colour-blind-safe palette, and exports transparent vector/raster files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt


# Unified colour-blind-safe palette used by all selected paper figures.
COLORS = {
    "pefnet": "#EE6677",
    "ci": "#228833",
    "waa_mm": "#4477AA",
    "mean": "#CCBB44",
    "single": "#667788",
    "posterior_only": "#8E6C8A",
    "evidence_pooled": "#56B4E9",
    "heterogeneous": "#009E73",
    "no_calibration": "#E69F00",
    "no_external": "#009E73",
    "truth": "#20262E",
    "text": "#20262E",
    "muted": "#647180",
    "axis": "#66717E",
    "grid": "#D9E0E7",
    "reference": "#8A96A3",
}


METHOD_ALIASES = {
    "Ground Truth": "truth",
    "PEFNet": "pefnet",
    "Full PEFNet": "pefnet",
    "full_pefnet": "pefnet",
    "Covariance Intersection": "ci",
    "covariance_intersection": "ci",
    "Covariance-Weighted Moment Matching": "waa_mm",
    "Mean Fusion": "mean",
    "Best Single Posterior": "single",
    "T1": "single",
    "t1": "single",
    "Posterior-Only GNN": "posterior_only",
    "Posterior-Only": "posterior_only",
    "posterior_only": "posterior_only",
    "Evidence-Pooled GNN": "evidence_pooled",
    "Heterogeneous P/M GNN": "heterogeneous",
    "PEFNet without Calibrated Fusion": "no_calibration",
    "No External Evidence": "no_external",
    "pefnet_no_external_evidence": "no_external",
}


LINE_STYLES = {
    "truth": {"linestyle": "-", "marker": None},
    "pefnet": {"linestyle": "-", "marker": "o"},
    "ci": {"linestyle": "-", "marker": "s"},
    "waa_mm": {"linestyle": "-", "marker": "^"},
    "mean": {"linestyle": "-", "marker": "v"},
    "single": {"linestyle": "-", "marker": "D"},
    "posterior_only": {"linestyle": "-", "marker": "D"},
    "evidence_pooled": {"linestyle": "-", "marker": "^"},
    "heterogeneous": {"linestyle": "-", "marker": "P"},
    "no_calibration": {"linestyle": "-", "marker": "v"},
    "no_external": {"linestyle": "-", "marker": "v"},
}


DISPLAY_LABELS = {
    "PEFNet": "PEFNet",
    "Best Single Posterior": "Best single",
    "Covariance Intersection": "CI",
    "Covariance-Weighted Moment Matching": "Cov.-weighted",
    "Mean Fusion": "Mean",
    "Posterior-Only GNN": "Post-only",
    "Evidence-Pooled GNN": "Evidence-pooled",
    "Heterogeneous P/M GNN": "Hetero P/M",
    "PEFNet without Calibrated Fusion": "w/o calib",
    "full_pefnet": "Full PEFNet",
    "covariance_intersection": "CI",
    "posterior_only": "Posterior-only",
    "pefnet_no_external_evidence": "No external evidence",
    "t1": "T1",
}


MOTION_STYLES = {
    "high_dynamic": {"color": "#D55E00", "marker": "o"},
    "longitudinal": {"color": "#0072B2", "marker": "s"},
    "stable_straight": {"color": "#009E73", "marker": "^"},
    "turning": {"color": "#8E6C8A", "marker": "D"},
}


def apply_paper_style() -> None:
    """Apply the common two-column-paper typography and structural styling."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.titleweight": "normal",
            "axes.labelsize": 8.0,
            "axes.labelcolor": COLORS["text"],
            "axes.edgecolor": COLORS["axis"],
            "axes.linewidth": 0.62,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.58,
            "ytick.major.width": 0.58,
            "legend.fontsize": 7.0,
            "legend.frameon": False,
            "legend.handlelength": 2.25,
            "legend.handletextpad": 0.45,
            "legend.columnspacing": 0.9,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.facecolor": "none",
            "figure.facecolor": "none",
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.facecolor": "none",
            "savefig.edgecolor": "none",
            "savefig.transparent": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
        }
    )


def method_style(
    method: str,
    *,
    markers: bool = False,
    markevery: Optional[object] = None,
    alpha: float = 1.0,
    emphasis: Optional[bool] = None,
) -> Dict[str, object]:
    """Return a stable style for a method without mutating global dictionaries."""
    key = METHOD_ALIASES.get(method, method)
    if key not in LINE_STYLES or key not in COLORS:
        raise KeyError("Unknown paper-figure method: {}".format(method))
    highlighted = key == "pefnet" if emphasis is None else emphasis
    result: Dict[str, object] = {
        "color": COLORS[key],
        "linewidth": 1.18 if highlighted else 0.92,
        "linestyle": LINE_STYLES[key]["linestyle"],
        "alpha": alpha,
        "zorder": 5 if highlighted else 3,
    }
    marker = LINE_STYLES[key]["marker"]
    if markers and marker is not None:
        result.update(
            {
                "marker": marker,
                "markersize": 2.45 if highlighted else 2.15,
                "markerfacecolor": "none",
                "markeredgewidth": 0.62,
                "markevery": markevery,
            }
        )
    return result


def style_axes(ax: plt.Axes, *, grid_axis: str = "both", zero_line: bool = False) -> None:
    """Apply thin neutral axes and grids while preserving a transparent canvas."""
    ax.set_facecolor("none")
    for side in ("left", "bottom", "top", "right"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(COLORS["axis"])
        ax.spines[side].set_linewidth(0.62)
    ax.tick_params(direction="out", pad=2.3)
    ax.grid(
        True,
        axis=grid_axis,
        color=COLORS["grid"],
        linewidth=0.46,
        alpha=0.78,
        zorder=0,
    )
    if zero_line:
        ax.axhline(0.0, color=COLORS["axis"], linewidth=0.65, zorder=1)


def save_figure(
    fig: plt.Figure,
    output_dir: Union[str, Path],
    stem: str,
    *,
    png_dpi: int = 600,
    pad_inches: float = 0.025,
) -> Dict[str, Path]:
    """Save transparent PDF/SVG and a high-resolution transparent PNG."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    fig.patch.set_alpha(0.0)
    for ax in fig.axes:
        ax.patch.set_alpha(0.0)
    outputs = {ext: directory / "{}.{}".format(stem, ext) for ext in ("pdf", "svg", "png")}
    fig.savefig(
        outputs["pdf"],
        bbox_inches="tight",
        pad_inches=pad_inches,
        transparent=True,
        metadata={"Creator": "PEFNet paper figure pipeline", "Title": stem},
    )
    fig.savefig(
        outputs["svg"],
        bbox_inches="tight",
        pad_inches=pad_inches,
        transparent=True,
        metadata={"Creator": "PEFNet paper figure pipeline", "Title": stem},
    )
    fig.savefig(
        outputs["png"],
        bbox_inches="tight",
        pad_inches=pad_inches,
        transparent=True,
        dpi=png_dpi,
        metadata={"Software": "PEFNet paper figure pipeline"},
    )
    return outputs
