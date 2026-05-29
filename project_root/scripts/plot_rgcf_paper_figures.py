"""
RGCF Balanced Mixed — Paper-Ready Figure Generation.

Generates 5 core figures from the latest RGCF experiment:
  fig1_dataset_balance.pdf/png
  fig2_training_curve.pdf/png
  fig3_rmse_comparison.pdf/png
  fig4_reliability_timeseries.pdf/png
  fig5_fault_normal_separation.pdf/png

Usage:
    py -u scripts/plot_rgcf_paper_figures.py \
      --result-dir "results/20260509_190726__train__hetero_robust_matrix_mixed_v2_rgcf__hetero_4sensor_scene__post_meas_soft_gate_fusion" \
      --dataset-dir "dataset_store/20260509_184532__hetero_robust_matrix_mixed_v2_rgcf__hetero_robust_matr__3d42f300" \
      --out-dir "figures/rgcf_balanced_mixed_v1"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ensure project_root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ============================================================
#  Style
# ============================================================

SENSOR_COLORS = {
    "S1": "#8da0cb",
    "S2": "#fc8d62",
    "S3": "#66c2a5",
    "S4": "#e78ac3",
}

METHOD_COLORS = {
    "S1": "#8da0cb",
    "S2": "#fc8d62",
    "S3": "#66c2a5",
    "S4": "#e78ac3",
    "AVG": "#999999",
    "CI-chain": "#2166ac",
    "CI-multi": "#4dac26",
    "Best-single": "#7b6fd0",
    "RGCF": "#d6604d",
}

METHOD_LS = {
    "AVG": "--",
    "CI-chain": "-.",
    "CI-multi": ":",
    "Best-single": (0, (5, 1)),
    "RGCF": "-",
}

TRUE_COLOR = "#202020"
FAULT_COLOR = "#d6604d"
NORMAL_COLOR = "#4dac26"


def apply_rgcf_paper_style(base_fontsize: float = 8.8) -> None:
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",

        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize,
        "axes.labelsize": base_fontsize - 0.4,
        "legend.fontsize": base_fontsize - 1.8,
        "xtick.labelsize": base_fontsize - 1.8,
        "ytick.labelsize": base_fontsize - 1.8,

        "figure.dpi": 150,
        "savefig.dpi": 600,

        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": "--",
        "grid.linewidth": 0.45,

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,

        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
    })


# ============================================================
#  Helpers
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_fig_bundle(fig, out_stem: Path, transparent: bool = True,
                    also_png: bool = True) -> None:
    ensure_dir(out_stem.parent)
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight",
                pad_inches=0.03, transparent=transparent)
    if also_png:
        fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight",
                    pad_inches=0.03, transparent=transparent)
    print(f"  Saved: {out_stem}.pdf")
    if also_png:
        print(f"  Saved: {out_stem}.png")
    plt.close(fig)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def pick_col(df: pd.DataFrame, candidates: list) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def rolling_mean(y, window: int = 9):
    arr = np.asarray(y, dtype=float)
    if window <= 1 or arr.size < window:
        return arr
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")[:arr.size]


# ============================================================
#  Data loaders
# ============================================================

def load_gate_wide_df(artifacts_dir: Path) -> pd.DataFrame:
    """Load the wide-format gate_timeseries.csv (contains gate/weight/cov_scale)."""
    path = artifacts_dir / "gate_timeseries.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing gate_timeseries.csv at {path}")
    return pd.read_csv(path)


def load_timeseries_long(artifacts_dir: Path, filename: str) -> pd.DataFrame:
    """Load a long-format timeseries file (sensor_id, t, value, ...)."""
    path = artifacts_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing {filename} at {path}")
    return pd.read_csv(path)


# ============================================================
#  Fig 1 — Dataset Balance
# ============================================================

def get_joint_count(split_summary: dict, fault_type: str, sensor_label: str) -> int:
    """Read joint count from either joint_table or joint_counts."""
    # Preferred structured table:
    jt = split_summary.get("joint_table", None)
    if isinstance(jt, dict):
        ft_row = jt.get(fault_type, {})
        if isinstance(ft_row, dict) and sensor_label in ft_row:
            return int(ft_row.get(sensor_label, 0))

    # Fallback flat joint_counts:
    jc = split_summary.get("joint_counts", None)
    if isinstance(jc, dict):
        key1 = f"{fault_type}_{sensor_label}"
        key2 = f"{fault_type}_{sensor_label.lower()}"
        key3 = f"{fault_type}_{sensor_label.replace('S', '')}"
        for key in (key1, key2, key3):
            if key in jc:
                return int(jc[key])

    return 0


def plot_dataset_balance(dataset_dir: Path, out_dir: Path,
                         transparent: bool = True, pdf_only: bool = False) -> None:
    meta = load_json(dataset_dir / "meta.json")
    summary = meta["split_case_summary"]

    apply_rgcf_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)

    # --- (a) Fault type counts by split ---
    ax = axes[0]
    modes = ["clean", "pollution", "bias_ramp", "dropout"]
    splits = ["train", "val", "test"]
    x = np.arange(len(modes))
    w = 0.24
    hatches = ["", "//", "\\\\"]
    colors = ["#bbbbbb", "#eeeeee", "#999999"]

    for i, split in enumerate(splits):
        counts = [summary[split]["mode_counts"].get(m, 0) for m in modes]
        offset = (i - 1) * w
        bars = ax.bar(x + offset, counts, w, label=split,
                      color=colors[i], edgecolor="#555555", linewidth=0.5,
                      hatch=hatches[i])
        for bar, v in zip(bars, counts):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                        str(v), ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize().replace("_", " ") for m in modes])
    ax.set_ylabel("Count")
    ax.legend(fontsize=6.5, ncol=3, loc="upper right", framealpha=0.7)
    ax.set_title("(a)", loc="left", fontweight="bold")

    # --- (b) Fault type × sensor heatmap (pooled train+val+test) ---
    ax = axes[1]
    fault_types = ["pollution", "bias_ramp", "dropout"]
    sensors = ["S1", "S2", "S3", "S4"]
    data = np.zeros((len(fault_types), len(sensors)), dtype=int)

    for i, ft in enumerate(fault_types):
        for j, sv in enumerate(sensors):
            val = 0
            for split in splits:
                val += get_joint_count(summary[split], ft, sv)
            data[i, j] = val

    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(len(sensors)))
    ax.set_xticklabels(sensors)
    ax.set_yticks(range(len(fault_types)))
    ax.set_yticklabels([f.capitalize().replace("_", " ") for f in fault_types])

    for i in range(len(fault_types)):
        for j in range(len(sensors)):
            ax.text(j, i, str(data[i, j]), ha="center", va="center",
                    fontsize=7.5, fontweight="bold",
                    color="white" if data[i, j] > max(np.max(data) // 2, 3) else "black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
    cbar.ax.tick_params(labelsize=6.5)
    ax.set_title("(b) Fault type × sensor counts, all splits", loc="left", fontweight="bold")

    also_png = not pdf_only
    save_fig_bundle(fig, out_dir / "fig1_dataset_balance", transparent=transparent,
                    also_png=also_png)


# ============================================================
#  Fig 2 — Training Curve
# ============================================================

def plot_training_curve(result_dir: Path, out_dir: Path,
                        transparent: bool = True, pdf_only: bool = False) -> None:
    history_path = result_dir / "history" / "train_history.csv"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing train_history.csv at {history_path}")
    df = load_csv(history_path)

    # find best epoch
    val_col = pick_col(df, ["val_loss", "val_loss_total"])
    if val_col is None:
        raise KeyError("Cannot find val_loss column in train_history.csv")
    best_idx = df[val_col].idxmin()
    best_epoch = int(df.loc[best_idx, "epoch"])

    apply_rgcf_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)

    train_loss_col = pick_col(df, ["train_loss", "train_loss_total"])
    val_loss_col = val_col
    train_pos_col = pick_col(df, ["train_loss_pos", "train_pos"])
    val_pos_col = pick_col(df, ["val_loss_pos", "val_pos"])
    train_vel_col = pick_col(df, ["train_loss_vel", "train_vel"])
    val_vel_col = pick_col(df, ["val_loss_vel", "val_vel"])
    train_gate_col = pick_col(df, ["train_loss_gate", "train_gate"])
    train_prior_col = pick_col(df, ["train_loss_gate_prior", "train_prior"])
    mean_gate_col = pick_col(df, ["train_mean_gate", "mean_gate"])

    # (a) Total loss
    ax = axes[0]
    epochs = df["epoch"].values
    if train_loss_col:
        ax.plot(epochs, df[train_loss_col], color="#333333", lw=1.0, label="Train")
    if val_loss_col:
        ax.plot(epochs, df[val_loss_col], color=FAULT_COLOR, lw=1.0, label="Val")
    ax.axvline(best_epoch, color=FAULT_COLOR, ls="--", lw=0.9)
    ax.text(
        0.48, 0.90,
        f"Best epoch = {best_epoch}",
        transform=ax.transAxes,
        fontsize=6.6,
        color=FAULT_COLOR,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=FAULT_COLOR, lw=0.45, alpha=0.70),
    )
    ax.set_ylabel("Total Loss")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=6.5, framealpha=0.7)
    ax.set_title("(a)", loc="left", fontweight="bold")

    # (b) Position loss
    ax = axes[1]
    if train_pos_col:
        ax.plot(epochs, df[train_pos_col], color="#333333", lw=1.0, label="Train Pos")
    if val_pos_col:
        ax.plot(epochs, df[val_pos_col], color=FAULT_COLOR, lw=1.0, label="Val Pos")
    ax.axvline(best_epoch, color=FAULT_COLOR, ls="--", lw=0.9)
    ax.text(
        0.48, 0.90,
        f"Best epoch = {best_epoch}",
        transform=ax.transAxes,
        fontsize=6.6,
        color=FAULT_COLOR,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=FAULT_COLOR, lw=0.45, alpha=0.70),
    )
    ax.set_ylabel("Position Loss")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=6.5, framealpha=0.7)
    ax.set_title("(b)", loc="left", fontweight="bold")

    # (c) Reliability losses / mean gate
    ax = axes[2]
    if train_gate_col:
        ax.plot(epochs, df[train_gate_col], color="#2166ac", lw=1.0, ls="-", label="Gate loss")
    if train_prior_col:
        ax.plot(epochs, df[train_prior_col], color="#4dac26", lw=0.8, ls="--", label="Prior loss")
    if mean_gate_col:
        ax2 = ax.twinx()
        ax2.plot(epochs, df[mean_gate_col], color=FAULT_COLOR, lw=1.0, ls=":",
                 label="Mean gate (right)")
        ax2.set_ylabel("Mean Gate", color=FAULT_COLOR)
        ax2.tick_params(axis="y", colors=FAULT_COLOR)
        ax2.set_ylim(0, 1)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    lines1, labels1 = ax.get_legend_handles_labels()
    if mean_gate_col:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, framealpha=0.7)
    else:
        ax.legend(fontsize=6.5, framealpha=0.7)
    ax.set_title("(c)", loc="left", fontweight="bold")

    also_png = not pdf_only
    save_fig_bundle(fig, out_dir / "fig2_training_curve", transparent=transparent,
                    also_png=also_png)


# ============================================================
#  Fig 3 — RMSE Comparison
# ============================================================

def plot_rmse_comparison(result_dir: Path, out_dir: Path,
                         transparent: bool = True, pdf_only: bool = False) -> None:
    baseline_path = result_dir / "metrics" / "quick_baseline_metrics.json"
    gnn_path = result_dir / "metrics" / "quick_gnn_metrics.json"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing {baseline_path}")
    if not gnn_path.exists():
        raise FileNotFoundError(f"Missing {gnn_path}")

    base = load_json(baseline_path)
    gnn = load_json(gnn_path)

    labels = ["S1", "S2", "S3", "S4", "AVG", "CI-chain", "CI-multi",
              "Best-single", "RGCF"]
    values = [
        base["rmse_s1"],
        base["rmse_s2"],
        base["rmse_s3"],
        base["rmse_s4"],
        base["rmse_avg"],
        base["rmse_ci_chain"],
        base["rmse_ci_multi"],
        base["rmse_best_single"],
        gnn["rmse_gnn_pos"],
    ]
    colors = [METHOD_COLORS.get(l, "#999999") for l in labels]

    apply_rgcf_paper_style()
    fig, ax = plt.subplots(figsize=(6.6, 3.2), constrained_layout=True)

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, edgecolor="#444444", linewidth=0.5, height=0.65)

    for bar, v, lbl in zip(bars, values, labels):
        x_pos = bar.get_width() + 0.15
        kw = {"fontsize": 7.5, "va": "center"}
        if lbl == "RGCF":
            kw["fontweight"] = "bold"
            kw["color"] = FAULT_COLOR
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{v:.2f}", **kw)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Position RMSE (m)")
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.25)

    # best-single reference line
    best_single_val = base["rmse_best_single"]
    ax.axvline(
        best_single_val,
        color=METHOD_COLORS["Best-single"],
        lw=0.85,
        ls=":",
        alpha=0.95,
    )
    ax.text(
        best_single_val + 0.05,
        1.02,
        "Best single",
        transform=ax.get_xaxis_transform(),
        color=METHOD_COLORS["Best-single"],
        fontsize=6.5,
        ha="left",
        va="bottom",
    )

    # highlight RGCF bar
    for bar, lbl in zip(bars, labels):
        if lbl == "RGCF":
            bar.set_edgecolor("#8b0000")
            bar.set_linewidth(1.2)

    also_png = not pdf_only
    save_fig_bundle(fig, out_dir / "fig3_rmse_comparison", transparent=transparent,
                    also_png=also_png)


# ============================================================
#  Fig 4 — Reliability Timeseries
# ============================================================

def plot_raw_and_smooth(ax, t, y, label, color, is_fault_sensor=False, smooth_window=9):
    """Plot raw faint line + smoothed solid line for one sensor."""
    lw_smooth = 1.35 if is_fault_sensor else 1.05
    alpha_raw = 0.18
    ax.plot(t, y, color=color, lw=0.45, alpha=alpha_raw)
    ax.plot(
        t,
        rolling_mean(y, smooth_window),
        color=color,
        lw=lw_smooth,
        alpha=0.95,
        label=label,
    )


def plot_reliability_timeseries(result_dir: Path, out_dir: Path,
                                transparent: bool = True, pdf_only: bool = False) -> None:
    artifacts_dir = result_dir / "artifacts"
    gnn_path = result_dir / "metrics" / "quick_gnn_metrics.json"

    # Prefer wide format (all data in gate_timeseries.csv)
    wide_path = artifacts_dir / "gate_timeseries.csv"
    if wide_path.exists():
        df = load_csv(wide_path)
        t = df["t"].values if "t" in df.columns else np.arange(len(df)) * 0.1
        gates = {f"S{i}": df[f"g_s{i}"].values for i in range(1, 5)}
        weights = {f"S{i}": df[f"w_s{i}"].values for i in range(1, 5)}
        covs = {f"S{i}": df[f"cov_scale_s{i}"].values for i in range(1, 5)}
        fault_active = {f"S{i}": df.get(f"fault_active_s{i}", pd.Series([0]*len(df))).values
                        for i in range(1, 5)}
    else:
        # Fallback: long format
        gw_df = load_timeseries_long(artifacts_dir, "gate_timeseries.csv")
        ww_df = load_timeseries_long(artifacts_dir, "weight_timeseries.csv")
        cw_df = load_timeseries_long(artifacts_dir, "cov_scale_timeseries.csv")
        t = gw_df["t"].values
        gates = {}
        weights = {}
        covs = {}
        fault_active = {}
        for sid in [1, 2, 3, 4]:
            mask = gw_df["sensor_id"] == sid
            gates[f"S{sid}"] = gw_df.loc[mask, "gate"].values
            weights[f"S{sid}"] = ww_df.loc[ww_df["sensor_id"] == sid, "weight"].values
            covs[f"S{sid}"] = cw_df.loc[cw_df["sensor_id"] == sid, "cov_scale"].values
            fault_active[f"S{sid}"] = cw_df.loc[cw_df["sensor_id"] == sid, "fault_active"].values

    # fault window
    t0 = 30.0
    t1 = 70.0
    if gnn_path.exists():
        gnn = load_json(gnn_path)
        t0 = float(gnn.get("fault_window_t0", 30.0))
        t1 = float(gnn.get("fault_window_t1", 70.0))

    # identify fault sensors
    fault_sensors = set()
    for label in ["S1", "S2", "S3", "S4"]:
        fa = np.asarray(fault_active.get(label, np.zeros_like(t)), dtype=float)
        if np.nanmax(fa) > 0.5:
            fault_sensors.add(label)

    print("  fault sensors:", sorted(fault_sensors))
    print("  gate columns:", list(gates.keys()))
    print("  weight columns:", list(weights.keys()))
    print("  cov columns:", list(covs.keys()))

    apply_rgcf_paper_style()
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.2), sharex=True,
                             constrained_layout=True)

    # (a) Reliability gate
    ax = axes[0]
    for label in ["S1", "S2", "S3", "S4"]:
        plot_raw_and_smooth(
            ax, t, gates[label], label, SENSOR_COLORS[label],
            is_fault_sensor=(label in fault_sensors), smooth_window=9,
        )
    ax.axvspan(t0, t1, color=FAULT_COLOR, alpha=0.10, lw=0)
    ax.set_ylabel("Gate $g$")
    ax.set_ylim(-0.05, 1.08)
    ax.legend(fontsize=6, ncol=4, loc="upper right", framealpha=0.7, title="Sensor")
    ax.text(
        0.01, 0.08,
        "Faint: raw, solid: rolling mean",
        transform=ax.transAxes,
        fontsize=6.2,
        color="#555555",
    )
    ax.set_title("(a) Reliability gate", loc="left", fontweight="bold")

    # (b) Sensor fusion weight
    ax = axes[1]
    for label in ["S1", "S2", "S3", "S4"]:
        plot_raw_and_smooth(
            ax, t, weights[label], label, SENSOR_COLORS[label],
            is_fault_sensor=(label in fault_sensors), smooth_window=9,
        )
    ax.axvspan(t0, t1, color=FAULT_COLOR, alpha=0.10, lw=0)
    ax.axhline(0.25, color="gray", lw=0.8, ls=":")
    ax.set_ylabel("Weight $w$")
    ax.legend(fontsize=6, ncol=4, loc="upper right", framealpha=0.7, title="Sensor")
    ax.text(
        0.01, 0.08,
        "Faint: raw, solid: rolling mean",
        transform=ax.transAxes,
        fontsize=6.2,
        color="#555555",
    )
    ax.set_title("(b) Sensor fusion weight", loc="left", fontweight="bold")

    # (c) Covariance scale
    ax = axes[2]
    for label in ["S1", "S2", "S3", "S4"]:
        plot_raw_and_smooth(
            ax, t, covs[label], label, SENSOR_COLORS[label],
            is_fault_sensor=(label in fault_sensors), smooth_window=9,
        )
    ax.axvspan(t0, t1, color=FAULT_COLOR, alpha=0.10, lw=0)
    ax.axhline(1.0, color="gray", lw=0.8, ls=":")
    ax.set_ylabel("Cov. scale $\\alpha$")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=6, ncol=4, loc="upper right", framealpha=0.7, title="Sensor")
    ax.text(
        0.01, 0.08,
        "Faint: raw, solid: rolling mean",
        transform=ax.transAxes,
        fontsize=6.2,
        color="#555555",
    )
    ax.set_title("(c) Covariance scale", loc="left", fontweight="bold")

    also_png = not pdf_only
    save_fig_bundle(fig, out_dir / "fig4_reliability_timeseries",
                    transparent=transparent, also_png=also_png)


# ============================================================
#  Fig 5 — Fault-Normal Separation
# ============================================================

def add_panel_annotation(ax, text: str):
    """Add annotation in axes coordinates, safe from PDF vertical-text bug."""
    ax.text(
        0.50,
        0.93,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.2,
        fontweight="bold",
        color="#202020",
        clip_on=False,
        bbox=dict(
            boxstyle="round,pad=0.16",
            fc="white",
            ec="none",
            alpha=0.72,
        ),
    )


def plot_fault_normal_separation(result_dir: Path, out_dir: Path,
                                 transparent: bool = True, pdf_only: bool = False) -> None:
    gnn_path = result_dir / "metrics" / "quick_gnn_metrics.json"
    if not gnn_path.exists():
        raise FileNotFoundError(f"Missing {gnn_path}")
    gnn = load_json(gnn_path)

    apply_rgcf_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), constrained_layout=True)

    # (a) Gate
    ax = axes[0]
    ax.bar(["Normal"], [gnn["mean_gate_normal"]], color=NORMAL_COLOR,
           edgecolor="#333", linewidth=0.5, width=0.45)
    ax.bar(["Fault"], [gnn["mean_gate_fault"]], color=FAULT_COLOR,
           edgecolor="#333", linewidth=0.5, width=0.45)
    ax.set_ylabel("Mean Gate")
    ax.set_ylim(0, 1.05)
    delta = gnn["gate_separation_normal_minus_fault"]
    add_panel_annotation(ax, rf"$\Delta={delta:.3f}$")
    ax.set_title("(a) Gate", loc="left", fontweight="bold")

    # (b) Weight
    ax = axes[1]
    ax.bar(["Normal"], [gnn["mean_weight_normal"]], color=NORMAL_COLOR,
           edgecolor="#333", linewidth=0.5, width=0.45)
    ax.bar(["Fault"], [gnn["mean_weight_fault"]], color=FAULT_COLOR,
           edgecolor="#333", linewidth=0.5, width=0.45)
    ax.set_ylabel("Mean Weight")
    ax.axhline(0.25, color="gray", lw=0.8, ls=":")
    w_max = max(gnn["mean_weight_normal"], gnn["mean_weight_fault"], 0.25)
    ax.set_ylim(0, w_max * 1.35)
    delta_w = gnn["weight_separation_normal_minus_fault"]
    add_panel_annotation(ax, rf"$\Delta={delta_w:.3f}$")
    ax.set_title("(b) Weight", loc="left", fontweight="bold")

    # (c) Covariance scale
    ax = axes[2]
    ax.bar(["Normal"], [gnn["mean_cov_scale_normal"]], color=NORMAL_COLOR,
           edgecolor="#333", linewidth=0.5, width=0.45)
    ax.bar(["Fault"], [gnn["mean_cov_scale_fault"]], color=FAULT_COLOR,
           edgecolor="#333", linewidth=0.5, width=0.45)
    ax.set_ylabel("Mean Cov. Scale")
    c_max = max(gnn["mean_cov_scale_normal"], gnn["mean_cov_scale_fault"])
    ax.set_ylim(0, c_max * 1.25)
    ratio = gnn["cov_scale_fault_to_normal_ratio"]
    add_panel_annotation(ax, rf"$\times {ratio:.2f}$")
    ax.set_title("(c) Cov. scale", loc="left", fontweight="bold")

    also_png = not pdf_only
    save_fig_bundle(fig, out_dir / "fig5_fault_normal_separation",
                    transparent=transparent, also_png=also_png)


# ============================================================
#  Optional Fig 6 — RGCF Trajectory Error (pred_timeseries.csv only)
# ============================================================

def plot_trajectory_error_if_available(result_dir: Path, out_dir: Path,
                                       transparent: bool = True,
                                       pdf_only: bool = False) -> bool:
    """Return True if figure was generated. Only uses pred_timeseries.csv."""
    artifacts_dir = result_dir / "artifacts"
    pred_path = artifacts_dir / "pred_timeseries.csv"

    if not pred_path.exists():
        print("  [skip] pred_timeseries.csv not found; not plotting local sensor error as RGCF error.")
        return False

    pred_df = load_csv(pred_path)

    if "error_pos" not in pred_df.columns:
        # try compute from truth / rgcf prediction columns
        required = ["truth_px", "truth_py", "rgcf_px", "rgcf_py"]
        if all(c in pred_df.columns for c in required):
            pred_df["error_pos"] = np.sqrt(
                (pred_df["rgcf_px"] - pred_df["truth_px"]) ** 2
                + (pred_df["rgcf_py"] - pred_df["truth_py"]) ** 2
            )
        else:
            print("  [skip] pred_timeseries.csv lacks error_pos or truth/rgcf position columns.")
            return False

    print("  using pred_timeseries.csv for RGCF error")
    t = pred_df["t"].values if "t" in pred_df.columns else np.arange(len(pred_df))
    err = pred_df["error_pos"].values

    # fault window
    gnn_path = result_dir / "metrics" / "quick_gnn_metrics.json"
    t0, t1 = 30.0, 70.0
    if gnn_path.exists():
        gnn = load_json(gnn_path)
        t0 = float(gnn.get("fault_window_t0", 30.0))
        t1 = float(gnn.get("fault_window_t1", 70.0))

    apply_rgcf_paper_style()
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.2), constrained_layout=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    # (a) RGCF position error over full time
    ax = axes[0]
    ax.plot(t, err, color=FAULT_COLOR, lw=1.2, label="RGCF")

    # overlay baselines if available
    for col, lbl, clr, ls in [
        ("error_best_single", "Best-single", METHOD_COLORS["Best-single"], "--"),
        ("error_ci_chain", "CI-chain", METHOD_COLORS["CI-chain"], "-."),
        ("error_avg", "AVG", METHOD_COLORS["AVG"], ":"),
    ]:
        if col in pred_df.columns:
            ax.plot(t, pred_df[col].values, color=clr, lw=0.7, ls=ls, label=lbl)

    ax.axvspan(t0, t1, color=FAULT_COLOR, alpha=0.08, lw=0)
    ax.set_ylabel("Position Error (m)")
    ax.legend(fontsize=6.5, ncol=4, loc="upper right", framealpha=0.7)
    ax.set_title("(a) RGCF position error", loc="left", fontweight="bold")

    # (b) Fault window zoom
    ax = axes[1]
    window_mask = (t >= t0) & (t <= t1)
    ax.plot(t[window_mask], err[window_mask], color=FAULT_COLOR, lw=1.2, label="RGCF")

    for col, lbl, clr, ls in [
        ("error_best_single", "Best-single", METHOD_COLORS["Best-single"], "--"),
        ("error_ci_chain", "CI-chain", METHOD_COLORS["CI-chain"], "-."),
        ("error_avg", "AVG", METHOD_COLORS["AVG"], ":"),
    ]:
        if col in pred_df.columns:
            ax.plot(t[window_mask], pred_df[col].values[window_mask],
                    color=clr, lw=0.7, ls=ls, label=lbl)

    ax.set_ylabel("Error (m)")
    ax.set_xlabel("Time (s)")
    ax.set_xlim(t0, t1)
    ax.legend(fontsize=6.5, framealpha=0.7)
    ax.set_title("(b) Fault window zoom", loc="left", fontweight="bold")

    also_png = not pdf_only
    save_fig_bundle(fig, out_dir / "fig6_trajectory_error",
                    transparent=transparent, also_png=also_png)
    return True


# ============================================================
#  CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RGCF Paper-Ready Figure Generator (5 core + optional)"
    )
    p.add_argument("--result-dir", type=str, required=True,
                   help="Training result directory")
    p.add_argument("--dataset-dir", type=str, required=True,
                   help="Dataset store directory (for meta.json / split_case_summary)")
    p.add_argument("--out-dir", type=str, default="figures/rgcf_balanced_mixed_v1",
                   help="Output directory for figures")
    p.add_argument("--pdf-only", action="store_true",
                   help="Save only PDF (skip PNG)")
    p.add_argument("--no-transparent", action="store_true",
                   help="Disable transparent background")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    result_dir = Path(args.result_dir)
    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)

    if not result_dir.exists():
        raise FileNotFoundError(f"result-dir not found: {result_dir}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset-dir not found: {dataset_dir}")

    transparent = not args.no_transparent
    pdf_only = args.pdf_only

    print("=" * 60)
    print("RGCF Paper Figures — Balanced Mixed v1")
    print(f"  result dir : {result_dir}")
    print(f"  dataset dir: {dataset_dir}")
    print(f"  out dir    : {out_dir}")
    print("=" * 60)

    # --- Core Fig 1 ---
    print("\n[1/5] Dataset Balance")
    try:
        plot_dataset_balance(dataset_dir, out_dir,
                             transparent=transparent, pdf_only=pdf_only)
    except Exception as e:
        print(f"  ERROR: {e}")

    # --- Core Fig 2 ---
    print("\n[2/5] Training Curve")
    try:
        plot_training_curve(result_dir, out_dir,
                            transparent=transparent, pdf_only=pdf_only)
    except Exception as e:
        print(f"  ERROR: {e}")

    # --- Core Fig 3 ---
    print("\n[3/5] RMSE Comparison")
    try:
        plot_rmse_comparison(result_dir, out_dir,
                             transparent=transparent, pdf_only=pdf_only)
    except Exception as e:
        print(f"  ERROR: {e}")

    # --- Core Fig 4 ---
    print("\n[4/5] Reliability Timeseries")
    try:
        plot_reliability_timeseries(result_dir, out_dir,
                                    transparent=transparent, pdf_only=pdf_only)
    except Exception as e:
        print(f"  ERROR: {e}")

    # --- Core Fig 5 ---
    print("\n[5/5] Fault-Normal Separation")
    try:
        plot_fault_normal_separation(result_dir, out_dir,
                                     transparent=transparent, pdf_only=pdf_only)
    except Exception as e:
        print(f"  ERROR: {e}")

    # --- Optional Fig 6 ---
    print("\n[Optional] RGCF Trajectory Error")
    try:
        ok = plot_trajectory_error_if_available(result_dir, out_dir,
                                                transparent=transparent,
                                                pdf_only=pdf_only)
        if not ok:
            print("  (skipped — RGCF prediction timeseries not available)")
    except Exception as e:
        print(f"  ERROR: {e}")

    print(f"\nDone. Figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
