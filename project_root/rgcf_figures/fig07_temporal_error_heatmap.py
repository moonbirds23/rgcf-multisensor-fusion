"""
Fig.7 — Temporal tracking error heatmap across ablation groups.

All methods share one absolute color scale. Exceeding values are saturated.
Outputs summary + color-scale diagnostics + per-method percentile reports.

Color scale modes:
  - global_percentile: vmax = p97 of all errors (default, recommended for paper)
  - fixed: user provides vmax (e.g. 5.0)
  - global_max: vmax = global max (not recommended)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_figure(fig, out_path, dpi=600, transparent=True, save_svg=False):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path)+".png", dpi=dpi, bbox_inches="tight", transparent=transparent)
    if save_svg:
        fig.savefig(str(out_path)+".svg", bbox_inches="tight", transparent=transparent)


def compute_position_error(df, *, truth_x_col="truth_x", truth_y_col="truth_y",
                           pred_x_col="pred_x", pred_y_col="pred_y"):
    missing = [c for c in [truth_x_col,truth_y_col,pred_x_col,pred_y_col] if c not in df.columns]
    if missing: raise KeyError(f"Missing: {missing}")
    dx = df[pred_x_col].to_numpy(float) - df[truth_x_col].to_numpy(float)
    dy = df[pred_y_col].to_numpy(float) - df[truth_y_col].to_numpy(float)
    return np.sqrt(dx**2 + dy**2)


def expand_as_heat_strip(values, strip_height):
    return np.repeat(np.asarray(values,float).reshape(1,-1), strip_height, axis=0)


def extract_fault_windows_from_mask(t, mask):
    t=np.asarray(t,float); mask=np.asarray(mask).astype(bool)
    idx=np.where(mask)[0]
    if len(idx)==0: return []
    breaks=np.where(np.diff(idx)>1)[0]+1
    return [(float(t[g[0]]),float(t[g[-1]])) for g in np.split(idx,breaks) if len(g)>0]


def resolve_fault_windows(df, *, time_col="t", fault_col="fault_active_any",
                          cfg_fault_windows=((30.0,70.0),)):
    if fault_col in df.columns:
        return extract_fault_windows_from_mask(
            df[time_col].to_numpy(float), df[fault_col].to_numpy().astype(bool))
    return [(float(a),float(b)) for a,b in cfg_fault_windows] if cfg_fault_windows else []


def build_fault_mask(t, windows):
    mask=np.zeros(len(t),bool)
    for t0,t1 in windows: mask|=(t>=t0)&(t<=t1)
    return mask


# ============================================================
#  Color scale resolution
# ============================================================

def resolve_vmax(error_matrix, *, vmin=0.0, vmax=None,
                 vmax_percentile=97.0, color_scale_mode="global_percentile"):
    """
    Resolve vmax for temporal error heatmap.  All methods share one color axis.

    color_scale_mode:
      - "global_percentile": vmax = p97 of all errors (recommended for main comparison)
      - "fixed": use user-provided vmax
      - "global_max": vmax = global max (not recommended)
    """
    finite = error_matrix[np.isfinite(error_matrix)]
    if finite.size == 0: raise ValueError("No finite error values")

    info = {
        "global_mean": float(np.nanmean(finite)),
        "global_p95": float(np.nanpercentile(finite,95)),
        "global_p97": float(np.nanpercentile(finite,97)),
        "global_p98": float(np.nanpercentile(finite,98)),
        "global_p99": float(np.nanpercentile(finite,99)),
        "global_max": float(np.nanmax(finite)),
    }

    if color_scale_mode == "fixed":
        if vmax is None: raise ValueError("fixed mode requires vmax")
        resolved = float(vmax)
    elif color_scale_mode == "global_max":
        resolved = float(np.nanmax(finite))
    else:  # global_percentile (default)
        resolved = float(np.nanpercentile(finite, vmax_percentile))

    if not np.isfinite(resolved) or resolved <= vmin:
        resolved = vmin + 1.0

    info.update({"vmin": float(vmin), "vmax": float(resolved),
                 "vmax_percentile": float(vmax_percentile),
                 "color_scale_mode": color_scale_mode,
                 "clipped_fraction": float(np.mean(error_matrix > resolved))})
    return resolved, info


# ============================================================
#  Main plotting function
# ============================================================

def plot_temporal_error_heatmap(
    method_dfs, *, output_dir=".", filename="fig07_temporal_error_heatmap",
    method_order=None,
    time_col="t", truth_x_col="truth_x", truth_y_col="truth_y",
    pred_x_col="pred_x", pred_y_col="pred_y",
    fault_col="fault_active_any", fault_windows=((30.0,70.0),),
    color_scale_mode: str = "global_percentile",
    vmax_percentile: float = 97.0,
    vmin: float = 0.0, vmax: Optional[float] = None,
    strip_height=24, cmap="magma",
    show_fault_label=True, fault_label="Fault window (30–70 s)",
    fault_color="#C44E52", fault_edge_alpha=0.75,
    show_title=False, title="Temporal tracking error heatmap",
    xlabel="Time (s)", ylabel="Method", colorbar_label="Tracking error (m)",
    figsize=(8.8,3.2), dpi=600, transparent=True, save_svg=False,
):
    if method_order is None: method_order = list(method_dfs.keys())
    else: method_order = list(method_order)
    for m in method_order:
        if m not in method_dfs: raise KeyError(f"{m} not in method_dfs")

    ref_df = method_dfs[method_order[0]]
    t_ref = ref_df[time_col].to_numpy(float); T = len(t_ref)

    # fault mask for per-method stats
    windows = resolve_fault_windows(ref_df, time_col=time_col, fault_col=fault_col,
                                    cfg_fault_windows=fault_windows)
    fault_mask = build_fault_mask(t_ref, windows)

    error_matrix = []; summary_rows = []
    for method_name in method_order:
        df = method_dfs[method_name]
        t = df[time_col].to_numpy(float)
        if len(t)!=T or not np.allclose(t, t_ref, atol=1e-6):
            raise ValueError(f"{method_name} time mismatch")
        err = compute_position_error(df, truth_x_col=truth_x_col, truth_y_col=truth_y_col,
                                     pred_x_col=pred_x_col, pred_y_col=pred_y_col)
        error_matrix.append(err)

        row = {"method":method_name,
               "mean_error":float(np.nanmean(err)),
               "rmse_error":float(np.sqrt(np.nanmean(err**2))),
               "p95_error":float(np.nanpercentile(err,95)),
               "max_error":float(np.nanmax(err))}
        # fault-window metrics
        err_fault = err[fault_mask] if fault_mask.any() else np.array([])
        if err_fault.size > 0:
            row.update({
                "fault_mean_error": float(np.nanmean(err_fault)),
                "fault_rmse_error": float(np.sqrt(np.nanmean(err_fault**2))),
                "fault_p95_error": float(np.nanpercentile(err_fault,95)),
                "fault_max_error": float(np.nanmax(err_fault)),
                "fault_time_above_4m": float(np.mean(err_fault > 4.0)),
                "fault_tail_area_above_4m": float(np.nanmean(np.maximum(err_fault - 4.0, 0))),
            })
        summary_rows.append(row)

    error_matrix = np.asarray(error_matrix, float)

    # resolve vmax with robust strategy
    vmax_resolved, scale_info = resolve_vmax(
        error_matrix, vmin=vmin, vmax=vmax,
        vmax_percentile=vmax_percentile, color_scale_mode=color_scale_mode)

    # build heatmap
    heat_strips = [expand_as_heat_strip(np.clip(error_matrix[i], vmin, vmax_resolved), strip_height)
                   for i in range(error_matrix.shape[0])]
    heatmap = np.vstack(heat_strips)

    total_height = heatmap.shape[0]
    extent = [float(t_ref[0]), float(t_ref[-1]), 0, total_height]

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0.0); ax.set_facecolor("none")

    im = ax.imshow(heatmap, aspect="auto", origin="upper", extent=extent,
                   cmap=cmap, vmin=vmin, vmax=vmax_resolved, interpolation="nearest", zorder=1)
    ax.set_ylim(total_height, 0)

    y_centers = [i*strip_height + strip_height/2 for i in range(len(method_order))]
    ax.set_yticks(y_centers); ax.set_yticklabels(method_order)

    for i in range(1, len(method_order)):
        ax.axhline(i*strip_height, color="white", linewidth=0.9, alpha=0.85, zorder=3)

    # Fault window: light overlay + edge lines (NOT masking the heatmap colors)
    for idx, (t0, t1) in enumerate(windows):
        ax.axvspan(t0, t1, facecolor=fault_color, alpha=0.06, zorder=2)
        ax.axvline(t0, color=fault_color, linewidth=1.0, alpha=fault_edge_alpha, zorder=5)
        ax.axvline(t1, color=fault_color, linewidth=1.0, alpha=fault_edge_alpha, zorder=5)
        if show_fault_label and idx == 0:
            ax.text((t0+t1)/2, -0.06*total_height, fault_label,
                    ha="center", va="bottom", fontsize=8, color=fault_color, clip_on=False)

    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if show_title: ax.set_title(title, pad=8)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    cbar = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.04)
    cbar.set_label(colorbar_label, fontsize=8); cbar.ax.tick_params(labelsize=8)

    out = Path(output_dir) / filename
    save_figure(fig, out, dpi=dpi, transparent=transparent, save_svg=save_svg)
    plt.close(fig)

    # Save reports
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(Path(output_dir) / f"{filename}_summary.csv", index=False)

    scale_df = pd.DataFrame([scale_info])
    scale_df.to_csv(Path(output_dir) / f"{filename}_color_scale.csv", index=False)

    per_method_rows = []
    for i, method_name in enumerate(method_order):
        row_err = error_matrix[i]
        per_method_rows.append({
            "method": method_name,
            "p95": float(np.nanpercentile(row_err,95)),
            "p97": float(np.nanpercentile(row_err,97)),
            "p98": float(np.nanpercentile(row_err,98)),
            "p99": float(np.nanpercentile(row_err,99)),
            "max": float(np.nanmax(row_err)),
            "clipped_fraction": float(np.mean(row_err > vmax_resolved)),
        })
    pd.DataFrame(per_method_rows).to_csv(
        Path(output_dir) / f"{filename}_per_method_scale.csv", index=False)

    return summary_df
