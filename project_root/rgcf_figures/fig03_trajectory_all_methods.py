"""
Fig.3 — Trajectory comparison across all ablation groups with auto zoom inset.

Shows Truth + AVG + CI-multi + WAA-MM + A1 + A3 + A4 on a single spatial plot.
Fault window highlighted on truth trajectory.
Auto-selects the most divergent local segment within fault window for zoom.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def save_figure(fig, out_path, dpi=600, transparent=True, save_svg=False):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path)+".png", dpi=dpi, bbox_inches="tight", transparent=transparent)
    if save_svg:
        fig.savefig(str(out_path)+".svg", bbox_inches="tight", transparent=transparent)


METHOD_STYLES = {
    "Truth": {"color":"#111111","linestyle":"-","linewidth":2.0,"alpha":1.0,"marker":None,"zorder":20},
    "AVG": {"color":"#8C8C8C","linestyle":"--","linewidth":1.25,"alpha":0.95,"marker":"o","zorder":8},
    "CI-multi": {"color":"#9467BD","linestyle":"-.","linewidth":1.25,"alpha":0.95,"marker":"s","zorder":9},
    "WAA-MM": {"color":"#7F7F00","linestyle":":","linewidth":1.5,"alpha":0.95,"marker":"^","zorder":10},
    "A1 Post-only": {"color":"#1F77B4","linestyle":"-","linewidth":1.25,"alpha":0.95,"marker":"D","zorder":11},
    "A3 Dual-stream": {"color":"#2CA02C","linestyle":"--","linewidth":1.45,"alpha":0.95,"marker":"v","zorder":12},
    "A4 Full RGCF": {"color":"#D62728","linestyle":"-","linewidth":1.7,"alpha":0.98,"marker":"*","zorder":13},
}


def _require_columns(df, cols, name="df"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing: {missing}")


def extract_fault_windows_from_mask(t, mask):
    t = np.asarray(t, dtype=float); mask = np.asarray(mask).astype(bool)
    idx = np.where(mask)[0]
    if len(idx)==0: return []
    breaks = np.where(np.diff(idx)>1)[0]+1
    groups = np.split(idx, breaks)
    return [(float(t[g[0]]), float(t[g[-1]])) for g in groups if len(g)>0]


def resolve_fault_windows(df, *, time_col="t", fault_col="fault_active_any",
                          cfg_fault_windows=((30.0,70.0),)):
    if fault_col in df.columns:
        return extract_fault_windows_from_mask(
            df[time_col].to_numpy(dtype=float),
            df[fault_col].to_numpy().astype(bool))
    return [(float(a),float(b)) for a,b in cfg_fault_windows] if cfg_fault_windows else []


def build_fault_mask(t, windows):
    mask = np.zeros(len(t), dtype=bool)
    for t0,t1 in windows: mask |= (t>=t0) & (t<=t1)
    return mask


def _get_xy(df, x_col, y_col):
    _require_columns(df, [x_col, y_col])
    return df[[x_col, y_col]].to_numpy(dtype=float)


def _max_pairwise_spread(method_xy):
    """method_xy: [M, T, 2] -> spread[t] = max pairwise distance."""
    M, T, _ = method_xy.shape
    spread = np.zeros(T)
    for i in range(M):
        for j in range(i+1, M):
            d = np.linalg.norm(method_xy[i]-method_xy[j], axis=1)
            spread = np.maximum(spread, d)
    return spread


def select_zoom_indices(t, method_xy, fault_mask=None, prefer_fault_window=True, zoom_duration=8.0):
    spread = _max_pairwise_spread(method_xy)
    candidate = np.where(fault_mask)[0] if (prefer_fault_window and fault_mask is not None and fault_mask.any()) else np.arange(len(t))
    center_idx = int(candidate[np.argmax(spread[candidate])])
    dt = float(np.median(np.diff(t))) if len(t)>1 else 1.0
    half = max(3, int(round((zoom_duration/dt)/2.0)))
    return max(0, center_idx-half), min(len(t)-1, center_idx+half), center_idx, spread


def compute_limits(arrays, padding_ratio=0.18, min_span=8.0):
    pts = np.concatenate(arrays, axis=0)
    xmin, ymin = np.nanmin(pts, axis=0); xmax, ymax = np.nanmax(pts, axis=0)
    xspan = max(xmax-xmin, min_span); yspan = max(ymax-ymin, min_span)
    xmid, ymid = 0.5*(xmin+xmax), 0.5*(ymin+ymax)
    xpad, ypad = xspan*padding_ratio, yspan*padding_ratio
    return ((xmid-xspan/2-xpad, xmid+xspan/2+xpad), (ymid-yspan/2-ypad, ymid+yspan/2+ypad))


def plot_masked_truth_segments(ax, truth_xy, fault_mask, color="#C44E52", lw=4.2, alpha=0.42, label="Fault window", zorder=6):
    if fault_mask is None or not fault_mask.any(): return
    idx = np.where(fault_mask)[0]
    breaks = np.where(np.diff(idx)>1)[0]+1
    used = False
    for g in np.split(idx, breaks):
        if len(g)<2: continue
        ax.plot(truth_xy[g,0], truth_xy[g,1], color=color, linewidth=lw, alpha=alpha,
                solid_capstyle="round", label=label if not used else None, zorder=zorder)
        used = True


def plot_trajectory_all_methods(
    method_dfs, *, output_dir=".", filename="fig03_trajectory_all_methods",
    method_order=None, time_col="t", truth_x_col="truth_x", truth_y_col="truth_y",
    pred_x_col="pred_x", pred_y_col="pred_y",
    fault_col="fault_active_any", fault_windows=((30.0,70.0),),
    zoom_duration=8.0, prefer_fault_window=True,
    inset_loc="lower right", inset_width="42%", inset_height="42%",
    inset_min_span=10.0, marker_every=12,
    show_title=False, title="Trajectory comparison across ablation groups",
    figsize=(6.4,5.7), dpi=600, transparent=True, save_svg=False,
):
    if method_order is None: method_order = list(method_dfs.keys())
    ref_df = method_dfs[method_order[0]]
    _require_columns(ref_df, [time_col, truth_x_col, truth_y_col], "ref")

    t = ref_df[time_col].to_numpy(dtype=float)
    truth_xy = ref_df[[truth_x_col, truth_y_col]].to_numpy(dtype=float)

    method_xy_list = []
    for method in method_order:
        df = method_dfs[method]
        _require_columns(df, [time_col, pred_x_col, pred_y_col], method)
        t_i = df[time_col].to_numpy(dtype=float)
        if len(t_i)!=len(t) or not np.allclose(t_i, t, atol=1e-6):
            raise ValueError(f"{method} time axis mismatch")
        method_xy_list.append(_get_xy(df, pred_x_col, pred_y_col))

    method_xy = np.stack(method_xy_list, axis=0)

    windows = resolve_fault_windows(ref_df, time_col=time_col, fault_col=fault_col,
                                    cfg_fault_windows=fault_windows)
    fault_mask = build_fault_mask(t, windows)

    s_idx, e_idx, c_idx, spread = select_zoom_indices(
        t, method_xy, fault_mask=fault_mask, prefer_fault_window=prefer_fault_window,
        zoom_duration=zoom_duration)
    zsl = slice(s_idx, e_idx+1)
    zoom_arrays = [truth_xy[zsl]] + [xy[zsl] for xy in method_xy_list]
    xlim_zoom, ylim_zoom = compute_limits(zoom_arrays, padding_ratio=0.20, min_span=inset_min_span)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0.0); ax.set_facecolor("none")

    ts = METHOD_STYLES["Truth"]
    ax.plot(truth_xy[:,0], truth_xy[:,1], label="Truth", color=ts["color"],
            linestyle=ts["linestyle"], linewidth=ts["linewidth"], alpha=ts["alpha"], zorder=ts["zorder"])
    plot_masked_truth_segments(ax, truth_xy, fault_mask)

    for method, xy in zip(method_order, method_xy_list):
        s = METHOD_STYLES.get(method, {"color":None,"linestyle":"-","linewidth":1.2,"alpha":0.9,"marker":None,"zorder":10})
        ax.plot(xy[:,0], xy[:,1], label=method, color=s["color"], linestyle=s["linestyle"],
                linewidth=s["linewidth"], alpha=s["alpha"], zorder=s["zorder"])

    rect = Rectangle((xlim_zoom[0],ylim_zoom[0]), xlim_zoom[1]-xlim_zoom[0], ylim_zoom[1]-ylim_zoom[0],
                     fill=False, edgecolor="#333", linewidth=1.0, linestyle="-", zorder=30)
    ax.add_patch(rect)

    axins = inset_axes(ax, width=inset_width, height=inset_height, loc=inset_loc, borderpad=1.1)
    axins.set_facecolor("none")
    axins.plot(truth_xy[zsl,0], truth_xy[zsl,1], color=ts["color"], linestyle=ts["linestyle"],
               linewidth=1.8, alpha=1.0, zorder=ts["zorder"])

    for method, xy in zip(method_order, method_xy_list):
        s = METHOD_STYLES.get(method, {})
        m = s.get("marker",None)
        axins.plot(xy[zsl,0], xy[zsl,1], color=s.get("color"), linestyle=s.get("linestyle","-"),
                   linewidth=max(1.7,s.get("linewidth",1.2)), alpha=1.0, marker=m,
                   markevery=max(1,marker_every), markersize=3.0 if m!="*" else 4.5,
                   markerfacecolor="none" if m!="*" else s.get("color"),
                   markeredgewidth=0.8, zorder=s.get("zorder",10))

    axins.set_xlim(*xlim_zoom); axins.set_ylim(*ylim_zoom)
    axins.grid(True, color="#DDD", linewidth=0.55, alpha=0.65)
    axins.tick_params(labelsize=6)
    axins.set_title(f"Zoom: {t[s_idx]:.1f}–{t[e_idx]:.1f}s", fontsize=7, pad=2)

    try:
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="#333", linewidth=0.8, alpha=0.85)
    except Exception: pass

    ax.set_xlabel("X position (m)"); ax.set_ylabel("Y position (m)")
    if show_title: ax.set_title(title, pad=8)
    ax.axis("equal")
    all_pts = [truth_xy] + method_xy_list
    xl, yl = compute_limits(all_pts, padding_ratio=0.08, min_span=100.0)
    ax.set_xlim(*xl); ax.set_ylim(*yl)
    ax.grid(True, color="#DADADA", linewidth=0.7, alpha=0.65)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=7, handlelength=2.4, ncol=1)

    out = Path(output_dir) / filename
    save_figure(fig, out, dpi=dpi, transparent=transparent, save_svg=save_svg)
    plt.close(fig)

    zoom_info = {"zoom_start_time":float(t[s_idx]), "zoom_end_time":float(t[e_idx]),
                 "zoom_center_time":float(t[c_idx]), "zoom_center_spread_m":float(spread[c_idx]),
                 "zoom_start_idx":int(s_idx), "zoom_end_idx":int(e_idx), "zoom_center_idx":int(c_idx),
                 "fault_windows":windows}
    pd.DataFrame([zoom_info]).to_csv(Path(output_dir)/f"{filename}_zoom_info.csv", index=False)
    return zoom_info
