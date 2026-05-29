from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .style import ACADEMIC_COLORS, save_figure


def shade_fault(ax, df):
    if "fault_active_any" not in df.columns or "t" not in df.columns:
        return
    f = df["fault_active_any"].astype(bool).to_numpy()
    if not f.any():
        return
    t = df["t"].to_numpy()
    ax.axvspan(t[f][0], t[f][-1], color=ACADEMIC_COLORS["fault"], alpha=0.12, lw=0)

def plot_trajectory_error(a3_df, a4_df, output_dir=".", cfg=None):
    cfg = cfg or {}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios":[1.05,1.25]})
    ax = axes[0]
    ax.plot(a4_df["truth_x"], a4_df["truth_y"], color=ACADEMIC_COLORS["truth"], lw=1.3, label="Truth")
    ax.plot(a3_df["pred_x"], a3_df["pred_y"], color=ACADEMIC_COLORS["A3"], lw=1.1, alpha=0.82, label="A3 Dual-stream")
    ax.plot(a4_df["pred_x"], a4_df["pred_y"], color=ACADEMIC_COLORS["A4"], lw=1.1, alpha=0.9, label="A4 Full RGCF")
    f = a4_df.get("fault_active_any", None)
    if f is not None and f.astype(bool).any():
        idx = f.astype(bool)
        ax.plot(a4_df.loc[idx,"truth_x"], a4_df.loc[idx,"truth_y"], color=ACADEMIC_COLORS["fault"], lw=3.0, alpha=0.7, label="Fault window")
    ax.set_xlabel("X position (m)"); ax.set_ylabel("Y position (m)"); ax.axis("equal")
    ax.set_title("Trajectory comparison")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    ax.plot(a3_df["t"], a3_df["error_pos"], color=ACADEMIC_COLORS["A3"], lw=1.0, label="A3 error")
    ax.plot(a4_df["t"], a4_df["error_pos"], color=ACADEMIC_COLORS["A4"], lw=1.0, label="A4 error")
    shade_fault(ax, a4_df)
    if "fault_active_any" in a4_df.columns:
        mask = a4_df["fault_active_any"].astype(bool)
        if mask.any():
            p95_a3 = np.nanpercentile(a3_df.loc[mask, "error_pos"], 95)
            p95_a4 = np.nanpercentile(a4_df.loc[mask, "error_pos"], 95)
            ax.axhline(p95_a3, color=ACADEMIC_COLORS["A3"], lw=0.9, ls="--", alpha=0.7)
            ax.axhline(p95_a4, color=ACADEMIC_COLORS["A4"], lw=0.9, ls="--", alpha=0.7)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Position error (m)")
    ax.set_title("Error over time")
    ax.legend(frameon=False, fontsize=7)
    out = Path(output_dir) / "fig03_trajectory_error"
    style = cfg.get("figure_style", {})
    save_figure(fig, out, dpi=style.get("dpi",300), transparent=style.get("transparent",True), save_svg=style.get("save_svg",True))
    plt.close(fig)
