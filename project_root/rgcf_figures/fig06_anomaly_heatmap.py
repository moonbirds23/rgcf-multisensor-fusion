from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .style import ACADEMIC_COLORS, save_figure


def plot_anomaly_heatmap(pred_df, rel_df=None, output_dir=".", cfg=None):
    cfg = cfg or {}
    df = pred_df.copy()
    # anomaly score priority: error_pos, then fault_active_any
    if "error_pos" in df.columns:
        score = df["error_pos"].to_numpy()
        label = "Position error (m)"
    else:
        score = df.get("fault_active_any", 0).to_numpy()
        label = "Fault activity"
    x = df["truth_x"].to_numpy(); y = df["truth_y"].to_numpy()
    bins = cfg.get("heatmap_bins", 60)
    H_sum, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=score)
    H_cnt, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
    H = H_sum / np.maximum(H_cnt, 1)
    H = np.ma.masked_where(H_cnt == 0, H)
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    im = ax.pcolormesh(xedges, yedges, H.T, shading="auto", cmap="magma", alpha=0.82)
    ax.plot(x, y, color="#222222", lw=0.8, alpha=0.55, label="Truth trajectory")
    if "fault_active_any" in df.columns and df["fault_active_any"].astype(bool).any():
        f = df["fault_active_any"].astype(bool)
        ax.plot(df.loc[f,"truth_x"], df.loc[f,"truth_y"], color=ACADEMIC_COLORS["fault"], lw=2.2, alpha=0.85, label="Fault window")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(label)
    ax.set_xlabel("X position (m)"); ax.set_ylabel("Y position (m)")
    ax.set_title("Spatial anomaly heatmap")
    ax.axis("equal")
    ax.legend(frameon=False, fontsize=7)
    out = Path(output_dir) / "fig06_anomaly_heatmap"
    style = cfg.get("figure_style", {})
    save_figure(fig, out, dpi=style.get("dpi",300), transparent=style.get("transparent",True), save_svg=style.get("save_svg",True))
    plt.close(fig)
