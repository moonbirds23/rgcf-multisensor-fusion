from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .style import ACADEMIC_COLORS, save_figure


def plot_fault_normal_distribution(rel_df, output_dir=".", cfg=None):
    cfg = cfg or {}
    df = rel_df.copy()
    df["fault_active"] = df.get("fault_active", 0).fillna(0).astype(int)
    if "effective_info" not in df.columns and {"weight", "cov_scale"}.issubset(df.columns):
        df["effective_info"] = df["weight"] / df["cov_scale"].replace(0, np.nan)
    metrics = [("gate", "Gate"), ("effective_info", "Effective info")]
    metrics = [(m,l) for m,l in metrics if m in df.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.5*len(metrics), 3.2))
    if len(metrics) == 1: axes = [axes]
    for ax, (m, label) in zip(axes, metrics):
        normal = df.loc[df.fault_active==0, m].dropna().to_numpy()
        fault = df.loc[df.fault_active==1, m].dropna().to_numpy()
        bp = ax.boxplot([normal, fault], labels=["Normal", "Fault"], patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], [ACADEMIC_COLORS["normal"], ACADEMIC_COLORS["fault"]]):
            patch.set_facecolor(color); patch.set_alpha(0.35); patch.set_edgecolor(color)
        ax.set_title(label + " distribution")
        ax.set_ylabel(label)
        if len(normal) > 0 and len(fault) > 0:
            if m == "gate":
                sep = np.nanmean(normal) - np.nanmean(fault)
                ax.text(0.5, 0.95, f"Gate Sep={sep:.3f}", transform=ax.transAxes, ha="center", va="top", fontsize=8)
            if m == "effective_info":
                ratio = np.nanmean(fault) / np.nanmean(normal)
                ax.text(0.5, 0.95, f"Info Ratio={ratio:.3f}", transform=ax.transAxes, ha="center", va="top", fontsize=8)
    out = Path(output_dir) / "fig05_fault_normal_distribution"
    style = cfg.get("figure_style", {})
    save_figure(fig, out, dpi=style.get("dpi",300), transparent=style.get("transparent",True), save_svg=style.get("save_svg",True))
    plt.close(fig)
