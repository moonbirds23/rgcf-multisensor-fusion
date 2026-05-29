from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .style import ACADEMIC_COLORS, save_figure


def _fault_normal_curves(df, value):
    tmp = df.copy()
    tmp["fault_active"] = tmp.get("fault_active", 0).fillna(0).astype(int)
    fault = tmp[tmp["fault_active"] == 1].groupby("t")[value].mean()
    normal = tmp[tmp["fault_active"] == 0].groupby("t")[value].mean()
    all_t = sorted(tmp["t"].unique())
    return np.array(all_t), fault.reindex(all_t).to_numpy(), normal.reindex(all_t).to_numpy()

def plot_reliability_chain(rel_df, output_dir=".", cfg=None):
    cfg = cfg or {}
    df = rel_df.copy()
    if "effective_info" not in df.columns and {"weight", "cov_scale"}.issubset(df.columns):
        df["effective_info"] = df["weight"] / df["cov_scale"].replace(0, np.nan)
    metrics = [("gate", "Gate"), ("weight", "Sensor weight"), ("cov_scale", "Covariance scale"), ("effective_info", "Effective info = w / scale")]
    metrics = [(m,l) for m,l in metrics if m in df.columns]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(7.2, 1.6*len(metrics)), sharex=True)
    if len(metrics) == 1: axes = [axes]
    fspan = df[df.get("fault_active", 0).astype(int)==1]
    for ax, (m, label) in zip(axes, metrics):
        t, fault, normal = _fault_normal_curves(df, m)
        ax.plot(t, normal, color=ACADEMIC_COLORS["normal"], lw=1.2, label="Normal sensors mean")
        ax.plot(t, fault, color=ACADEMIC_COLORS["fault"], lw=1.2, label="Fault sensor")
        if not fspan.empty:
            ax.axvspan(fspan["t"].min(), fspan["t"].max(), color=ACADEMIC_COLORS["fault"], alpha=0.10, lw=0)
        ax.set_ylabel(label)
        ax.legend(frameon=False, fontsize=7, loc="best")
    axes[0].set_title("A4 reliability chain over time")
    axes[-1].set_xlabel("Time (s)")
    out = Path(output_dir) / "fig04_reliability_chain"
    style = cfg.get("figure_style", {})
    save_figure(fig, out, dpi=style.get("dpi",300), transparent=style.get("transparent",True), save_svg=style.get("save_svg",True))
    plt.close(fig)
