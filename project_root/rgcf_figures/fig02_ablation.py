from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .style import ACADEMIC_COLORS, save_figure


def _col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def plot_ablation_summary(ablation_df, output_dir=".", cfg=None):
    cfg = cfg or {}
    df = ablation_df.copy()
    method_col = _col(df, ["method", "Method"])
    metrics = [("overall_rmse", "Overall RMSE"), ("fault_rmse", "Fault RMSE"), ("fault_p95", "Fault P95")]
    metrics = [(c,l) for c,l in metrics if c in df.columns]
    x = np.arange(len(df))
    width = 0.22
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    colors = ["#8C8C8C", "#2E7BB4", "#D95F02"]
    for j, (c, label) in enumerate(metrics):
        vals = df[c].astype(float).to_numpy()
        ax.bar(x + (j-(len(metrics)-1)/2)*width, vals, width=width, label=label, color=colors[j], alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(df[method_col], rotation=28, ha="right")
    ax.set_ylabel("Error (m)")
    ax.set_title("Main ablation results")
    ax.legend(frameon=False, ncol=len(metrics), loc="upper right")
    # annotate A4 reliability metrics if available
    if "gate_sep" in df.columns and "info_ratio" in df.columns:
        a4 = df[df[method_col].astype(str).str.contains("A4|RGCF", case=False, regex=True)]
        if not a4.empty:
            gs = a4["gate_sep"].iloc[0]
            ir = a4["info_ratio"].iloc[0]
            ax.text(0.99, 0.96, f"A4 reliability\nGate Sep={gs:.3g}\nInfo Ratio={ir:.3g}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#CCCCCC", alpha=0.75))
    out = Path(output_dir) / "fig02_ablation"
    style = cfg.get("figure_style", {})
    save_figure(fig, out, dpi=style.get("dpi",300), transparent=style.get("transparent",True), save_svg=style.get("save_svg",True))
    plt.close(fig)
