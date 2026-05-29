from dataclasses import dataclass
import matplotlib.pyplot as plt

ACADEMIC_COLORS = {
    "truth": "#222222",
    "A1": "#8C8C8C",
    "A3": "#2E7BB4",
    "A4": "#D95F02",
    "fault": "#C44E52",
    "normal": "#4C9F70",
    "gate": "#6A51A3",
    "weight": "#1F78B4",
    "cov": "#E69F00",
    "info": "#009E73",
    "grid": "#D9D9D9",
}

SENSOR_MARKERS = {"S1": "o", "S2": "s", "S3": "^", "S4": "D"}
SENSOR_COLORS = {"S1": "#0072B2", "S2": "#E69F00", "S3": "#009E73", "S4": "#CC79A7"}

def apply_paper_style(font_family="DejaVu Sans"):
    plt.rcParams.update({
        "font.family": font_family,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": ACADEMIC_COLORS["grid"],
        "grid.linewidth": 0.5,
        "grid.alpha": 0.55,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.edgecolor": "none",
    })

def save_figure(fig, out_path, dpi=300, transparent=True, save_svg=True, save_pdf=False):
    out_path = str(out_path)
    fig.savefig(out_path + ".png", dpi=dpi, bbox_inches="tight", transparent=transparent)
    if save_svg:
        fig.savefig(out_path + ".svg", bbox_inches="tight", transparent=transparent)
    if save_pdf:
        fig.savefig(out_path + ".pdf", bbox_inches="tight", transparent=transparent)
