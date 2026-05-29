from pathlib import Path
import matplotlib.pyplot as plt
from .style import ACADEMIC_COLORS, SENSOR_COLORS, SENSOR_MARKERS, save_figure
from .data_loader import default_sensor_positions


def plot_scene(truth_df, sensor_df=None, output_dir=".", cfg=None):
    cfg = cfg or {}
    if sensor_df is None:
        sensor_df = default_sensor_positions()
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot(truth_df["truth_x"], truth_df["truth_y"], color=ACADEMIC_COLORS["truth"], lw=1.4, label="Truth trajectory")
    if "fault_active_any" in truth_df.columns:
        f = truth_df["fault_active_any"].astype(bool)
        if f.any():
            ax.plot(truth_df.loc[f,"truth_x"], truth_df.loc[f,"truth_y"], color=ACADEMIC_COLORS["fault"], lw=3.0, label="Fault window")
    for _, row in sensor_df.iterrows():
        sid = str(row["sensor_id"]).upper()
        ax.scatter(row["x"], row["y"], s=72, marker=SENSOR_MARKERS.get(sid, "o"),
                   color=SENSOR_COLORS.get(sid, "#333333"), edgecolor="black", linewidth=0.6, zorder=5)
        ax.text(row["x"]+10, row["y"]+10, f"{sid}\n{row.get('sensor_type','')}", fontsize=7, va="bottom")
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_title("Heterogeneous 4-sensor tracking scenario")
    ax.axis("equal")
    ax.legend(frameon=False, loc="best")
    out = Path(output_dir) / "fig01_scene"
    style = cfg.get("figure_style", {})
    save_figure(fig, out, dpi=style.get("dpi",300), transparent=style.get("transparent",True), save_svg=style.get("save_svg",True))
    plt.close(fig)
