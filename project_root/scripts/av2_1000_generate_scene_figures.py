"""Create a paper-style SVG preview from the completed AV2 1000 statistics."""
from __future__ import annotations

import argparse
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "full_pefnet": "#D55E00",
    "covariance_intersection": "#0072B2",
    "posterior_only": "#CC79A7",
    "pefnet_no_external_evidence": "#009E73",
    "t1": "#E69F00",
    "grid": "#D9E0E7",
    "text": "#17202A",
    "muted": "#5D6D7E",
}
METHODS = ["full_pefnet", "covariance_intersection", "posterior_only", "pefnet_no_external_evidence", "t1"]
METHOD_LABELS = {
    "full_pefnet": "Full PEFNet",
    "covariance_intersection": "Covariance Intersection",
    "posterior_only": "Posterior-Only",
    "pefnet_no_external_evidence": "No External Evidence",
    "t1": "T1",
}


def line(x1: float, y1: float, x2: float, y2: float, **attrs: str) -> str:
    values = " ".join(f'{key}="{escape(str(value))}"' for key, value in attrs.items())
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" {values}/>'


def text(x: float, y: float, value: str, size: int = 24, anchor: str = "start", color: str | None = None, weight: int = 400) -> str:
    color = color or COLORS["text"]
    return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" fill="{color}" font-weight="{weight}">{escape(value)}</text>'


def circle(x: float, y: float, radius: float, color: str) -> str:
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{color}" stroke="#FFFFFF" stroke-width="1.5"/>'


def rect(x: float, y: float, width: float, height: float, color: str) -> str:
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" fill="{color}"/>'


def xscale(value: float, low: float, high: float, left: float, right: float) -> float:
    return left + (value - low) / (high - low) * (right - left)


def ticks(parts: list[str], low: float, high: float, left: float, right: float, top: float, bottom: float) -> None:
    for value in [low + (high - low) * index / 4 for index in range(5)]:
        x = xscale(value, low, high, left, right)
        parts.append(line(x, top, x, bottom, stroke=COLORS["grid"], **{"stroke-width": "1"}))
        parts.append(text(x, bottom + 30, f"{value:.1f}", 18, "middle", COLORS["muted"]))


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size=size)


def _anchor(draw: ImageDraw.ImageDraw, xy: tuple[float, float], value: str, font: ImageFont.FreeTypeFont, anchor: str = "la", fill: str = COLORS["text"]) -> None:
    draw.text(xy, value, font=font, fill=fill, anchor=anchor)


def render_png(output: Path, motion: pd.DataFrame, city: pd.DataFrame, win_counts: pd.DataFrame) -> None:
    width, height = 1500, 1280
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    large, normal, small = _font(32, True), _font(21), _font(18)
    _anchor(draw, (90, 58), "AV2 1000: Scene and subgroup comparison", large)
    _anchor(draw, (90, 90), "Test-set position RMSE after measurement-seed and model-seed aggregation; lower is better.", small, fill=COLORS["muted"])
    legend_x = 90
    for method in METHODS:
        draw.ellipse((legend_x - 8, 114, legend_x + 8, 130), fill=COLORS[method], outline="white", width=1)
        _anchor(draw, (legend_x + 15, 122), METHOD_LABELS[method], small, anchor="lm")
        legend_x += 235 if method != "covariance_intersection" else 290

    left, right, top, bottom = 330, 1390, 190, 490
    _anchor(draw, (90, 168), "A", _font(27, True))
    _anchor(draw, (128, 168), "Position RMSE by motion type", _font(27, True))
    _anchor(draw, (right, 168), "Lower is better", small, anchor="ra", fill=COLORS["muted"])
    for value in [2.0 + 2.8 * index / 4 for index in range(5)]:
        x = xscale(value, 2.0, 4.8, left, right)
        draw.line((x, top, x, bottom), fill=COLORS["grid"], width=1)
        _anchor(draw, (x, bottom + 30), f"{value:.1f}", small, anchor="ma", fill=COLORS["muted"])
    draw.line((left, bottom, right, bottom), fill=COLORS["muted"], width=2)
    motions = [("high_dynamic", "High dynamic"), ("longitudinal", "Longitudinal"), ("stable_straight", "Stable straight"), ("turning", "Turning")]
    for row_index, (motion_key, label) in enumerate(motions):
        y = 235 + row_index * 62
        rows = motion[motion["motion_type"] == motion_key].set_index("method")
        _anchor(draw, (left - 20, y), f"{label} (n={int(rows.iloc[0]['scenarios'])})", normal, anchor="ra")
        draw.line((left, y + 21, right, y + 21), fill=COLORS["grid"], width=1)
        for method_index, method in enumerate(METHODS):
            x = xscale(float(rows.loc[method, "position_rmse_mean"]), 2.0, 4.8, left, right)
            radius = 7 if method == "full_pefnet" else 6
            draw.ellipse((x - radius, y + (method_index - 2) * 8 - radius, x + radius, y + (method_index - 2) * 8 + radius), fill=COLORS[method], outline="white", width=2)
    _anchor(draw, ((left + right) / 2, 530), "Position RMSE (m)", normal, anchor="ma", fill=COLORS["muted"])

    left, right, top, bottom = 330, 1390, 625, 1015
    _anchor(draw, (90, 603), "B", _font(27, True))
    _anchor(draw, (128, 603), "City-level descriptive comparison", _font(27, True))
    _anchor(draw, (right, 603), "Full PEFNet vs Covariance Intersection", small, anchor="ra", fill=COLORS["muted"])
    for value in [2.4 + 1.8 * index / 4 for index in range(5)]:
        x = xscale(value, 2.4, 4.2, left, right)
        draw.line((x, top, x, bottom), fill=COLORS["grid"], width=1)
        _anchor(draw, (x, bottom + 30), f"{value:.1f}", small, anchor="ma", fill=COLORS["muted"])
    draw.line((left, bottom, right, bottom), fill=COLORS["muted"], width=2)
    city_order = [("austin", "Austin"), ("dearborn", "Dearborn"), ("miami", "Miami"), ("palo-alto", "Palo Alto"), ("pittsburgh", "Pittsburgh"), ("washington-dc", "Washington DC")]
    for row_index, (city_key, label) in enumerate(city_order):
        y = 675 + row_index * 53
        rows = city[city["city_name"] == city_key].set_index("method")
        full = xscale(float(rows.loc["full_pefnet", "position_rmse_mean"]), 2.4, 4.2, left, right)
        ci = xscale(float(rows.loc["covariance_intersection", "position_rmse_mean"]), 2.4, 4.2, left, right)
        _anchor(draw, (left - 20, y), f"{label} (n={int(rows.iloc[0]['scenarios'])})", normal, anchor="ra")
        draw.line((min(full, ci), y, max(full, ci), y), fill=COLORS["grid"], width=8)
        draw.ellipse((full - 8, y - 8, full + 8, y + 8), fill=COLORS["full_pefnet"], outline="white", width=2)
        draw.rectangle((ci - 7, y - 7, ci + 7, y + 7), fill=COLORS["covariance_intersection"], outline="white", width=1)
    _anchor(draw, ((left + right) / 2, 1055), "Position RMSE (m)", normal, anchor="ma", fill=COLORS["muted"])

    left, right = 330, 1390
    _anchor(draw, (90, 1088), "C", _font(27, True))
    _anchor(draw, (128, 1088), "Full PEFNet reduction relative to Covariance Intersection", _font(27, True))
    for row_index, (motion_key, label) in enumerate(motions):
        y = 1132 + row_index * 28
        value = float(motion[(motion["method"] == "covariance_intersection") & (motion["motion_type"] == motion_key)]["position_rmse_mean"].iloc[0] - motion[(motion["method"] == "full_pefnet") & (motion["motion_type"] == motion_key)]["position_rmse_mean"].iloc[0])
        wins, scenes = win_counts.loc[motion_key, ["wins", "scenes"]]
        x = xscale(value, 0, 0.8, left, right)
        _anchor(draw, (left - 20, y), label, small, anchor="ra")
        draw.rectangle((left, y - 10, x, y + 9), fill=COLORS["full_pefnet"])
        _anchor(draw, (x + 10, y), f"+{value:.3f} m; {int(wins)}/{int(scenes)} scenes improved", small, anchor="lm", fill=COLORS["muted"])
    _anchor(draw, ((left + right) / 2, 1270), "Mean position-RMSE reduction (m)", small, anchor="ma", fill=COLORS["muted"])
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--png-output", type=Path, help="Optional PNG rendering of the same figure")
    args = parser.parse_args()
    root = args.root.resolve()
    results = root / "results" / "av2_1000"
    motion = pd.read_csv(results / "by_motion.csv")
    city = pd.read_csv(results / "by_city.csv")
    raw = pd.read_csv(results / "scenario_level.csv")
    metadata = pd.read_csv(root / "data" / "manifests" / "av2_1000_v2" / "test_200.csv")
    scenario = (
        raw.groupby(["method", "model_seed", "scenario_id"], as_index=False)["position_rmse"].mean()
        .groupby(["method", "scenario_id"], as_index=False)["position_rmse"].mean()
        .merge(metadata[["scenario_id", "motion_type"]], on="scenario_id", validate="many_to_one")
    )
    paired = scenario[scenario["method"].isin(["full_pefnet", "covariance_intersection"])]
    paired = paired.pivot(index=["scenario_id", "motion_type"], columns="method", values="position_rmse").reset_index()
    win_counts = paired.assign(win=paired["full_pefnet"] < paired["covariance_intersection"]).groupby("motion_type").agg(wins=("win", "sum"), scenes=("win", "size"))

    width, height = 1500, 1280
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif}.axis{stroke:#5D6D7E;stroke-width:1.2}</style>',
        text(90, 58, "AV2 1000: Scene and subgroup comparison", 32, weight=500),
        text(90, 90, "Test-set position RMSE after measurement-seed and model-seed aggregation; lower is better.", 19, color=COLORS["muted"]),
    ]
    legend_x = 90
    for method in METHODS:
        parts.append(circle(legend_x, 122, 8, COLORS[method]))
        parts.append(text(legend_x + 15, 128, METHOD_LABELS[method], 18))
        legend_x += 235 if method != "covariance_intersection" else 290

    left, right, top, bottom = 330, 1390, 190, 490
    parts.extend([text(90, 168, "A", 27, weight=500), text(128, 168, "Position RMSE by motion type", 27, weight=500), text(right, 168, "Lower is better", 18, "end", COLORS["muted"])])
    ticks(parts, 2.0, 4.8, left, right, top, bottom)
    parts.append(line(left, bottom, right, bottom, **{"class": "axis"}))
    motions = [("high_dynamic", "High dynamic"), ("longitudinal", "Longitudinal"), ("stable_straight", "Stable straight"), ("turning", "Turning")]
    for row_index, (motion_key, label) in enumerate(motions):
        y = 235 + row_index * 62
        rows = motion[motion["motion_type"] == motion_key].set_index("method")
        n = int(rows.iloc[0]["scenarios"])
        parts.append(text(left - 20, y + 6, f"{label} (n={n})", 21, "end"))
        parts.append(line(left, y + 21, right, y + 21, stroke=COLORS["grid"], **{"stroke-width": "1"}))
        for method_index, method in enumerate(METHODS):
            value = float(rows.loc[method, "position_rmse_mean"])
            parts.append(circle(xscale(value, 2.0, 4.8, left, right), y + (method_index - 2) * 8, 7 if method == "full_pefnet" else 6, COLORS[method]))
    parts.append(text((left + right) / 2, 530, "Position RMSE (m)", 20, "middle", COLORS["muted"]))

    left, right, top, bottom = 330, 1390, 625, 1015
    parts.extend([text(90, 603, "B", 27, weight=500), text(128, 603, "City-level descriptive comparison", 27, weight=500), text(right, 603, "Full PEFNet vs Covariance Intersection", 18, "end", COLORS["muted"])])
    ticks(parts, 2.4, 4.2, left, right, top, bottom)
    parts.append(line(left, bottom, right, bottom, **{"class": "axis"}))
    city_order = [("austin", "Austin"), ("dearborn", "Dearborn"), ("miami", "Miami"), ("palo-alto", "Palo Alto"), ("pittsburgh", "Pittsburgh"), ("washington-dc", "Washington DC")]
    for row_index, (city_key, label) in enumerate(city_order):
        y = 675 + row_index * 53
        rows = city[city["city_name"] == city_key].set_index("method")
        n = int(rows.iloc[0]["scenarios"])
        full = xscale(float(rows.loc["full_pefnet", "position_rmse_mean"]), 2.4, 4.2, left, right)
        ci = xscale(float(rows.loc["covariance_intersection", "position_rmse_mean"]), 2.4, 4.2, left, right)
        parts.append(text(left - 20, y + 7, f"{label} (n={n})", 21, "end"))
        parts.append(line(min(full, ci), y, max(full, ci), y, stroke=COLORS["grid"], **{"stroke-width": "8"}))
        parts.append(circle(full, y, 8, COLORS["full_pefnet"]))
        parts.append(rect(ci - 7, y - 7, 14, 14, COLORS["covariance_intersection"]))
    parts.append(text((left + right) / 2, 1055, "Position RMSE (m)", 20, "middle", COLORS["muted"]))

    left, right, top, bottom = 330, 1390, 1110, 1240
    parts.extend([text(90, 1088, "C", 27, weight=500), text(128, 1088, "Full PEFNet reduction relative to Covariance Intersection", 27, weight=500)])
    for row_index, (motion_key, label) in enumerate(motions):
        y = 1132 + row_index * 28
        value = float(motion[(motion["method"] == "covariance_intersection") & (motion["motion_type"] == motion_key)]["position_rmse_mean"].iloc[0] - motion[(motion["method"] == "full_pefnet") & (motion["motion_type"] == motion_key)]["position_rmse_mean"].iloc[0])
        wins, scenes = win_counts.loc[motion_key, ["wins", "scenes"]]
        x = xscale(value, 0, 0.8, left, right)
        parts.append(text(left - 20, y + 5, label, 19, "end"))
        parts.append(rect(left, y - 10, x - left, 19, COLORS["full_pefnet"]))
        parts.append(text(x + 10, y + 5, f"+{value:.3f} m; {int(wins)}/{int(scenes)} scenes improved", 18, color=COLORS["muted"]))
    parts.append(text((left + right) / 2, 1270, "Mean position-RMSE reduction (m)", 18, "middle", COLORS["muted"]))
    parts.append("</svg>")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(parts), encoding="utf-8")
    if args.png_output:
        render_png(args.png_output, motion, city, win_counts)
    print(args.output)


if __name__ == "__main__":
    main()
