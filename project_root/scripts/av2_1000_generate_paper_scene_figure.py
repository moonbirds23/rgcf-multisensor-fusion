"""Generate a publication-ready vector PDF for AV2 1000 scene comparisons."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib.colors import Color, HexColor, white
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


METHODS = ["full_pefnet", "covariance_intersection", "posterior_only", "pefnet_no_external_evidence", "t1"]
METHOD_LABELS = {
    "full_pefnet": "Full",
    "covariance_intersection": "CI",
    "posterior_only": "Posterior",
    "pefnet_no_external_evidence": "No evidence",
    "t1": "T1",
}
MOTIONS = ["high_dynamic", "longitudinal", "stable_straight", "turning"]
MOTION_LABELS = {
    "high_dynamic": "High dynamic",
    "longitudinal": "Longitudinal",
    "stable_straight": "Stable straight",
    "turning": "Turning",
}
CITIES = ["austin", "dearborn", "miami", "palo-alto", "pittsburgh", "washington-dc"]
CITY_LABELS = {
    "austin": "Austin",
    "dearborn": "Dearborn",
    "miami": "Miami",
    "palo-alto": "Palo Alto",
    "pittsburgh": "Pittsburgh",
    "washington-dc": "Washington DC",
}
COLORS = {
    "full_pefnet": HexColor("#D55E00"),
    "covariance_intersection": HexColor("#0072B2"),
    "posterior_only": HexColor("#CC79A7"),
    "pefnet_no_external_evidence": HexColor("#009E73"),
    "t1": HexColor("#E69F00"),
    "high_dynamic": HexColor("#D55E00"),
    "longitudinal": HexColor("#0072B2"),
    "stable_straight": HexColor("#009E73"),
    "turning": HexColor("#CC79A7"),
    "text": HexColor("#20262E"),
    "muted": HexColor("#596675"),
    "grid": HexColor("#D8DEE5"),
}
MARKERS = {
    "full_pefnet": "circle",
    "covariance_intersection": "square",
    "posterior_only": "diamond",
    "pefnet_no_external_evidence": "triangle",
    "t1": "cross",
    "high_dynamic": "circle",
    "longitudinal": "square",
    "stable_straight": "triangle",
    "turning": "diamond",
}


def sx(value: float, low: float, high: float, left: float, right: float) -> float:
    return left + (value - low) / (high - low) * (right - left)


def sy(value: float, low: float, high: float, bottom: float, top: float) -> float:
    return bottom + (value - low) / (high - low) * (top - bottom)


def marker(c: canvas.Canvas, x: float, y: float, size: float, color: Color, shape: str, alpha: float = 1.0) -> None:
    c.saveState()
    if hasattr(c, "setFillAlpha"):
        c.setFillAlpha(alpha)
        c.setStrokeAlpha(alpha)
    c.setFillColor(color)
    c.setStrokeColor(white)
    c.setLineWidth(0.45)
    if shape == "circle":
        c.circle(x, y, size, fill=1, stroke=1)
    elif shape == "square":
        c.rect(x - size, y - size, 2 * size, 2 * size, fill=1, stroke=1)
    elif shape == "diamond":
        path = c.beginPath()
        path.moveTo(x, y + size * 1.25)
        path.lineTo(x + size * 1.05, y)
        path.lineTo(x, y - size * 1.25)
        path.lineTo(x - size * 1.05, y)
        path.close()
        c.drawPath(path, fill=1, stroke=1)
    elif shape == "triangle":
        path = c.beginPath()
        path.moveTo(x, y + size * 1.25)
        path.lineTo(x + size * 1.15, y - size)
        path.lineTo(x - size * 1.15, y - size)
        path.close()
        c.drawPath(path, fill=1, stroke=1)
    else:
        c.setStrokeColor(color)
        c.setLineWidth(1.1)
        c.line(x - size, y - size, x + size, y + size)
        c.line(x - size, y + size, x + size, y - size)
    c.restoreState()


def bootstrap_summary(frame: pd.DataFrame, group_column: str, groups: list[str], seed: int = 20260712) -> dict[tuple[str, str], tuple[float, float, float, int]]:
    rng = np.random.default_rng(seed)
    output: dict[tuple[str, str], tuple[float, float, float, int]] = {}
    for group in groups:
        for method in METHODS:
            values = frame.loc[(frame[group_column] == group) & (frame["method"] == method), "position_rmse"].to_numpy(float)
            means = values[rng.integers(0, len(values), size=(4000, len(values)))].mean(axis=1)
            output[(group, method)] = (float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), len(values))
    return output


def label(c: canvas.Canvas, x: float, y: float, value: str, size: float = 7.2, font: str = "Arial", color: Color | None = None, align: str = "left") -> None:
    c.setFont(font, size)
    c.setFillColor(color or COLORS["text"])
    if align == "right":
        c.drawRightString(x, y, value)
    elif align == "center":
        c.drawCentredString(x, y, value)
    else:
        c.drawString(x, y, value)


def axes(c: canvas.Canvas, left: float, bottom: float, right: float, top: float, ticks: list[float], low: float, high: float, vertical: bool = False) -> None:
    c.setLineWidth(0.45)
    for value in ticks:
        position = sy(value, low, high, bottom, top) if vertical else sx(value, low, high, left, right)
        c.setStrokeColor(COLORS["grid"])
        if vertical:
            c.line(left, position, right, position)
            label(c, left - 1.5 * mm, position - 2.1, f"{value:g}", 6.2, color=COLORS["muted"], align="right")
        else:
            c.line(position, bottom, position, top)
            label(c, position, bottom - 3.7 * mm, f"{value:g}", 6.2, color=COLORS["muted"], align="center")
    c.setStrokeColor(COLORS["muted"])
    c.line(left, bottom, right, bottom)
    c.line(left, bottom, left, top)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    results = root / "results" / "av2_1000"
    raw = pd.read_csv(results / "scenario_level.csv")
    metadata = pd.read_csv(root / "data" / "manifests" / "av2_1000_v2" / "test_200.csv")
    scenario = (
        raw.groupby(["method", "model_seed", "scenario_id"], as_index=False)["position_rmse"].mean()
        .groupby(["method", "scenario_id"], as_index=False)["position_rmse"].mean()
        .merge(metadata[["scenario_id", "motion_type", "city_name"]], on="scenario_id", validate="many_to_one")
    )
    paired = scenario[scenario["method"].isin(["full_pefnet", "covariance_intersection"])]
    paired = paired.pivot(index=["scenario_id", "motion_type", "city_name"], columns="method", values="position_rmse").reset_index()
    motion_stats = bootstrap_summary(scenario[scenario["method"].isin(METHODS)], "motion_type", MOTIONS)
    city_stats = bootstrap_summary(scenario[scenario["method"].isin(METHODS)], "city_name", CITIES, seed=20260713)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(TTFont("Arial", "C:/Windows/Fonts/arial.ttf"))
    pdfmetrics.registerFont(TTFont("Arial-Bold", "C:/Windows/Fonts/arialbd.ttf"))
    width, height = 180 * mm, 145 * mm
    c = canvas.Canvas(str(args.output), pagesize=(width, height), pageCompression=1)
    c.setTitle("AV2 1000 scene-level and subgroup comparison")

    # Panel (a): paired scene-level comparison with a full-range inset.
    label(c, 8 * mm, 139 * mm, "(a) Paired scene-level comparison", 9.2, "Arial-Bold")
    legend_x = 10 * mm
    for motion in MOTIONS:
        marker(c, legend_x, 132.5 * mm, 1.15 * mm, COLORS[motion], MARKERS[motion])
        label(c, legend_x + 2.2 * mm, 131.4 * mm, MOTION_LABELS[motion], 6.2)
        legend_x += {"high_dynamic": 22, "longitudinal": 21, "stable_straight": 25, "turning": 17}[motion] * mm
    left, right, bottom, top = 14 * mm, 92 * mm, 27 * mm, 108 * mm
    low, high = 2.2, 5.5
    axes(c, left, bottom, right, top, [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5], low, high)
    axes(c, left, bottom, right, top, [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5], low, high, vertical=True)
    c.saveState()
    c.setDash(2, 2)
    c.setStrokeColor(COLORS["muted"])
    c.line(left, bottom, right, top)
    c.restoreState()
    in_zoom = 0
    wins = 0
    for row in paired.itertuples(index=False):
        wins += int(row.full_pefnet < row.covariance_intersection)
        if low <= row.covariance_intersection <= high and low <= row.full_pefnet <= high:
            in_zoom += 1
            marker(c, sx(row.covariance_intersection, low, high, left, right), sy(row.full_pefnet, low, high, bottom, top), 0.85 * mm, COLORS[row.motion_type], MARKERS[row.motion_type], alpha=0.68)
    label(c, (left + right) / 2, 20.5 * mm, "Covariance Intersection position RMSE (m)", 7.0, align="center")
    c.saveState()
    c.translate(4.7 * mm, (bottom + top) / 2)
    c.rotate(90)
    label(c, 0, 0, "Full PEFNet position RMSE (m)", 7.0, align="center")
    c.restoreState()
    label(c, right - 1 * mm, bottom + 3 * mm, f"{wins}/200 scenes below y=x", 6.3, color=COLORS["muted"], align="right")
    label(c, right - 1 * mm, top - 3 * mm, f"Zoom: {in_zoom}/200 scenes", 6.1, color=COLORS["muted"], align="right")

    inset_left, inset_right, inset_bottom, inset_top = 18 * mm, 43 * mm, 80 * mm, 105 * mm
    c.setFillColor(white)
    c.setStrokeColor(COLORS["muted"])
    c.rect(inset_left, inset_bottom, inset_right - inset_left, inset_top - inset_bottom, fill=1, stroke=1)
    c.saveState()
    c.setDash(1.2, 1.2)
    c.line(inset_left, inset_bottom, inset_right, inset_top)
    c.restoreState()
    for row in paired.itertuples(index=False):
        marker(c, sx(row.covariance_intersection, 2.0, 10.5, inset_left, inset_right), sy(row.full_pefnet, 2.0, 10.5, inset_bottom, inset_top), 0.36 * mm, COLORS[row.motion_type], MARKERS[row.motion_type], alpha=0.55)
    label(c, inset_left + 1 * mm, inset_top - 2.7 * mm, "Full range", 5.4, "Arial-Bold")

    # Panel (b): motion subgroups with scene-bootstrap confidence intervals.
    label(c, 101 * mm, 139 * mm, "(b) Motion subgroups", 9.2, "Arial-Bold")
    legend_positions = [(101, 132.5), (117, 132.5), (130, 132.5), (149, 132.5), (172, 132.5)]
    for method, (x_mm, y_mm) in zip(METHODS, legend_positions):
        marker(c, x_mm * mm, y_mm * mm, 1.05 * mm, COLORS[method], MARKERS[method])
        label(c, (x_mm + 1.8) * mm, (y_mm - 1.1) * mm, METHOD_LABELS[method], 5.8)
    left, right, bottom, top = 113 * mm, 177 * mm, 83 * mm, 125 * mm
    axes(c, left, bottom, right, top, [2.0, 2.7, 3.4, 4.1, 4.8], 2.0, 4.8)
    offsets = [-3.2, -1.6, 0.0, 1.6, 3.2]
    for group_index, motion in enumerate(MOTIONS):
        center_y = top - (group_index + 0.5) * (top - bottom) / len(MOTIONS)
        n = motion_stats[(motion, "full_pefnet")][3]
        label(c, left - 2 * mm, center_y - 1.7, f"{MOTION_LABELS[motion]} (n={n})", 6.2, align="right")
        for method, offset in zip(METHODS, offsets):
            mean, ci_low, ci_high, _ = motion_stats[(motion, method)]
            y = center_y + offset * mm
            x0, x1, xm = sx(ci_low, 2.0, 4.8, left, right), sx(ci_high, 2.0, 4.8, left, right), sx(mean, 2.0, 4.8, left, right)
            c.setStrokeColor(COLORS[method])
            c.setLineWidth(0.75)
            c.line(x0, y, x1, y)
            c.line(x0, y - 0.7 * mm, x0, y + 0.7 * mm)
            c.line(x1, y - 0.7 * mm, x1, y + 0.7 * mm)
            marker(c, xm, y, 0.9 * mm, COLORS[method], MARKERS[method])
    label(c, (left + right) / 2, 76.7 * mm, "Position RMSE (m), mean and 95% bootstrap CI", 6.7, align="center")

    # Panel (c): city subgroups; descriptive because motion mix differs by city.
    label(c, 101 * mm, 70 * mm, "(c) City subgroups (descriptive)", 9.2, "Arial-Bold")
    left, right, bottom, top = 113 * mm, 177 * mm, 17 * mm, 61 * mm
    axes(c, left, bottom, right, top, [2.4, 2.9, 3.4, 3.9, 4.4], 2.4, 4.4)
    for group_index, city in enumerate(CITIES):
        center_y = top - (group_index + 0.5) * (top - bottom) / len(CITIES)
        n = city_stats[(city, "full_pefnet")][3]
        label(c, left - 2 * mm, center_y - 1.7, f"{CITY_LABELS[city]} (n={n})", 6.2, align="right")
        full_mean, full_low, full_high, _ = city_stats[(city, "full_pefnet")]
        ci_mean, ci_low, ci_high, _ = city_stats[(city, "covariance_intersection")]
        full_y, ci_y = center_y + 1.4 * mm, center_y - 1.4 * mm
        for method, mean, low_ci, high_ci, y in [
            ("full_pefnet", full_mean, full_low, full_high, full_y),
            ("covariance_intersection", ci_mean, ci_low, ci_high, ci_y),
        ]:
            x0, x1, xm = sx(low_ci, 2.4, 4.4, left, right), sx(high_ci, 2.4, 4.4, left, right), sx(mean, 2.4, 4.4, left, right)
            c.setStrokeColor(COLORS[method])
            c.setLineWidth(0.8)
            c.line(x0, y, x1, y)
            c.line(x0, y - 0.65 * mm, x0, y + 0.65 * mm)
            c.line(x1, y - 0.65 * mm, x1, y + 0.65 * mm)
            marker(c, xm, y, 0.95 * mm, COLORS[method], MARKERS[method])
    label(c, (left + right) / 2, 10.7 * mm, "Position RMSE (m), mean and 95% bootstrap CI", 6.7, align="center")

    c.showPage()
    c.save()
    print(args.output)


if __name__ == "__main__":
    main()
