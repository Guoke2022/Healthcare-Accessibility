# -*- coding: utf-8 -*-
"""Render the concentration-index panels used in Figure 4.

Only GDP-ranked concentration curves are rendered. The CI trend output is the
national GDP-ranked trend used in the manuscript.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from config import CI_ANALYSIS_ROOT, SERVICE_SCOPES, PROFILES, PLOT_DPI
from utils.figure_data import figure_dir, read_ci_curve_points

YEAR_COLORS = {
    2014: "#084081", 2015: "#2b8cbe", 2016: "#4eb3d3", 2017: "#7bccc4",
    2018: "#a8ddb5", 2019: "#d9f0a3", 2020: "#fee391", 2021: "#fec44f",
    2022: "#fe9929", 2023: "#e69f00", 2024: "#d95f0e",
}
CITY_DISPLAY = {
    "Mega City": "Mega Cities",
    "Super City": "Super Cities",
    "Mega/Super City": "Mega & Super Cities",
    "Large City": "Large Cities",
    "Medium/Small City": "Medium & Small Cities",
}


def _scope_profile() -> tuple[str, str]:
    if len(SERVICE_SCOPES) != 1 or len(PROFILES) != 1:
        raise RuntimeError("Figure 4 requires exactly one service scope and one speed profile")
    return SERVICE_SCOPES[0], PROFILES[0]


def _save_trend(df: pd.DataFrame, out_path) -> None:
    x = df.sort_values("year").dropna(subset=["CI"]).copy()
    if x.empty:
        raise ValueError("The national GDP-ranked CI table contains no valid values")

    years = x["year"].to_numpy(int)
    values = x["CI"].to_numpy(float)
    highlighted = {int(years[0]), int(years[-1])}

    plt.figure(figsize=(2.6, 3.2))
    ax = plt.gca()
    ax.plot(values, years, color="gray", alpha=0.4, linewidth=2, zorder=4)
    ax.scatter(
        values,
        years,
        s=[80 if int(y) in highlighted else 50 for y in years],
        c=[YEAR_COLORS.get(int(y), "gray") if int(y) in highlighted else "gray" for y in years],
        edgecolors="white",
        linewidths=1,
        zorder=5,
    )

    for year, value in zip(years, values):
        color = YEAR_COLORS.get(int(year), "#bdbdbd") if int(year) in highlighted else "#d9d9d9"
        ax.hlines(year, 0, value, colors=color, linewidth=4, zorder=3)

    ax.invert_yaxis()
    ax.set_xlim(0.15, 0.30)
    xmin, xmax = ax.get_xlim()
    text_offset = (xmax - xmin) * 0.035
    for year, value in zip(years, values):
        if int(year) in highlighted:
            ax.text(value + text_offset, year, f"CI={value:.3f}", va="center", ha="left", fontsize=13)

    tick_years = list(dict.fromkeys([int(years[0]), int(years[min(3, len(years)-1)]), int(years[min(6, len(years)-1)]), int(years[-1])]))
    ax.set_xticks([])
    ax.set_yticks(tick_years)
    ax.tick_params(axis="y", labelsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI, transparent=True, bbox_inches="tight")
    plt.close()


def _save_curve(points: pd.DataFrame, out_path, annotation: str) -> None:
    if points.empty:
        return

    plt.figure(figsize=(4, 4))
    for year, group in points.groupby("year"):
        group = group.sort_values("x")
        ci_values = pd.to_numeric(group["ci"], errors="coerce").dropna()
        ci_value = float(ci_values.iloc[0]) if len(ci_values) else np.nan
        plt.plot(
            pd.to_numeric(group["x"], errors="coerce"),
            pd.to_numeric(group["y"], errors="coerce"),
            linewidth=5,
            color=YEAR_COLORS.get(int(year), "#777777"),
            label=f"{year} (CI={ci_value:.3f})" if np.isfinite(ci_value) else str(year),
        )

    plt.plot([0, 1], [0, 1], "--", color="black", linewidth=1.2)
    plt.xlabel("Cumulative share of population ranked by GDP per capita", fontsize=18)
    plt.ylabel("Cumulative share of accessibility", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(frameon=False, fontsize=15, loc="upper left")
    plt.text(0.98, 0.02, annotation, transform=ax.transAxes, ha="right", va="bottom", style="italic", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


def main() -> None:
    scope, profile = _scope_profile()
    analysis_root = CI_ANALYSIS_ROOT / scope / profile
    output_dir = figure_dir("Figure 4", "CI_plots")

    national = pd.read_csv(analysis_root / "acc_CI_results_by_GDP.csv", encoding="utf-8-sig")
    _save_trend(national, output_dir / "acc_CI_trend_by_GDP.png")

    points = read_ci_curve_points()

    national_points = points[points["curve_family"].astype(str).eq("national_gdp")].copy()
    _save_curve(national_points, output_dir / "acc_concentration_curves_by_GDP.png", "National")

    region_points = points[points["curve_family"].astype(str).eq("region_gdp")].copy()
    for region, group in region_points.groupby("group_name"):
        safe = str(region).replace("/", "_").replace(" ", "_")
        annotation = "Eastern China" if str(region) == "Eastern" else "Non-Eastern China"
        _save_curve(group, output_dir / f"acc_concentration_curves_by_GDP_region_{safe}.png", annotation)

    city_points = points[points["curve_family"].astype(str).eq("citylevel_gdp")].copy()
    for city_level, group in city_points.groupby("group_name"):
        safe = str(city_level).replace("/", "_").replace(" ", "_")
        _save_curve(
            group,
            output_dir / f"acc_concentration_curves_by_GDP_city_level_{safe}.png",
            CITY_DISPLAY.get(str(city_level), str(city_level)),
        )

    print(f"Figure 4 CI plots written to {output_dir}")


if __name__ == "__main__":
    main()
