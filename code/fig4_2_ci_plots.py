"""Figure 4 concentration-index plots based on GDP ranking."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from config import (
    CI_ANALYSIS_ROOT,
    SERVICE_SCOPES,
    PROFILES,
    REGIONS,
    MEGA_LABEL,
    SUPER_LABEL,
    COMBINED_GROUP_NAME,
    HIGHLIGHT_YEARS,
    PLOT_DPI,
)
from utils.figure_data import figure_dir, read_ci_curve_points

YEAR_COLORS = {
    2014: "#084081", 2015: "#2b8cbe", 2016: "#4eb3d3", 2017: "#7bccc4",
    2018: "#a8ddb5", 2019: "#d9f0a3", 2020: "#fee391", 2021: "#fec44f",
    2022: "#fe9929", 2023: "#e69f00", 2024: "#d95f0e",
}


def _scope_profile() -> tuple[str, str]:
    if len(SERVICE_SCOPES) != 1 or len(PROFILES) != 1:
        raise RuntimeError("Figure 4 requires exactly one service scope and one speed profile")
    return SERVICE_SCOPES[0], PROFILES[0]


def _highlight(ci_df: pd.DataFrame) -> list[int]:
    if HIGHLIGHT_YEARS is not None:
        return [int(y) for y in HIGHLIGHT_YEARS]
    years = ci_df.dropna(subset=["CI"])["year"].astype(int).tolist()
    return [] if not years else [years[0]] if len(years) == 1 else [years[0], years[-1]]


def _save_trend(df: pd.DataFrame, out: Path, fixed_xlim=(0.15, 0.30)) -> None:
    x = df.sort_values("year").dropna(subset=["CI"]).copy()
    if x.empty:
        return
    years = x["year"].to_numpy(int)
    values = x["CI"].to_numpy(float)
    highlighted = set(_highlight(x))

    plt.figure(figsize=(2.6, 3.2))
    ax = plt.gca()
    ax.plot(values, years, color="gray", alpha=0.4, linewidth=2, zorder=4)
    ax.scatter(
        values,
        years,
        s=[80 if int(y) in highlighted else 50 for y in years],
        c=[YEAR_COLORS.get(int(y), "gray") if int(y) in highlighted else "gray" for y in years],
        zorder=5,
        edgecolors="white",
        linewidths=1,
    )
    for year, ci in zip(years, values):
        color = YEAR_COLORS.get(int(year), "#bdbdbd") if int(year) in highlighted else "#d9d9d9"
        ax.hlines(year, 0, ci, colors=color, linewidth=4, zorder=3)
    ax.invert_yaxis()
    ax.set_xlim(*fixed_xlim)
    offset = (fixed_xlim[1] - fixed_xlim[0]) * 0.035
    for year, ci in zip(years, values):
        if int(year) in highlighted:
            ax.text(ci + offset, year, f"CI={ci:.3f}", va="center", ha="left", fontsize=13)
    ax.set_xticks([])
    tick_years = list(dict.fromkeys([int(years[0]), int(years[min(3, len(years)-1)]), int(years[min(6, len(years)-1)]), int(years[-1])]))
    ax.set_yticks(tick_years)
    ax.tick_params(axis="y", labelsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out, dpi=PLOT_DPI, transparent=True, bbox_inches="tight")
    plt.close()


def _save_curve(points: pd.DataFrame, out: Path, annotation: str, xlabel: str = "") -> None:
    if points.empty:
        return
    plt.figure(figsize=(4, 4))
    for year, group in points.groupby("year"):
        group = group.sort_values("x")
        ci = pd.to_numeric(group["ci"], errors="coerce").dropna()
        ci_value = float(ci.iloc[0]) if len(ci) else np.nan
        plt.plot(
            pd.to_numeric(group["x"], errors="coerce"),
            pd.to_numeric(group["y"], errors="coerce"),
            linewidth=5,
            color=YEAR_COLORS.get(int(year), "#777777"),
            label=f"{year} (CI={ci_value:.3f})" if np.isfinite(ci_value) else str(year),
        )
    plt.plot([0, 1], [0, 1], "--", color="black", linewidth=1.2)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel("Cumulative share of accessibility", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(frameon=False, fontsize=15, loc="upper left")
    plt.text(0.98, 0.02, annotation, transform=ax.transAxes, ha="right", va="bottom", style="italic", fontsize=16)
    plt.tight_layout()
    plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


def main() -> None:
    scope, profile = _scope_profile()
    root = CI_ANALYSIS_ROOT / scope / profile
    out = figure_dir("Figure 4", "CI_plots")

    national = pd.read_csv(root / "acc_CI_results_by_GDP.csv", encoding="utf-8-sig")
    region = pd.read_csv(root / "acc_CI_results_by_GDP_region.csv", encoding="utf-8-sig")
    city_level = pd.read_csv(root / "acc_CI_results_by_GDP_city_level.csv", encoding="utf-8-sig")
    curves = read_ci_curve_points()

    _save_trend(national, out / "acc_CI_trend_by_GDP.png")

    national_curve = curves[curves["curve_family"].eq("national_gdp")]
    _save_curve(national_curve, out / "acc_concentration_curves_by_GDP.png", "National", "Cumulative share of population by SES")

    for region_name in REGIONS:
        points = curves[(curves["curve_family"].eq("region_gdp")) & (curves["group_name"].eq(region_name))]
        annotation = "Eastern China" if region_name == "Eastern" else "Non-Eastern China"
        _save_curve(points, out / f"acc_concentration_curves_by_GDP_{region_name}.png", annotation)

    preferred = [MEGA_LABEL, SUPER_LABEL, COMBINED_GROUP_NAME, "Large City", "Medium/Small City"]
    groups = [g for g in preferred if g in set(city_level["city_level"].dropna().astype(str))]
    display = {
        MEGA_LABEL: "Mega Cities",
        SUPER_LABEL: "Super Cities",
        COMBINED_GROUP_NAME: "Mega & Super Cities",
        "Large City": "Large Cities",
        "Medium/Small City": "Medium & Small Cities",
    }
    for group_name in groups:
        points = curves[(curves["curve_family"].eq("citylevel_gdp")) & (curves["group_name"].eq(group_name))]
        safe = group_name.replace("/", "_").replace(" ", "_")
        _save_curve(points, out / f"acc_concentration_curves_by_GDP_city_level_{safe}.png", display.get(group_name, group_name))

    print(f"Figure 4 CI plots -> {out}")


if __name__ == "__main__":
    main()
