from __future__ import annotations

import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

from config import PLOT_DPI
from utils.figure_data import figure_dir, read_ci_kde_curves, read_ci_year

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.options.mode.chained_assignment = None

LABEL_FONTSIZE = 24
TICKS_FONTSIZE = 20
FIGURE_SIZE = (10, 5)

year_color = {
    2014: "#084081", 2015: "#2b8cbe", 2016: "#4eb3d3", 2017: "#7bccc4",
    2018: "#a8ddb5", 2019: "#d9f0a3", 2020: "#fee391", 2021: "#fec44f",
    2022: "#fe9929", 2023: "#e69f00", 2024: "#cc4c02",
}
HIGHLIGHT_YEARS = {2014, 2024}


def _style_and_save(out_path: str, xlabel: str, *, gdp_axis: bool = False) -> None:
    ax = plt.gca()
    if gdp_axis:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/10000:.0f}"))
    plt.xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICKS_FONTSIZE)
    plt.ylabel("Density", fontsize=LABEL_FONTSIZE)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI, transparent=True, bbox_inches="tight")
    plt.close()


def _plot_precomputed(variable: str, out_path: str, xlabel: str, *, gdp_axis: bool = False) -> None:
    curves = read_ci_kde_curves(variable)
    if curves.empty:
        raise ValueError(f"预计算 KDE 文件中没有 variable={variable!r}")

    plt.figure(figsize=FIGURE_SIZE)
    for year in range(2014, 2025):
        z = curves[pd.to_numeric(curves["year"], errors="coerce").eq(year)].copy()
        if z.empty:
            continue
        z = z.sort_values("x")
        is_highlight = year in HIGHLIGHT_YEARS
        plt.plot(
            pd.to_numeric(z["x"], errors="coerce"),
            pd.to_numeric(z["density"], errors="coerce"),
            label=str(year),
            color=year_color.get(year, "#bdbdbd"),
            linewidth=5 if is_highlight else 2,
            linestyle="-" if is_highlight else "--",
            alpha=1.0,
            zorder=10 if is_highlight else 3,
        )
    _style_and_save(out_path, xlabel, gdp_axis=gdp_axis)


def _plot_from_private_matched(variable: str, out_path: str, xlabel: str, *, gdp_axis: bool = False) -> None:
    """Private/HPC fallback matching the historical Figure 4 KDE definitions."""
    if variable == "GDP_per":
        columns, bw = ["GDP_per", "pop"], 5.0
        valid = lambda d: d[(d["GDP_per"] > 0) & (d["pop"] > 0)].copy()
        clip_max = lambda d: d["GDP_per"].max()
    elif variable == "acc":
        columns, bw = ["acc", "pop"], 4.0
        valid = lambda d: d[(d["pop"] > 0) & (d["acc"] < 90)].copy()
        clip_max = lambda d: d["acc"].max()
    elif variable == "Minority_rate":
        columns, bw = ["Minority_rate", "pop"], 4.0
        valid = lambda d: d[(d["Minority_rate"] >= 0) & (d["pop"] > 0)].copy()
        clip_max = lambda d: d["Minority_rate"].max()
    else:
        raise ValueError(variable)

    plt.figure(figsize=FIGURE_SIZE)
    for year in range(2014, 2025):
        try:
            df = read_ci_year(year, columns)
        except FileNotFoundError:
            continue
        df = valid(df)
        if df.empty:
            continue
        is_highlight = year in HIGHLIGHT_YEARS
        sns.kdeplot(
            data=df,
            x=variable,
            weights="pop",
            clip=(0, clip_max(df)),
            label=str(year),
            bw_adjust=bw,
            color=year_color.get(year, "#bdbdbd"),
            linewidth=5 if is_highlight else 2,
            linestyle="-" if is_highlight else "--",
            alpha=1.0,
            zorder=10 if is_highlight else 3,
        )
    _style_and_save(out_path, xlabel, gdp_axis=gdp_axis)


def _plot(variable: str, out_path: str, xlabel: str, *, gdp_axis: bool = False) -> None:
    # Public reproduction intentionally reads small released curves.  The fallback
    # keeps the private/HPC project backward-compatible when those curves have not
    # yet been generated.
    try:
        _plot_precomputed(variable, out_path, xlabel, gdp_axis=gdp_axis)
    except FileNotFoundError:
        _plot_from_private_matched(variable, out_path, xlabel, gdp_axis=gdp_axis)


def main() -> None:
    kde_dir = str(figure_dir("Figure 4", "KDE"))
    os.makedirs(kde_dir, exist_ok=True)

    _plot(
        "GDP_per",
        os.path.join(kde_dir, "GDP_KDE_2014_2024.png"),
        "GDP per capita (10⁴ RMB)",
        gdp_axis=True,
    )
    _plot(
        "acc",
        os.path.join(kde_dir, "accessibility_KDE_2014_2024.png"),
        "Accessibility",
    )
    _plot(
        "Minority_rate",
        os.path.join(kde_dir, "minority_share_KDE_2014_2024.png"),
        "Ethnic minority population prop. (%)",
    )


if __name__ == "__main__":
    main()
