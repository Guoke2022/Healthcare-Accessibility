from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

from utils.figure_data import figure_dir, read_ci_kde_curves, read_ci_year

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.options.mode.chained_assignment = None

LABEL_FONTSIZE = 24
TICKS_FONTSIZE = 20
FIGURE_SIZE = (10, 5)

YEAR_COLORS = {
    2014: "#084081", 2015: "#2b8cbe", 2016: "#4eb3d3", 2017: "#7bccc4",
    2018: "#a8ddb5", 2019: "#d9f0a3", 2020: "#fee391", 2021: "#fec44f",
    2022: "#fe9929", 2023: "#e69f00", 2024: "#cc4c02",
}
HIGHLIGHT_YEARS = {2014, 2024}


def _style_and_save(out_path, xlabel: str, *, gdp_axis: bool = False) -> None:
    ax = plt.gca()
    if gdp_axis:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 10000:.0f}"))
    plt.xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICKS_FONTSIZE)
    plt.ylabel("Density", fontsize=LABEL_FONTSIZE)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=600, transparent=True, bbox_inches="tight")
    plt.close()


def _plot_precomputed(variable: str, out_path, xlabel: str, *, gdp_axis: bool = False) -> None:
    curves = read_ci_kde_curves(variable)
    if curves.empty:
        raise ValueError(f"No precomputed KDE curve found for {variable!r}")

    plt.figure(figsize=FIGURE_SIZE)
    for year in range(2014, 2025):
        z = curves[pd.to_numeric(curves["year"], errors="coerce").eq(year)].copy()
        if z.empty:
            continue
        z = z.sort_values("x")
        highlighted = year in HIGHLIGHT_YEARS
        plt.plot(
            pd.to_numeric(z["x"], errors="coerce"),
            pd.to_numeric(z["density"], errors="coerce"),
            label=str(year),
            color=YEAR_COLORS.get(year, "#bdbdbd"),
            linewidth=5 if highlighted else 2,
            linestyle="-" if highlighted else "--",
            alpha=1.0,
            zorder=10 if highlighted else 3,
        )
    _style_and_save(out_path, xlabel, gdp_axis=gdp_axis)


def _plot_from_private_matched(variable: str, out_path, xlabel: str, *, gdp_axis: bool = False) -> None:
    """Fallback used only in the private reconstruction workflow."""
    if variable == "GDP_per":
        columns, bw = ["GDP_per", "pop"], 5.0
        valid = lambda d: d[(d["GDP_per"] > 0) & (d["pop"] > 0)].copy()
    elif variable == "acc":
        columns, bw = ["acc", "pop"], 4.0
        valid = lambda d: d[(d["pop"] > 0) & (d["acc"] < 90)].copy()
    elif variable == "Minority_rate":
        columns, bw = ["Minority_rate", "pop"], 4.0
        valid = lambda d: d[(d["Minority_rate"] >= 0) & (d["pop"] > 0)].copy()
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
        highlighted = year in HIGHLIGHT_YEARS
        sns.kdeplot(
            data=df,
            x=variable,
            weights="pop",
            clip=(0, float(pd.to_numeric(df[variable], errors="coerce").max())),
            label=str(year),
            bw_adjust=bw,
            color=YEAR_COLORS.get(year, "#bdbdbd"),
            linewidth=5 if highlighted else 2,
            linestyle="-" if highlighted else "--",
            alpha=1.0,
            zorder=10 if highlighted else 3,
        )
    _style_and_save(out_path, xlabel, gdp_axis=gdp_axis)


def _plot(variable: str, out_path, xlabel: str, *, gdp_axis: bool = False) -> None:
    try:
        _plot_precomputed(variable, out_path, xlabel, gdp_axis=gdp_axis)
    except FileNotFoundError:
        _plot_from_private_matched(variable, out_path, xlabel, gdp_axis=gdp_axis)


def main() -> None:
    output_dir = figure_dir("Figure 4", "KDE")

    _plot(
        "GDP_per",
        output_dir / "GDP_KDE_2014_2024.png",
        "GDP per capita (10⁴ RMB)",
        gdp_axis=True,
    )
    _plot(
        "acc",
        output_dir / "accessibility_KDE_2014_2024.png",
        "Accessibility",
    )
    _plot(
        "Minority_rate",
        output_dir / "minority_share_KDE_2014_2024.png",
        "Ethnic minority population proportion (%)",
    )


if __name__ == "__main__":
    main()
