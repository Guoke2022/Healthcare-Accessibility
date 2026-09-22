"""Figure 5a: city-level share of added bed capacity attributable to SEE."""
from __future__ import annotations

import pandas as pd

from config import BASE_YEAR, END_YEAR, SEE_CIE_PANEL_ROOT, FIGURE5_ROOT, CITY_ORDER_4, PLOT_DPI
from utils.extended_analysis import read_csv_robust, set_nature_style

LEVEL_LABELS = ["Small & Medium\nCities", "Large Cities", "Super Cities", "Mega Cities"]
OUTPUT_NAME = "Fig_SEE_CIE_pathway_share_citysize_v2.png"
BOX_COLOR = "#f3d7a3"


def load_city_data() -> pd.DataFrame:
    path = SEE_CIE_PANEL_ROOT / f"city_{BASE_YEAR}_{END_YEAR}_index.csv"
    df = read_csv_robust(path)
    df = df[df["city_level"].isin(CITY_ORDER_4)].copy()
    for col in ["new_hosp_beds", "expanded_beds"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["gross_added_beds"] = df["new_hosp_beds"] + df["expanded_beds"]
    df = df[df["gross_added_beds"] > 0].copy()
    df["SEE_share_pct"] = df["new_hosp_beds"] / df["gross_added_beds"] * 100.0
    return df


def main() -> None:
    city = load_city_data()
    plt = set_nature_style()
    FIGURE5_ROOT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    positions = list(range(len(CITY_ORDER_4)))
    data = [
        pd.to_numeric(city.loc[city["city_level"] == level, "SEE_share_pct"], errors="coerce").dropna().values
        for level in CITY_ORDER_4
    ]
    boxplot = ax.boxplot(
        data,
        positions=positions,
        widths=0.2,
        showfliers=False,
        patch_artist=True,
        whis=1.5,
        boxprops=dict(facecolor=BOX_COLOR, edgecolor="black", linewidth=0.9, alpha=0.85),
        medianprops=dict(color="black", linewidth=1.1),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0),
    )
    ax.axhline(50, color="#666666", linewidth=0.8, linestyle="--", zorder=0)
    for group in ["medians", "whiskers", "caps"]:
        for artist in boxplot[group]:
            artist.set_zorder(5)

    for i, values in enumerate(data):
        values = pd.Series(values).dropna()
        if len(values):
            ax.text(i, 102, f"Median={values.median():.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(positions)
    ax.set_xticklabels(LEVEL_LABELS)
    ax.set_ylabel("SEE share of added bed capacity (%)")
    ax.set_ylim(15, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_xlim(-0.5, len(CITY_ORDER_4) - 0.5)
    fig.tight_layout()

    out = FIGURE5_ROOT / OUTPUT_NAME
    fig.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
