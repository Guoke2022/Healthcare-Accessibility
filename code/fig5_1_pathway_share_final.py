# -*- coding: utf-8 -*-


from __future__ import annotations

import pandas as pd

from config import (
    CHANGE_YEARS,
    SEE_CIE_ANNUAL_ROOT,
    SEE_CIE_DESCRIPTIVE_ROOT,
    FIGURE5_ROOT,
    CITY_LEVEL_MERGE_MAP,
    CITY_ORDER_4,
    MAKE_PLOTS,
    PLOT_DPI,
)
from utils.extended_analysis import read_csv_robust, set_nature_style

LEVEL_LABELS = [
    "Small & Medium\nCities",
    "Large Cities",
    "Super Cities",
    "Mega Cities",
]

OUTPUT_NAME = "Fig_SEE_CIE_pathway_share_citysize_v2.png"
BOX_COLOR = "#f3d7a3"


def load_city_panel() -> pd.DataFrame:
    frames = []
    for year in CHANGE_YEARS:
        df = read_csv_robust(SEE_CIE_ANNUAL_ROOT / f"city_SEE_CIE_{year}.csv")
        df["city_level"] = df["city_level"].replace(CITY_LEVEL_MERGE_MAP)
        df["year"] = year
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    return panel.loc[panel["city_level"].isin(CITY_ORDER_4)].copy()


def build_city_pathway_share(panel: pd.DataFrame) -> pd.DataFrame:
    city = (
        panel.groupby(["省级", "地级", "city_level"], as_index=False)
        .agg(
            see_added_beds=("new_hosp_beds", "sum"),
            cie_added_beds=("expanded_beds", "sum"),
        )
    )

    city["see_added_beds"] = pd.to_numeric(city["see_added_beds"], errors="coerce").fillna(0)
    city["cie_added_beds"] = pd.to_numeric(city["cie_added_beds"], errors="coerce").fillna(0)
    city["gross_added_beds"] = city["see_added_beds"] + city["cie_added_beds"]

    city = city.loc[city["gross_added_beds"] > 0].copy()
    city["SEE_share_pct"] = city["see_added_beds"] / city["gross_added_beds"] * 100.0
    city["CIE_share_pct"] = 100.0 - city["SEE_share_pct"]
    return city


def export_tables(city: pd.DataFrame) -> None:
    SEE_CIE_DESCRIPTIVE_ROOT.mkdir(parents=True, exist_ok=True)
    city.to_csv(
        SEE_CIE_DESCRIPTIVE_ROOT / "city_level_SEE_CIE_share_2015_2024.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = (
        city.groupby("city_level")
        .agg(
            n_city=("地级", "count"),
            median_SEE_share=("SEE_share_pct", "median"),
            q1_SEE_share=("SEE_share_pct", lambda x: x.quantile(0.25)),
            q3_SEE_share=("SEE_share_pct", lambda x: x.quantile(0.75)),
            mean_SEE_share=("SEE_share_pct", "mean"),
            median_CIE_share=("CIE_share_pct", "median"),
        )
        .reindex(CITY_ORDER_4)
        .reset_index()
    )
    summary.to_csv(
        SEE_CIE_DESCRIPTIVE_ROOT / "city_level_SEE_CIE_share_summary_2015_2024.csv",
        index=False,
        encoding="utf-8-sig",
    )


def make_plot(city: pd.DataFrame) -> None:
    if not MAKE_PLOTS:
        return

    plt = set_nature_style()
    FIGURE5_ROOT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    pos = list(range(len(CITY_ORDER_4)))
    data = [
        pd.to_numeric(
            city.loc[city["city_level"] == level, "SEE_share_pct"],
            errors="coerce",
        ).dropna().values
        for level in CITY_ORDER_4
    ]

    bp = ax.boxplot(
        data,
        positions=pos,
        widths=0.2,
        showfliers=False,
        patch_artist=True,
        whis=1.5,
        boxprops=dict(
            facecolor=BOX_COLOR,
            edgecolor="black",
            linewidth=0.9,
            alpha=0.85,
            zorder=1,
        ),
        medianprops=dict(color="black", linewidth=1.1, zorder=4),
        whiskerprops=dict(color="black", linewidth=1.0, zorder=4),
        capprops=dict(color="black", linewidth=1.0, zorder=4),
    )

    # Draw the 50% reference line without text annotation; explain in caption.
    ax.axhline(50, color="#666666", linewidth=0.8, linestyle="--", zorder=0)


    # Re-raise line artists above the boxes for maximum visibility.
    for artist_group in ["medians", "whiskers", "caps"]:
        for artist in bp[artist_group]:
            artist.set_zorder(5)


    # Add median labels above each boxplot.
    for i, values in enumerate(data):
        values = pd.Series(values).dropna()
        if len(values) == 0:
            continue

        median_val = values.median()

        ax.text(
            pos[i],
            102,
            f"Median={median_val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
        )


    ax.set_xticks(pos)
    ax.set_xticklabels(LEVEL_LABELS)

    ax.set_xticks(pos)
    ax.set_xticklabels(LEVEL_LABELS)
    ax.set_ylabel("SEE share of added bed capacity (%)")
    ax.set_xlabel("")
    ax.set_ylim(15, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_xlim(-0.5, len(CITY_ORDER_4) - 0.5)
    ax.tick_params(axis="x", pad=4)

    fig.tight_layout()
    out = FIGURE5_ROOT / OUTPUT_NAME
    fig.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 5a done -> {out}")


def main() -> None:
    panel = load_city_panel()
    city = build_city_pathway_share(panel)
    export_tables(city)
    make_plot(city)


if __name__ == "__main__":
    main()
