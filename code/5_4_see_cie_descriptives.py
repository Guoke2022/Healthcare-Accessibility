# -*- coding: utf-8 -*-
"""Prepare the compact descriptive table used by Figure 5.2.

This stage operates on released annual city-level SEE/CIE tables. It does not
render figures; manuscript plotting is handled by dedicated ``fig5_*`` scripts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import BASE_YEAR, CHANGE_YEARS, SEE_CIE_ANNUAL_ROOT, SEE_CIE_DESCRIPTIVE_ROOT, CITY_LEVEL_MERGE_MAP, CITY_ORDER_4
from utils.extended_analysis import read_csv_robust, read_stats


def load_city_panel() -> pd.DataFrame:
    frames = []
    for year in CHANGE_YEARS:
        df = read_csv_robust(SEE_CIE_ANNUAL_ROOT / f"city_SEE_CIE_{year}.csv")
        df["city_level"] = df["city_level"].replace(CITY_LEVEL_MERGE_MAP)
        df["year"] = year
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    return panel[panel["city_level"].isin(CITY_ORDER_4)].copy()


def build_accessibility_change_table(panel: pd.DataFrame) -> pd.DataFrame:
    absolute_gain = (
        panel.groupby(["地级", "city_level"], as_index=False)["acc_delta"]
        .sum()
        .rename(columns={"acc_delta": "abs_gain"})
    )

    baseline = read_stats("accessibility", "city")
    baseline = (
        baseline.loc[baseline["Year"] == BASE_YEAR, ["地级", "pop_median"]]
        .drop_duplicates(subset=["地级"])
        .rename(columns={"pop_median": "base_acc_2014"})
    )

    out = absolute_gain.merge(baseline, on="地级", how="left")
    out["rel_gain_pct"] = np.where(
        out["base_acc_2014"].notna() & (out["base_acc_2014"] != 0),
        out["abs_gain"] / out["base_acc_2014"] * 100.0,
        np.nan,
    )

    values = pd.to_numeric(out["rel_gain_pct"], errors="coerce").dropna().to_numpy()
    if len(values):
        lower = float(np.quantile(values, 0.01))
        upper = float(np.quantile(values, 0.95))
        out["rel_plot"] = pd.to_numeric(out["rel_gain_pct"], errors="coerce").clip(lower=lower, upper=upper)
        out["rel_plot_clip_lower"] = lower
        out["rel_plot_clip_upper"] = upper
    else:
        out["rel_plot"] = np.nan
        out["rel_plot_clip_lower"] = np.nan
        out["rel_plot_clip_upper"] = np.nan

    return out


def build_group_summary(table: pd.DataFrame) -> pd.DataFrame:
    return (
        table.groupby("city_level", as_index=False)
        .agg(
            n=("abs_gain", "count"),
            absolute_mean=("abs_gain", "mean"),
            absolute_median=("abs_gain", "median"),
            relative_mean=("rel_gain_pct", "mean"),
            relative_median=("rel_gain_pct", "median"),
        )
    )


def main() -> None:
    SEE_CIE_DESCRIPTIVE_ROOT.mkdir(parents=True, exist_ok=True)
    panel = load_city_panel()
    change = build_accessibility_change_table(panel)
    change.to_csv(
        SEE_CIE_DESCRIPTIVE_ROOT / "city_accessibility_absolute_relative_change.csv",
        index=False,
        encoding="utf-8-sig",
    )
    build_group_summary(change).to_csv(
        SEE_CIE_DESCRIPTIVE_ROOT / "city_level_accessibility_change_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"SEE/CIE descriptive tables written to {SEE_CIE_DESCRIPTIVE_ROOT}")


if __name__ == "__main__":
    main()
