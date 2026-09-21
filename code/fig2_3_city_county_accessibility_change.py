# -*- coding: utf-8 -*-
"""Figure 2c-e: changes in median Ga2SFCA accessibility, 2014-2024.

The original plotting script merged annual province/city/county statistics by
administrative-unit *name*.  That is unsafe for counties because many counties
share the same name across China (for example, 市中区 or 鼓楼区), which can create
many-to-many Cartesian matches across years.  Null administrative names can
also match each other in ``pandas.merge`` and be counted as a pseudo-unit.

This version uses stable administrative codes for province and county units,
and the exact ``city_name_norm`` name universe used by the upstream city-level
aggregation for city units.  All joins are validated as one-to-one, invalid
placeholder units are excluded before counting, and map layers are constructed
with the same analysis-unit definitions as the statistics.
"""

from __future__ import annotations

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import PLOT_DPI
from utils.figure_data import figure_dir, load_admin_level, read_stats

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.options.mode.chained_assignment = None
pd.set_option("display.float_format", "{:.2f}".format)


LABEL_FONTSIZE = 25
TICKS_FONTSIZE = 45
LEGEND_FONTSIZE = 50

COLOR_N = "#ffeaad"
COLOR_P = "#b1c1da"

BASE_YEAR = 2014
END_YEAR = 2024


def _clean_name(series: pd.Series) -> pd.Series:
    """Normalize administrative-unit names used as analysis keys."""
    out = series.astype("string[python]").str.strip()
    return out.mask(out.eq("").fillna(False))


def _clean_code(series: pd.Series) -> pd.Series:
    """Normalize six-digit administrative codes to pandas StringDtype."""
    out = series.astype("string[python]").str.strip().str.replace(r"\.0$", "", regex=True)
    # zfill leaves <NA> as <NA> under pandas StringDtype.
    return out.str.zfill(6)


def _assert_unique_key(df: pd.DataFrame, key: str, scale_name: str, year: int) -> None:
    dup = df[key].duplicated(keep=False)
    if dup.any():
        cols = [c for c in [key, "省级", "地级", "县级", "pop_median"] if c in df.columns]
        raise RuntimeError(
            f"{scale_name} {year} statistics contain duplicate analysis keys; "
            "a one-to-one 2014/2024 comparison is not possible:\n"
            f"{df.loc[dup, cols].head(40).to_string(index=False)}"
        )


def _report_unmatched_keys(
    keys_2014: set[str],
    keys_2024: set[str],
    *,
    scale_name: str,
) -> None:
    only_2014 = sorted(keys_2014 - keys_2024)
    only_2024 = sorted(keys_2024 - keys_2014)
    if only_2014 or only_2024:
        print(
            f"{scale_name}: comparison uses units present in both {BASE_YEAR} and {END_YEAR}. "
            f"{BASE_YEAR}-only={len(only_2014)}, {END_YEAR}-only={len(only_2024)}"
        )
        if only_2014:
            print(f"  {BASE_YEAR}-only keys (sample): {only_2014[:20]}")
        if only_2024:
            print(f"  {END_YEAR}-only keys (sample): {only_2024[:20]}")


def _prepare_code_change_frame(
    *,
    group: str,
    name_col: str,
    code_col: str,
    scale_name: str,
) -> pd.DataFrame:
    """Build a safe 2014/2024 change table keyed by six-digit admin code."""
    raw = read_stats(group, "accessibility")[["Year", name_col, code_col, "pop_median"]].copy()
    raw[name_col] = _clean_name(raw[name_col])
    raw[code_col] = _clean_code(raw[code_col])

    # Build a plain NumPy-backed boolean mask explicitly.  This avoids
    # ``pd.NA`` ambiguity under pandas versions that infer Arrow-backed
    # string/boolean dtypes from CSV input.
    name_valid = raw[name_col].notna().fillna(False).astype(bool)
    code_format_valid = (
        raw[code_col]
        .str.fullmatch(r"\d{6}", na=False)
        .fillna(False)
        .astype(bool)
    )
    code_nonzero = raw[code_col].ne("000000").fillna(False).astype(bool)
    valid = name_valid & code_format_valid & code_nonzero
    if (~valid).any():
        bad = raw.loc[~valid, ["Year", name_col, code_col, "pop_median"]].copy()
        print(
            f"{scale_name}: dropping {int((~valid).sum())} invalid/placeholder statistics row(s) "
            "before 2014/2024 comparison:"
        )
        print(bad.to_string(index=False))
    raw = raw.loc[valid].copy()

    left = raw.loc[raw["Year"].eq(BASE_YEAR), [code_col, name_col, "pop_median"]].copy()
    right = raw.loc[raw["Year"].eq(END_YEAR), [code_col, name_col, "pop_median"]].copy()
    _assert_unique_key(left, code_col, scale_name, BASE_YEAR)
    _assert_unique_key(right, code_col, scale_name, END_YEAR)

    _report_unmatched_keys(
        set(left[code_col].astype(str)),
        set(right[code_col].astype(str)),
        scale_name=scale_name,
    )

    merged = left.merge(
        right,
        on=code_col,
        how="inner",
        validate="one_to_one",
        suffixes=("_2014", "_2024"),
    )

    # Administrative names may change while codes remain stable.  Keep the 2024
    # name for human-readable output, but report such changes for auditability.
    name_2014 = f"{name_col}_2014"
    name_2024 = f"{name_col}_2024"
    renamed = merged[name_2014].ne(merged[name_2024])
    if renamed.any():
        print(f"{scale_name}: {int(renamed.sum())} matched code(s) changed name between years:")
        print(
            merged.loc[renamed, [code_col, name_2014, name_2024]]
            .head(30)
            .to_string(index=False)
        )

    out = merged.rename(
        columns={
            name_2024: name_col,
            "pop_median_2014": "2014_median",
            "pop_median_2024": "2024_median",
        }
    )[[code_col, name_col, "2014_median", "2024_median"]].copy()
    out["24-14"] = out["2024_median"] - out["2014_median"]
    return out


def _prepare_city_change_frame() -> pd.DataFrame:
    """Build the city change table using the upstream ``city_name_norm`` key.

    ``city_acc_stats.csv`` is aggregated upstream by ``city_name_norm`` and then
    written with that key renamed to ``地级``.  ``地级码`` is not a unique key for
    all analysis units: municipalities and directly administered county-level
    units can legitimately carry code 0.  Therefore city comparisons must use
    this normalized analysis-unit name, after excluding the null placeholder.
    """
    raw = read_stats("city", "accessibility")[["Year", "地级", "地级码", "pop_median"]].copy()
    raw["地级"] = _clean_name(raw["地级"])

    valid = raw["地级"].notna()
    if (~valid).any():
        bad = raw.loc[~valid, ["Year", "地级", "地级码", "pop_median"]].copy()
        print(
            f"City: dropping {int((~valid).sum())} null/blank city analysis row(s) "
            "before 2014/2024 comparison:"
        )
        print(bad.to_string(index=False))
    raw = raw.loc[valid].copy()

    left = raw.loc[raw["Year"].eq(BASE_YEAR), ["地级", "pop_median"]].copy()
    right = raw.loc[raw["Year"].eq(END_YEAR), ["地级", "pop_median"]].copy()
    _assert_unique_key(left, "地级", "City", BASE_YEAR)
    _assert_unique_key(right, "地级", "City", END_YEAR)

    _report_unmatched_keys(
        set(left["地级"].astype(str)),
        set(right["地级"].astype(str)),
        scale_name="City",
    )

    out = left.merge(
        right,
        on="地级",
        how="inner",
        validate="one_to_one",
        suffixes=("_2014", "_2024"),
    ).rename(
        columns={
            "pop_median_2014": "2014_median",
            "pop_median_2024": "2024_median",
        }
    )
    out["24-14"] = out["2024_median"] - out["2014_median"]
    return out[["地级", "2014_median", "2024_median", "24-14"]].copy()


def _load_city_analysis_geometry():
    """Construct city polygons with the same key rules as ``add_city_level``.

    The released city shapefile stores municipalities/directly-administered units
    with ``地级='不统计'``.  The statistical city universe, however, was generated
    upstream from ``city_name_norm``.  Reconstructing polygons from county
    boundaries keeps the map geometry synchronized with that analysis universe.
    """
    county = load_admin_level("county").copy()
    required = ["县级", "地级", "省级"]
    missing = [col for col in required if col not in county.columns]
    if missing:
        raise KeyError(f"City geometry is missing required county-boundary columns: {missing}")

    for col in required:
        county[col] = _clean_name(county[col])

    city_key = county["地级"].copy()

    # Keep these rules synchronized with add_city_level() in
    # code/2_1_multiscale_match.py and Fig.1's city geometry builder.
    fallback_to_province = (
        city_key.isna().fillna(False).astype(bool)
        | city_key.eq("不统计").fillna(False).astype(bool)
    )
    city_key = city_key.mask(fallback_to_province, county["省级"])

    directly_administered = city_key.isin(["海南省", "湖北省"])
    city_key = city_key.mask(directly_administered, county["县级"])

    still_missing = city_key.isna()
    city_key = city_key.mask(still_missing, county["县级"])

    tmp = county.copy()
    tmp["地级"] = city_key
    tmp = tmp.loc[tmp["地级"].notna()].copy()

    out = tmp.dissolve(by="地级", as_index=False)
    if out["地级"].duplicated().any():
        raise RuntimeError("City geometry still contains duplicate analysis-unit names after dissolve.")
    print(f"City geometry: {len(out)} polygons built with city_name_norm rules")
    return out


def _prepare_admin_geometry(level: str, join_key: str, scale_name: str):
    """Load/clean a map layer so the requested join key is one-to-one."""
    if level == "city":
        admin = _load_city_analysis_geometry()
        admin[join_key] = _clean_name(admin[join_key])
        admin = admin.loc[admin[join_key].notna()].copy()
    else:
        admin = load_admin_level(level).copy()
        if join_key not in admin.columns:
            raise KeyError(f"{scale_name} boundary layer is missing join key {join_key}")
        admin[join_key] = _clean_code(admin[join_key])
        valid = (
            admin[join_key].str.fullmatch(r"\d{6}", na=False).fillna(False).astype(bool)
            & admin[join_key].ne("000000").fillna(False).astype(bool)
        )
        if (~valid).any():
            print(
                f"{scale_name} geometry: dropping {int((~valid).sum())} "
                f"invalid/placeholder {join_key} polygon record(s)."
            )
        admin = admin.loc[valid].copy()

    if admin[join_key].duplicated().any():
        # This is defensive.  The current released direct boundaries are unique,
        # while city geometries are explicitly dissolved above.
        n_dup = int(admin.loc[admin[join_key].duplicated(keep=False), join_key].nunique())
        print(f"{scale_name} geometry: dissolving {n_dup} duplicated {join_key} key(s).")
        admin = admin.dissolve(by=join_key, as_index=False)

    if admin[join_key].duplicated().any():
        raise RuntimeError(f"{scale_name} geometry still contains duplicate {join_key} values.")
    return admin


def _export_change_layer(
    *,
    change_df: pd.DataFrame,
    level: str,
    join_key: str,
    scale_name: str,
    output_path: str,
) -> None:
    """Export one polygon per matched 2014/2024 analysis unit."""
    attrs = change_df[[join_key, "24-14"]].copy()
    if join_key.endswith("码"):
        attrs[join_key] = _clean_code(attrs[join_key])
    else:
        attrs[join_key] = _clean_name(attrs[join_key])

    if attrs[join_key].duplicated().any():
        raise RuntimeError(f"{scale_name} change table contains duplicate {join_key} values.")

    admin = _prepare_admin_geometry(level, join_key, scale_name)

    stat_keys = set(attrs[join_key].dropna().astype(str))
    map_keys = set(admin[join_key].dropna().astype(str))
    missing_in_map = sorted(stat_keys - map_keys)
    if missing_in_map:
        raise RuntimeError(
            f"{scale_name}: {len(missing_in_map)} matched statistical unit(s) are absent from "
            f"the released boundary layer; sample={missing_in_map[:20]}"
        )

    layer = admin.merge(attrs, on=join_key, how="inner", validate="one_to_one")
    if len(layer) != len(attrs):
        raise RuntimeError(
            f"{scale_name} map join mismatch: {len(attrs)} matched statistical units but "
            f"{len(layer)} output polygons."
        )

    if layer[join_key].duplicated().any():
        raise RuntimeError(f"{scale_name} exported layer contains duplicate {join_key} values.")

    layer.to_file(output_path, encoding="utf-8")
    print(f"{scale_name} map layer saved: {output_path} | polygons={len(layer)}")


def _plot_change_histogram(
    *,
    change_df: pd.DataFrame,
    output_path: str,
    bins_count: int,
    x_ticks: list[float],
    include_negative: bool,
    plot_range: tuple[float, float] | None = None,
) -> None:
    """Plot the compact positive/negative change histogram used beside Fig.2 maps."""
    plot_df = change_df.copy()
    if plot_range is not None:
        lo, hi = plot_range
        plot_df = plot_df[(plot_df["24-14"] > lo) & (plot_df["24-14"] < hi)].copy()

    all_vals = plot_df["24-14"].dropna().to_numpy(float)
    if len(all_vals) == 0:
        raise RuntimeError(f"No finite accessibility-change values available for {output_path}")

    bins = np.histogram_bin_edges(all_vals, bins=bins_count)

    sns.set_context("talk")
    plt.figure(figsize=(10, 6))

    if include_negative:
        neg = plot_df.loc[plot_df["24-14"] < 0, "24-14"].to_numpy(float)
        counts_neg, edges_neg = np.histogram(neg, bins=bins)
        plt.bar(
            edges_neg[:-1],
            -counts_neg,
            width=np.diff(edges_neg),
            align="edge",
            color=COLOR_N,
            alpha=0.6,
            edgecolor="white",
            label="< 0",
        )

    pos = plot_df.loc[plot_df["24-14"] >= 0, "24-14"].to_numpy(float)
    counts_pos, edges_pos = np.histogram(pos, bins=bins)
    plt.bar(
        edges_pos[:-1],
        counts_pos,
        width=np.diff(edges_pos),
        align="edge",
        color=COLOR_P,
        alpha=0.7 if not include_negative else 0.6,
        edgecolor="white",
        label="≥ 0",
    )

    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.tick_params(axis="x", which="both", bottom=True, labelbottom=False)

    ax = plt.gca()
    ax.set_xticks(x_ticks)
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["bottom"].set_color("gray")
    ax.spines["bottom"].set_linestyle("--")
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="x", colors="gray", width=1.5, length=6)
    ax.spines["top"].set_visible(False)

    plt.yticks([])
    plt.ylabel("")
    sns.despine(left=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_DPI, transparent=True, bbox_inches="tight")
    plt.close()

    print(f"{os.path.basename(output_path)} x ticks: {np.round(ax.get_xticks(), 2).tolist()}")


def main() -> None:
    output_dir = str(figure_dir("Figure 2"))
    os.makedirs(output_dir, exist_ok=True)

    output_dir2 = str(figure_dir("Map_layers", "acc_median_change_14_24"))
    os.makedirs(output_dir2, exist_ok=True)

    # ------------------------------------------------------------------
    # Province: stable six-digit province code
    # ------------------------------------------------------------------
    df_province = _prepare_code_change_frame(
        group="provincial",
        name_col="省级",
        code_col="省级码",
        scale_name="Province",
    )
    print(f"Provinces compared (present in both years): {len(df_province)}")
    print(f"Provinces with accessibility gains: {(df_province['24-14'] > 0).sum()}")

    _export_change_layer(
        change_df=df_province,
        level="province",
        join_key="省级码",
        scale_name="Province",
        output_path=os.path.join(output_dir2, "acc_median_change_province.shp"),
    )
    _plot_change_histogram(
        change_df=df_province,
        output_path=os.path.join(output_dir, "acc_change_province.png"),
        bins_count=6,
        x_ticks=[0, 5, 10, 15, 20],
        include_negative=False,
    )

    # ------------------------------------------------------------------
    # City: exact city_name_norm analysis key (written as 地级 in stats)
    # ------------------------------------------------------------------
    df_city = _prepare_city_change_frame()
    print(f"Cities compared (present in both years): {len(df_city)}")
    print(f"Cities with accessibility gains: {(df_city['24-14'] > 0).sum()}")

    _export_change_layer(
        change_df=df_city,
        level="city",
        join_key="地级",
        scale_name="City",
        output_path=os.path.join(output_dir2, "acc_median_change_city.shp"),
    )
    _plot_change_histogram(
        change_df=df_city,
        output_path=os.path.join(output_dir, "acc_change_city.png"),
        bins_count=13,
        x_ticks=[-10, 0, 10, 20, 30, 40],
        include_negative=True,
    )

    # ------------------------------------------------------------------
    # County: stable six-digit county code; never merge on county name.
    # ------------------------------------------------------------------
    df_county = _prepare_code_change_frame(
        group="county",
        name_col="县级",
        code_col="县级码",
        scale_name="County",
    )
    print(f"Counties compared (present in both years): {len(df_county)}")
    print(f"Counties with accessibility gains: {(df_county['24-14'] > 0).sum()}")

    _export_change_layer(
        change_df=df_county,
        level="county",
        join_key="县级码",
        scale_name="County",
        output_path=os.path.join(output_dir2, "acc_median_change_county.shp"),
    )
    _plot_change_histogram(
        change_df=df_county,
        output_path=os.path.join(output_dir, "acc_change_county.png"),
        bins_count=17,
        x_ticks=[-20, -10, 0, 10, 20, 30, 40, 50],
        include_negative=True,
        # Preserve the original display-range trimming used only for the inset
        # histogram.  Counts and map layers above use the full matched universe.
        plot_range=(-17, 35),
    )


if __name__ == "__main__":
    main()
