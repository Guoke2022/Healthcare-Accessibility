# -*- coding: utf-8 -*-

import os
import warnings
from config import PLOT_DPI
from utils.figure_data import read_group_stats, figure_dir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.options.mode.chained_assignment = None
pd.set_option("display.float_format", "{:.2f}".format)

# =========================
# Style params
# =========================
LABEL_FONTSIZE = 20
TICKS_FONTSIZE = 20
LEGEND_FONTSIZE = 20

FIGSIZE_COMBINED = (8.5, 18)

BAR_ALPHA = 0.18
BAR_EDGE_ALPHA = 0.30

LINE_ALPHA = 1.0
LINE_WIDTH = 3.2
MARKER_SIZE = 8
MARKER_EDGE_W = 1.0


BAR_H_FIXED = 0.16

# =========================
# Unified color palette (8 colors)
# =========================
PALETTE = {
    # 2 + 2
    "Inland": "#c26e8d",
    "Coastal": "#a498ba",
    "Rural": "#445c47",
    "Urban": "#c3cf9e",

    # 4 city levels
    "Medium/Small City": "#3c9bc8",
    "Large City": "#66bdbb",
    "Super City": "#faa36e",
    "Mega City": "#ffe199",
}


def add_y_break_marker(ax, y_mid_data, x_pos=0.04, dx=0.018, dy=0.10, lw=1.2):

    trans = ax.get_yaxis_transform()  # x: axes fraction, y: data


    ax.plot([x_pos - dx, x_pos + dx], [y_mid_data - dy, y_mid_data - dy/3],
            transform=trans, color="black", lw=lw, clip_on=False)
    ax.plot([x_pos - dx, x_pos + dx], [y_mid_data + dy/3, y_mid_data + dy],
            transform=trans, color="black", lw=lw, clip_on=False)

def _prepare_pivot_matrix(df_pivot: pd.DataFrame, group_cols, desired_years):

    d = df_pivot.copy()
    if "Year" not in d.columns:
        raise ValueError("df_pivot must contain a 'Year' column.")
    d["Year"] = d["Year"].astype(int)

    missing_cols = [c for c in group_cols if c not in d.columns]
    if missing_cols:
        raise ValueError(f"Missing group columns in df_pivot: {missing_cols}")

    years = sorted([int(x) for x in desired_years], reverse=True)
    d = d.set_index("Year").reindex(years).reset_index()
    mat = d.set_index("Year")[group_cols]
    return mat


def plot_pivot_barline_portrait_on_ax(
    ax,
    mat: pd.DataFrame,
    group_cols,
    group_colors: dict,
    value_label="Accessibility",
    xlim=None,
    show_xlabel=True,
    show_x_axis=True,
    bar_h_fixed=BAR_H_FIXED
):

    years = mat.index.values
    n_year = len(years)
    n_grp = len(group_cols)
    y0 = np.arange(n_year)

    bar_h = float(bar_h_fixed)
    offsets = (np.arange(n_grp) - (n_grp - 1) / 2) * bar_h

    for j, g in enumerate(group_cols):
        y = y0 + offsets[j]
        x = pd.to_numeric(mat[g], errors="coerce").values

        ax.barh(
            y=y,
            width=x,
            height=bar_h * 0.92,
            color=group_colors[g],
            alpha=BAR_ALPHA,
            edgecolor=(0, 0, 0, BAR_EDGE_ALPHA),
            linewidth=0.6
        )

        ax.plot(
            x,
            y,
            color=group_colors[g],
            alpha=LINE_ALPHA,
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE,
            markeredgecolor="black",
            markeredgewidth=MARKER_EDGE_W,
            label=g
        )

    ax.set_yticks(y0)
    ax.set_yticklabels(years, fontsize=TICKS_FONTSIZE)
    # ax.set_ylabel("Year", fontsize=LABEL_FONTSIZE)

    if xlim is not None:
        ax.set_xlim(*xlim)

    ax.set_xlim(0, 46)

    # x axis control
    if show_x_axis:
        ax.tick_params(axis="x", labelsize=TICKS_FONTSIZE)
        if show_xlabel:
            ax.set_xlabel(value_label, fontsize=LABEL_FONTSIZE)
        else:
            ax.set_xlabel("")
    else:
        ax.set_xlabel("")
        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        ax.spines["bottom"].set_visible(False)

    ax.grid(True, axis="x", linestyle="-", alpha=0.60)
    # ax.grid(True, axis="y", linestyle="--", alpha=0.16)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def get_global_xlim(mats, pad_ratio=0.05):
    vals = []
    for mat in mats:
        v = mat.values.flatten()
        v = v[~np.isnan(v)]
        if v.size > 0:
            vals.append(v)
    if len(vals) == 0:
        return None

    allv = np.concatenate(vals)
    vmin = float(np.min(allv))
    vmax = float(np.max(allv))

    left = 0 if vmin >= 0 else vmin
    pad = pad_ratio * (vmax - left) if vmax > left else 1.0
    return (left, vmax + pad)


if __name__ == "__main__":

    output_directory = str(figure_dir('Figure 3'))
    os.makedirs(output_directory, exist_ok=True)

    years_full = list(range(2014, 2025))
    years_ur = [2014, 2015, 2016, 2019, 2020, 2021]

    # =========================
    # 1) Coastal / Inland
    # =========================
    df_ci = read_group_stats('Coastal_Inland', 'accessibility')
    df_ci["Year"] = df_ci["Year"].astype(int)
    df_ci_pivot = df_ci.pivot(index="Year", columns="Coastal_Inland", values="pop_median").reset_index()

    ci_order = ["Inland", "Coastal"]
    ci_colors = {k: PALETTE[k] for k in ci_order}
    mat_ci = _prepare_pivot_matrix(df_ci_pivot, ci_order, desired_years=years_full)

    # =========================
    # 2) Urban / Rural
    # =========================
    df_ur = read_group_stats('Urban_Rural', 'accessibility')
    df_ur = df_ur[df_ur["Urban_Rural"] != "Unknown"].copy()
    df_ur["Year"] = df_ur["Year"].astype(int)
    df_ur_pivot = df_ur.pivot(index="Year", columns="Urban_Rural", values="pop_median").reset_index()

    ur_order = ["Urban", "Rural"]
    ur_colors = {k: PALETTE[k] for k in ur_order}
    mat_ur = _prepare_pivot_matrix(df_ur_pivot, ur_order, desired_years=years_ur)

    # =========================
    # 3) City level
    # =========================
    df_cl = read_group_stats('city_level', 'accessibility')
    df_cl["Year"] = df_cl["Year"].astype(int)

    CITY_LEVEL_ORDER = ["Medium/Small City", "Large City", "Super City", "Mega City"]
    CITY_LEVEL_COLORS = {k: PALETTE[k] for k in CITY_LEVEL_ORDER}

    df_cl_pivot = (
        df_cl.pivot(index="Year", columns="city_level", values="pop_median")
             .reindex(columns=CITY_LEVEL_ORDER)
             .reset_index()
    )
    mat_cl = _prepare_pivot_matrix(df_cl_pivot, CITY_LEVEL_ORDER, desired_years=years_full)

    # =========================

    # =========================
    global_xlim = get_global_xlim([mat_ci, mat_ur, mat_cl], pad_ratio=0.05)

    # =========================
    # Draw combined figure
    # =========================
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=FIGSIZE_COMBINED,
        gridspec_kw={
            "height_ratios": [8, 4, 9]
        }
    )


    plot_pivot_barline_portrait_on_ax(
        ax=axes[0],
        mat=mat_ci,
        group_cols=ci_order,
        group_colors=ci_colors,
        value_label="Accessibility",
        xlim=global_xlim,
        show_xlabel=False,
        show_x_axis=False,
        bar_h_fixed=BAR_H_FIXED
    )


    plot_pivot_barline_portrait_on_ax(
        ax=axes[1],
        mat=mat_ur,
        group_cols=ur_order,
        group_colors=ur_colors,
        value_label="Accessibility",
        xlim=global_xlim,
        show_xlabel=False,
        show_x_axis=False,
        bar_h_fixed=BAR_H_FIXED
    )


    years_ur_desc = mat_ur.index.tolist()
    idx_2019 = years_ur_desc.index(2019)
    idx_2016 = years_ur_desc.index(2016)


    y_mid = (idx_2019 + idx_2016) / 2

    add_y_break_marker(
        axes[1],
        y_mid_data=y_mid,
        x_pos=0,
        dx=0.012,
        dy=0.10,
        lw=1.2
    )


    plot_pivot_barline_portrait_on_ax(
        ax=axes[2],
        mat=mat_cl,
        group_cols=CITY_LEVEL_ORDER,
        group_colors=CITY_LEVEL_COLORS,
        value_label="Accessibility",
        xlim=global_xlim,
        show_xlabel=True,
        show_x_axis=True,
        bar_h_fixed=BAR_H_FIXED
    )


    plt.subplots_adjust(hspace=0.04)

    # =========================
    # Unified legend (8 lines) on subplot 1 (top-right)
    # =========================


    legend_items = [
        ("Coastal", "Eastern China"),
        ("Inland", "Non-Eastern China"),
        ("Urban", "Urban"),
        ("Rural", "Rural"),
        ("Mega City", "Mega Cities"),
        ("Super City", "Super Cities"),
        ("Large City", "Large Cities"),
        ("Medium/Small City", "Medium & Small Cities"),
    ]

    handles = [
        plt.Line2D(
            [0], [0],
            color=PALETTE[key],
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE,
            markeredgecolor="black",
            markeredgewidth=MARKER_EDGE_W
        )
        for key, _ in legend_items
    ]

    labels = [label for _, label in legend_items]

    axes[0].legend(
        handles,
        labels,
        loc="upper right",
        fontsize=TICKS_FONTSIZE,
        frameon=False,
        borderaxespad=0.4,
        handlelength=2.4,
        labelspacing=0.5,
        markerfirst=False
    )

    out_png = os.path.join(output_directory, "acc_three_panels_portrait_barline_combined.png")
    plt.savefig(out_png, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

    print("Saved combined long figure to:", out_png)
