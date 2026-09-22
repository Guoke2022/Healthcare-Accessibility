# -*- coding: utf-8 -*-
"""Module utilities for fig5 3 see cie effect plots."""


# ============================================================
# Figure 5.3 SEE/CIE effect plots (4 city-size groups; 3 rows × 4 cols)
#   Output: Figure/Figure 5/Fig_SEE_CIE_Total_citysize.png
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import warnings
from config import SEE_CIE_REGRESSION_ROOT, FIGURE5_ROOT, PLOT_DPI
from utils.extended_analysis import p_to_star

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.options.mode.chained_assignment = None

# ============================================================
# Global font-size settings
# ============================================================
FS_BASE   = 12
FS_TITLE  = 15
FS_LABEL  = 15
FS_TICK   = 12
FS_LEGEND = 14
FS_SMALL  = 12

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],

    "font.size": FS_BASE,
    "axes.titlesize": FS_TITLE,
    "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND,

    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
})


BG = "#f2f2f2"
GRID_KW = dict(linestyle="--", linewidth=0.5, alpha=0.6)
ZERO_KW = dict(color="black", linestyle="--", linewidth=1.0)

SEE_COLOR = "#DD8452"
CIE_COLOR = "#385a7a"

METRIC_ORDER  = ["gini", "theil", "atkinson_05"]
METRIC_LABELS = ["Gini", "Theil", "Atkinson (ε=0.5)"]
METRIC_COLORS = {
    "gini":  "#92b4b8",
    "theil": "#f3d7a3",
    "atkinson_05":   "#cfa0a2"
}

CITY_ORDER  = ["Medium/Small City", "Large City", "Super City", "Mega City"]
CITY_LABELS = ["Medium & Small Cities", "Large Cities", "Super Cities", "Mega Cities"]

def split_coef_star(df):
    num = df.replace(r"[^0-9\.\-]", "", regex=True).replace("", np.nan).astype(float)
    star = df.replace(r"[0-9\.\-]", "", regex=True)
    return num, star

def get_ylim(mat):
    vals = mat.values.flatten()
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return (-1, 1)
    ymax = np.max(np.abs(vals))
    pad = 0.2 * ymax if ymax > 0 else 0.5
    return (-ymax - pad, ymax + pad)

def style_ax(ax, grid_axis="y"):
    ax.set_facecolor(BG)
    ax.grid(True, axis=grid_axis, **GRID_KW)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.axhline(0, **ZERO_KW)

def remove_x_axis(ax):
    ax.set_xticks([])
    ax.tick_params(axis="x", bottom=False, labelbottom=False)

def main():
    # ------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------
    OUTPUT_PATH3 = SEE_CIE_REGRESSION_ROOT
    OUTPUT_PATH2 = FIGURE5_ROOT
    OUTPUT_PATH2.mkdir(parents=True, exist_ok=True)

    effects = pd.read_csv(OUTPUT_PATH3 / "city_size_effects.csv")

    acc = effects[
        effects["outcome"].eq("accessibility")
        & effects["expansion"].isin(["SEE", "CIE"])
        & effects["city_level"].isin(CITY_ORDER)
    ].copy()
    if acc.empty:
        raise ValueError("No accessibility city-size effects found in city_size_effects.csv")
    me = acc.rename(columns={"expansion": "path", "significance": "star"})[
        ["city_level", "path", "effect", "se", "ci_low", "ci_high", "p", "star"]
    ].copy()

    fair = effects[
        effects["outcome"].isin(METRIC_ORDER)
        & effects["city_level"].isin(CITY_ORDER)
    ].copy()

    def effect_matrices(expansion):
        d = fair[fair["expansion"].eq(expansion)].copy()
        if d.empty:
            raise ValueError(f"No city-specific inequality effects found for expansion={expansion}")
        num = d.pivot(index="city_level", columns="outcome", values="effect").reindex(index=CITY_ORDER, columns=METRIC_ORDER)
        pmat = d.pivot(index="city_level", columns="outcome", values="p").reindex(index=CITY_ORDER, columns=METRIC_ORDER)
        # Elementwise significance labels. Use Series.map via DataFrame.apply for
        # compatibility with pandas 2.x and pandas 3.x (DataFrame.applymap was removed).
        star = pmat.apply(lambda col: col.map(p_to_star))
        sig = pmat < 0.10
        return num, star, sig

    see_num, see_star, see_sig = effect_matrices("SEE")
    cie_num, cie_star, cie_sig = effect_matrices("CIE")
    tot_num, tot_star, tot_sig = effect_matrices("TotalExpansion")

    # Dynamic ranges avoid clipping when the metric set or effect magnitudes change.
    SEE_CIE_YLIM = get_ylim(pd.concat([see_num, cie_num], axis=0))
    TOT_YLIM = get_ylim(tot_num)

    # ------------------------------------------------------------
    # 3) Plot layout: NOW 3 rows × 4 cols
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(9.6, 12))

    gs = GridSpec(
        nrows=3, ncols=4, figure=fig,
        height_ratios=[0.8, 1.0, 1.0],
        hspace=0.10,
        wspace=0.08
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.10)

    axes_map = {}

    # ------------------------------------------------------------
    # 4) Row 1: Acc marginal effects (SEE/CIE points + 95%CI)
    # ------------------------------------------------------------
    x_pos = np.array([0, 1])
    paths = ["SEE", "CIE"]
    cols  = [SEE_COLOR, CIE_COLOR]
    markers = {"SEE": "o", "CIE": "^"}

    for j, (city, ct) in enumerate(zip(CITY_ORDER, CITY_LABELS)):
        ax = fig.add_subplot(gs[0, j])
        axes_map[(0, j)] = ax
        style_ax(ax, grid_axis="y")
        ax.set_title(ct, fontsize=FS_TITLE)

        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(-5.2, 15.2)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        tmp = me[me["city_level"] == city].set_index("path").loc[paths]

        for i, path_i in enumerate(paths):
            y  = float(tmp.loc[path_i, "effect"])
            lo = float(tmp.loc[path_i, "ci_low"])
            hi = float(tmp.loc[path_i, "ci_high"])

            yerr = np.array([[y - lo], [hi - y]])
            is_sig = (lo > 0) or (hi < 0)

            ax.errorbar(
                x_pos[i], y, yerr=yerr,
                fmt=markers[path_i],
                color=cols[i], ecolor=cols[i],
                capsize=6, elinewidth=1.8,
                markersize=8, markeredgewidth=1.6,
                linewidth=1.8,
                markerfacecolor=(cols[i] if is_sig else "none"),
                markeredgecolor=cols[i],
                zorder=3,
                label=path_i
            )

            # --- numeric label for point estimate ---
            if is_sig:
                dx = 0.04 * (ax.get_xlim()[1] - ax.get_xlim()[0])
                ax.text(
                    x_pos[i] + dx,
                    y,
                    f"{y:.2f}{tmp.loc[path_i, 'star']}",
                    ha="left",
                    va="center",
                    fontsize=FS_SMALL
                )

        if j == 0:
            ax.legend(loc="upper left", frameon=False, handlelength=2.2)

        ax.set_xticks([])
        ax.tick_params(axis="x", bottom=False, labelbottom=False)

        if j == 0:
            ax.set_ylabel("Marginal effect on ΔAccessibility", labelpad=6)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)

    # ------------------------------------------------------------
    # 5) Row 2: TotalExpansion
    # ------------------------------------------------------------
    def plot_fairness_row_single(row_idx, num_df, star_df, sig_df, ylab, ylim):
        for j, (city, ct) in enumerate(zip(CITY_ORDER, CITY_LABELS)):
            ax = fig.add_subplot(gs[row_idx, j])
            axes_map[(row_idx, j)] = ax
            style_ax(ax, grid_axis="y")
            ax.set_ylim(ylim)

            x = np.arange(len(METRIC_ORDER))
            vals  = num_df.loc[city, METRIC_ORDER].values
            stars = star_df.loc[city, METRIC_ORDER].values
            sigs  = sig_df.loc[city, METRIC_ORDER].values

            for i, m in enumerate(METRIC_ORDER):
                v = vals[i]
                if np.isnan(v):
                    continue

                face = METRIC_COLORS[m]
                alpha = 1.0 if sigs[i] else 0.25
                lw = 1.1 if sigs[i] else 0.85

                ax.bar(
                    x[i], v,
                    width=0.22,
                    color=face,
                    alpha=alpha,
                    edgecolor="black",
                    linewidth=lw
                )

                dy = 0.03 * (ylim[1] - ylim[0])
                if sigs[i]:
                    ax.text(
                        x[i],
                        v + (dy if v >= 0 else -dy),
                        f"{v:.2f}{stars[i]}",
                        ha="center",
                        va="bottom" if v >= 0 else "top",
                        fontsize=FS_SMALL
                    )

            remove_x_axis(ax)

            if j == 0:
                ax.set_ylabel(ylab, labelpad=0)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
                ax.tick_params(axis="y", length=0)

    plot_fairness_row_single(
        row_idx=1,
        num_df=tot_num, star_df=tot_star, sig_df=tot_sig,
        ylab="Total effect on ΔInequality",
        ylim=TOT_YLIM
    )


    # ------------------------------------------------------------
    # 6) Row 3: COMBINED Fairness bars (SEE + CIE in same panel)
    # ------------------------------------------------------------
    def plot_fairness_row_combined(row_idx,
                                  see_num_df, see_star_df, see_sig_df,
                                  cie_num_df, cie_star_df, cie_sig_df,
                                  ylab, ylim):

        x = np.arange(len(METRIC_ORDER))
        bar_w = 0.16
        offset = 0.14

        EDGE_BLACK = "black"
        SEE_MARK = "s"
        CIE_MARK = "^"
        MS = 6.0         # marker size
        MEW = 1        # marker edge width

        for j, (city, ct) in enumerate(zip(CITY_ORDER, CITY_LABELS)):
            ax = fig.add_subplot(gs[row_idx, j])
            axes_map[(row_idx, j)] = ax
            style_ax(ax, grid_axis="y")
            ax.set_ylim(ylim)

            see_vals  = see_num_df.loc[city, METRIC_ORDER].values
            see_stars = see_star_df.loc[city, METRIC_ORDER].values
            see_sigs  = see_sig_df.loc[city, METRIC_ORDER].values

            cie_vals  = cie_num_df.loc[city, METRIC_ORDER].values
            cie_stars = cie_star_df.loc[city, METRIC_ORDER].values
            cie_sigs  = cie_sig_df.loc[city, METRIC_ORDER].values

            for i, m in enumerate(METRIC_ORDER):
                face = METRIC_COLORS[m]

                # --- SEE bar ---
                v = see_vals[i]
                if not np.isnan(v):
                    a = 1.0 if see_sigs[i] else 0.25
                    lw = 1.0
                    xpos = x[i] - offset

                    ax.bar(
                        xpos, v,
                        width=bar_w,
                        color=face,
                        alpha=a,
                        edgecolor=EDGE_BLACK,
                        linewidth=lw,
                        zorder=2
                    )


                    ax.plot(
                        xpos, v,
                        marker=SEE_MARK,
                        markersize=MS,
                        markerfacecolor=EDGE_BLACK,
                        markeredgecolor=EDGE_BLACK,
                        markeredgewidth=MEW,
                        linestyle="None",
                        alpha=a,
                        zorder=3
                    )

                    dy = 0.03 * (ylim[1] - ylim[0])
                    if see_sigs[i]:
                        ax.text(
                            xpos,
                            v + (dy if v >= 0 else -dy),
                            f"{v:.2f}{see_stars[i]}",
                            ha="center",
                            va="bottom" if v >= 0 else "top",
                            fontsize=FS_SMALL,
                            zorder=4
                        )

                # --- CIE bar ---
                v = cie_vals[i]
                if not np.isnan(v):
                    a = 1.0 if cie_sigs[i] else 0.25
                    lw = 1.0
                    xpos = x[i] + offset

                    ax.bar(
                        xpos, v,
                        width=bar_w,
                        color=face,
                        alpha=a,
                        edgecolor=EDGE_BLACK,
                        linewidth=lw,
                        zorder=2
                    )


                    ax.plot(
                        xpos, v,
                        marker=CIE_MARK,
                        markersize=MS,
                        markerfacecolor=EDGE_BLACK,
                        markeredgecolor=EDGE_BLACK,
                        markeredgewidth=MEW,
                        linestyle="None",
                        alpha=a,
                        zorder=3
                    )

                    dy = 0.03 * (ylim[1] - ylim[0])
                    if cie_sigs[i]:
                        ax.text(
                            xpos,
                            v + (dy if v >= 0 else -dy),
                            f"{v:.2f}{cie_stars[i]}",
                            ha="center",
                            va="bottom" if v >= 0 else "top",
                            fontsize=FS_SMALL,
                            zorder=4
                        )

            remove_x_axis(ax)

            if j == 0:
                ax.set_ylabel(ylab, labelpad=0)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
                ax.tick_params(axis="y", length=0)

    plot_fairness_row_combined(
        row_idx=2,
        see_num_df=see_num, see_star_df=see_star, see_sig_df=see_sig,
        cie_num_df=cie_num, cie_star_df=cie_star, cie_sig_df=cie_sig,
        ylab="Pathway effects on ΔInequality",
        ylim=SEE_CIE_YLIM
    )

    # ------------------------------------------------------------
    # Legends (transparent background)
    #   Row 2 (TotalExpansion): metrics only
    #   Row 3 (Combined): hatches only
    # ------------------------------------------------------------
    EDGE_BLACK = "black"
    SEE_HATCH = "///"
    CIE_HATCH = "xx"

    # Row 2 legend: formal inequality metrics
    legend_metrics = [
        Patch(facecolor=METRIC_COLORS["gini"],  edgecolor=EDGE_BLACK , label="Gini"),
        Patch(facecolor=METRIC_COLORS["theil"], edgecolor=EDGE_BLACK , label="Theil"),
        Patch(facecolor=METRIC_COLORS["atkinson_05"],   edgecolor=EDGE_BLACK , label="Atkinson"),
    ]
    ax_leg2 = axes_map[(1, 0)]
    leg_metrics = ax_leg2.legend(
        handles=legend_metrics,
        loc="upper left",
        frameon=False,
        handlelength=2.2,
        columnspacing=1.6
    )

    leg_metrics.get_frame().set_facecolor("none")
    leg_metrics.get_frame().set_alpha(0.0)

    # Row 3 legend: markers (SEE/CIE)
    ax_leg3 = axes_map[(2, 0)]
    legend_paths = [
        Line2D([0], [0], marker="s", linestyle="None",
               markerfacecolor="black", markeredgecolor="black",
               markeredgewidth=1.0, markersize=6.5, label="SEE"),
        Line2D([0], [0], marker="^", linestyle="None",
               markerfacecolor="black", markeredgecolor="black",
               markeredgewidth=1.0, markersize=6.5, label="CIE"),
    ]
    leg_paths = ax_leg3.legend(
        handles=legend_paths,
        loc="upper left",
        frameon=False,
        handlelength=1.2,
        columnspacing=1.2
    )
    leg_paths.get_frame().set_facecolor("none")
    leg_paths.get_frame().set_alpha(0.0)

    # ------------------------------------------------------------
    # 8) Save
    # ------------------------------------------------------------
    outpath = OUTPUT_PATH2 / "Fig_SEE_CIE_Total_citysize.png"
    plt.savefig(outpath, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)



if __name__ == "__main__":
    main()
