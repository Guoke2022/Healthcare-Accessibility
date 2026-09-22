# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import PLOT_DPI
from utils.figure_data import shapley_plot_table, figure_dir

# =========================
# Paths
# =========================
CSV_PATH = None
OUT_DIR = None

# =========================
# What to plot
# =========================
PLOTS = [
    ("accessibility",  "pop_median", "acc_pop_median"),
    ("inequality_acc", "pop_gini",   "ineq_pop_gini"),
    ("inequality_acc", "pop_theil",  "ineq_pop_theil"),
    ("inequality_acc", "pop_atkinson_05",    "ineq_pop_atkinson_05"),
]

TITLE_MAP = {
    "pop_median": "Median accessibility",
    "pop_gini": "Gini",
    "pop_theil": "Theil",
    "pop_atkinson_05": "Atkinson (ε=0.5)",
}


IMPROVE_SIGN = {
    "accessibility": +1,
    "inequality_acc": -1,
}

FACTOR_ORDER = ["2014", "Bed", "Population", "Road", "2024"]

FACTOR_LABEL = {
    "Bed": "Hospital\nexpansion",
    "Population": "Population\nchange",
    "Road": "Mapped road\nconditions",
}

# =========================
# Global typography

# =========================
FONT_FAMILY = "sans-serif"

TUHAO_FS = 16
TITLE_FS = 15
YLABEL_FS = 14
XTICK_FS = 12
STEP_TEXT_FS = 14
YEAR_TEXT_FS = 12

# =========================
# Figure / style
# =========================
DPI = PLOT_DPI
FIGSIZE_SINGLE = (6.2, 4.4)
FIGSIZE_GRID = (12.2, 8.4)

AXIS_COLOR = "#222222"
TEXT_COLOR = "#222222"
LIGHT_TEXT = "#7a7a7a"
GUIDE_COLOR = "#cfcfcf"

COLOR_POS = "#E6862E"   # warm orange
COLOR_NEG = "#4E79A7"   # restrained blue

START_EDGE = "#8a8a8a"
END_FACE = "#8a8a8a"

ARROW_LW = 1.5
AXIS_LW = 1.0
GUIDE_LW = 0.8
TICK_LW = 0.8

START_SIZE = 30
END_SIZE = 30


YMAP = {
    "Bed": 2.0,
    "Population": 1.0,
    "Road": 0.0,
}


LABEL_DY = {
    "Bed": 0.08,
    "Population": 0.08,
    "Road": 0.08,
}


YEAR2014_DY = 0.12
YEAR2024_DY = -0.12


YMIN = -0.58
YMAX = 2.82
TOP_AXIS_Y = 2.58

# =========================
# Matplotlib rc
# =========================
plt.rcParams["font.family"] = FONT_FAMILY
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def build_positions(df_sub):
    """Helper for build_positions."""


    d = df_sub.set_index("factor").reindex(FACTOR_ORDER)

    val = d["Contribution_abs"].to_dict()
    pct = d["Contribution_pct"].to_dict()

    start = float(val["2014"])
    bed = float(val["Bed"])
    pop = float(val["Population"])
    road = float(val["Road"])
    end = float(val["2024"])

    x0 = start
    x1 = x0 + bed
    x2 = x1 + pop
    x3 = x2 + road

    pos = {
        "2014": x0,
        "Bed": (x0, x1),
        "Population": (x1, x2),
        "Road": (x2, x3),
        "2024": end,
    }
    delta = {
        "Bed": bed,
        "Population": pop,
        "Road": road,
    }
    return pos, pct, delta


def format_tick_value(x):
    return f"{x:.2f}".rstrip("0").rstrip(".")


def add_top_axis_ticks(ax, tick_values):
    tick_len = 0.07
    for xv in tick_values:
        ax.plot(
            [xv, xv],
            [TOP_AXIS_Y, TOP_AXIS_Y + tick_len],
            color=AXIS_COLOR,
            lw=TICK_LW,
            zorder=5
        )
        ax.text(
            xv, TOP_AXIS_Y + tick_len + 0.06,
            format_tick_value(xv),
            ha="center", va="bottom",
            fontsize=XTICK_FS,
            color=TEXT_COLOR
        )


def draw_arrow(ax, x0, x1, y, color):
    ax.annotate(
        "",
        xy=(x1, y), xytext=(x0, y),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=ARROW_LW,
            color=color,
            mutation_scale=8,
            shrinkA=0,
            shrinkB=0,
            joinstyle="miter",
            capstyle="butt",
        ),
        zorder=4
    )


def get_color_map(task, delta):
    sign = IMPROVE_SIGN[task]
    return {
        k: (COLOR_POS if sign * delta[k] >= 0 else COLOR_NEG)
        for k in ["Bed", "Population", "Road"]
    }


def compute_xlim(pos):
    xs = [pos["2014"], pos["2024"]]
    for k in ["Bed", "Population", "Road"]:
        xs.extend(list(pos[k]))

    xmin = min(xs)
    xmax = max(xs)
    xr = xmax - xmin
    if xr == 0:
        xr = 1.0

    left_pad = 0.12 * xr
    right_pad = 0.06 * xr
    return xmin - left_pad, xmax + right_pad, xr


def draw_custom_axes(ax, xmin, xmax, xr):

    x_axis_left = xmin
    x_axis_right = xmax

    ax.plot(
        [x_axis_left, x_axis_left],
        [YMIN, TOP_AXIS_Y],
        color=AXIS_COLOR, lw=AXIS_LW, zorder=1
    )
    ax.plot(
        [x_axis_left, x_axis_right],
        [TOP_AXIS_Y, TOP_AXIS_Y],
        color=AXIS_COLOR, lw=AXIS_LW, zorder=1
    )

    y_tick_len = 0.018 * xr
    for y in [2, 1, 0]:
        ax.plot(
            [x_axis_left - y_tick_len, x_axis_left],
            [y, y],
            color=AXIS_COLOR,
            lw=TICK_LW,
            zorder=3,
            solid_capstyle="butt"
        )

    label_x = x_axis_left - 0.02 * xr
    ax.text(label_x, 2, FACTOR_LABEL["Bed"],
            ha="right", va="center", fontsize=YLABEL_FS, color=TEXT_COLOR)
    ax.text(label_x, 1, FACTOR_LABEL["Population"],
            ha="right", va="center", fontsize=YLABEL_FS, color=TEXT_COLOR)
    ax.text(label_x, 0, FACTOR_LABEL["Road"],
            ha="right", va="center", fontsize=YLABEL_FS, color=TEXT_COLOR)


def draw_guides(ax, pos):

    ax.plot(
        [pos["Bed"][1], pos["Bed"][1]],
        [YMAP["Population"], YMAP["Bed"]],
        ls=(0, (2.2, 2.2)), lw=GUIDE_LW, color=GUIDE_COLOR, zorder=1
    )
    ax.plot(
        [pos["Population"][1], pos["Population"][1]],
        [YMAP["Road"], YMAP["Population"]],
        ls=(0, (2.2, 2.2)), lw=GUIDE_LW, color=GUIDE_COLOR, zorder=1
    )


def draw_end_points(ax, pos):

    ax.scatter(
        [pos["2014"]], [YMAP["Bed"]],
        s=START_SIZE,
        marker="o",
        facecolors="white",
        edgecolors=START_EDGE,
        linewidths=1.0,
        zorder=6
    )

    ax.scatter(
        [pos["2024"]], [YMAP["Road"]],
        s=END_SIZE,
        marker="o",
        facecolors=END_FACE,
        edgecolors=END_FACE,
        linewidths=0.8,
        zorder=6
    )

    ax.text(
        pos["2014"], YMAP["Bed"] + YEAR2014_DY, "2014",
        ha="center", va="bottom",
        fontsize=YEAR_TEXT_FS, color=LIGHT_TEXT
    )
    ax.text(
        pos["2024"], YMAP["Road"] + YEAR2024_DY, "2024",
        ha="center", va="top",
        fontsize=YEAR_TEXT_FS, color=LIGHT_TEXT
    )


def draw_steps(ax, pos, pct, color_map):
    for k in ["Bed", "Population", "Road"]:
        x0, x1 = pos[k]
        draw_arrow(ax, x0, x1, YMAP[k], color_map[k])

        xm = (x0 + x1) / 2
        ax.text(
            xm, YMAP[k] + LABEL_DY[k],
            f"{pct[k]:+.1f}%",
            ha="center", va="bottom",
            fontsize=STEP_TEXT_FS,
            color=TEXT_COLOR
        )


def plot_one(ax, df_sub, task, indicator):
    pos, pct, delta = build_positions(df_sub)
    color_map = get_color_map(task, delta)

    xmin, xmax, xr = compute_xlim(pos)


    x_axis_left = xmin - 0.04 * xr
    x_axis_right = xmax

    ax.set_xlim(x_axis_left - 0.01 * xr, x_axis_right + 0.005 * xr)
    ax.set_ylim(YMIN, YMAX)

    ax.set_xticks([])
    ax.set_yticks([])

    draw_custom_axes(ax, x_axis_left, x_axis_right, xr)
    add_top_axis_ticks(ax, [pos["2014"], pos["2024"]])
    draw_guides(ax, pos)
    draw_end_points(ax, pos)
    draw_steps(ax, pos, pct, color_map)

    ax.set_title(
        TITLE_MAP[indicator],
        fontsize=TITLE_FS,
        color=TEXT_COLOR,
        pad=8,
        fontweight="normal"
    )

    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.set_facecolor("white")

def main() -> None:
    out_dir = figure_dir("Shapley_decomposition")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = shapley_plot_table()
    df["factor"] = df["factor"].astype(str)

    # =========================
    # 1) four single panels
    # =========================
    for task, indicator, tag in PLOTS:
        sub = df[(df["task"] == task) & (df["indicator"] == indicator)].copy()
        if sub.empty:
            raise ValueError(f"没找到数据：task={task}, indicator={indicator}")

        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE, facecolor="white")
        plot_one(ax, sub, task, indicator)

        fig.subplots_adjust(left=0.18, right=0.98, top=0.86, bottom=0.10)
        out_png = out_dir / f"{tag}_step_arrow_nature_style.png"
        fig.savefig(out_png, dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # =========================
    # 2) 2x2 composite
    # =========================
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_GRID, facecolor="white")
    axes = axes.ravel()

    panel_labels = ["a", "b", "c", "d"]

    for ax, (task, indicator, tag), lab in zip(axes, PLOTS, panel_labels):
        sub = df[(df["task"] == task) & (df["indicator"] == indicator)].copy()
        if sub.empty:
            raise ValueError(f"没找到数据：task={task}, indicator={indicator}")
        plot_one(ax, sub, task, indicator)

        ax.text(
            -0.1, 1.00, lab,
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=TUHAO_FS,
            fontweight="bold",
            color=TEXT_COLOR
        )

    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        top=0.92,
        bottom=0.08,
        wspace=0.3,
        hspace=0.3
    )

    out_grid = out_dir / "Shapley_decomposition_2x2.png"
    fig.savefig(out_grid, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)



if __name__ == "__main__":
    main()
