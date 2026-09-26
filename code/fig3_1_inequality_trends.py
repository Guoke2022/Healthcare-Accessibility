# -*- coding = utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

from config import PLOT_DPI
from utils.figure_data import read_group_stats, figure_dir
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None
pd.set_option('display.float_format', '{:.2f}'.format)

LABEL_FONTSIZE = 25
TICKS_FONTSIZE = 23
TEXT_FONTSIZE = 25

LINE_WIDTH = 2
MARKER_SIZE = 11


def add_x_break_marker(ax, x_mid, y_pos=0.02, dx=0.12, dy=0.035, lw=1.2):
    """Helper for add_x_break_marker."""


    trans = ax.get_xaxis_transform()  # x: data, y: axes fraction


    ax.plot([x_mid - dx, x_mid - dx/3], [y_pos - dy, y_pos + dy],
            transform=trans, color="black", lw=lw, clip_on=False)
    ax.plot([x_mid + dx/3, x_mid + dx], [y_pos - dy, y_pos + dy],
            transform=trans, color="black", lw=lw, clip_on=False)


def _auto_ylim(dfs) -> tuple[float, float]:
    cols = ["pop_gini", "pop_theil", "pop_atkinson_05"]
    vals = []
    for df in dfs:
        for c in cols:
            if c in df.columns:
                v = pd.to_numeric(df[c], errors="coerce").dropna().values
                vals.extend(v.tolist())
    if not vals:
        return (0.0, 1.0)
    lo, hi = float(min(vals)), float(max(vals))
    span = hi - lo
    pad = max(0.03, span * 0.10)
    return (max(0.0, lo - pad), hi + pad)


# =========================
# Core: plot on a given axis
# =========================
def plot_inequality_on_ax(
    ax: plt.Axes,
    df: pd.DataFrame,
    plot_label: str = "",
    ylim: tuple | None = None,
    yticks: tuple = None,
    years_to_show: list = None,
    show_yaxis: bool = True
):
    """Helper for plot_inequality_on_ax."""


    line_color = 'black'
    marker_colors = {
        'pop_gini': '#f4c761',
        'pop_theil': '#9fa2d1',
        'pop_atkinson_05': '#7dc78a'
    }


    needed_cols = ['Year', 'pop_gini', 'pop_theil', 'pop_atkinson_05']
    df = df[needed_cols].copy()
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Year']).sort_values('Year')
    df.set_index('Year', inplace=True)


    ax.text(
        0.97, 0.96, plot_label,
        transform=ax.transAxes,
        fontsize=TEXT_FONTSIZE,
        fontweight='normal',
        style="italic",
        ha='right',
        va='top'
    )


    if years_to_show is not None:
        years_to_show = [int(y) for y in years_to_show]
        pos_map = {y: i for i, y in enumerate(years_to_show)}


        df = df[df.index.isin(years_to_show)].copy()

        x = df.index.to_series().map(pos_map).values  # 0..n-1
        xticks = list(range(len(years_to_show)))
        xticklabels = [str(y) for y in years_to_show]
    else:
        x = df.index.values


        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(v)}"))

        xticks = None
        xticklabels = None


    for metric in ['pop_gini', 'pop_theil', 'pop_atkinson_05']:
        ax.plot(
            x,
            df[metric].values,
            marker='o',
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            color=line_color,
            markerfacecolor=marker_colors.get(metric, 'white'),
            markeredgecolor=line_color
        )


    if ylim is None:
        ylim = _auto_ylim([df.reset_index()])
    ax.set_ylim(ylim)
    ax.set_xlabel('')
    ax.tick_params(axis="x", labelsize=TICKS_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICKS_FONTSIZE)

    if yticks:
        ax.set_yticks(list(range(*yticks)))


    # ax.grid(True, axis='x', linestyle='--', alpha=0.6)
    ax.grid(True, axis='y', linestyle='-', alpha=0.6)
    ax.set_axisbelow(True)

    # spine
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)


    if not show_yaxis:
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        ax.set_ylabel("")


# =========================
# Wrapper: single figure save (National)
# =========================
def save_single_inequality_plot(
    df: pd.DataFrame,
    output_path: str,
    plot_label: str = '',
    ylim: tuple | None = None,
    yticks: tuple = None,
    years_to_show: list = None
):
    fig, ax = plt.subplots(figsize=(8, 7))
    if ylim is None:
        ylim = _auto_ylim([df])
    plot_inequality_on_ax(
        ax=ax,
        df=df,
        plot_label=plot_label,
        ylim=ylim,
        yticks=yticks,
        years_to_show=years_to_show,
        show_yaxis=True
    )
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)


# =========================
# Wrapper: horizontal panels
# =========================
def save_horizontal_panels(
    dfs,
    labels,
    output_path,
    fig_size=(16, 6.5),
    ylim=None,
    yticks=None,
    years_to_show=None,
    keep_yaxis_first_only=True,
    x_break_between=None,
    hide_all_yaxis=False
):
    n = len(dfs)
    if ylim is None:
        ylim = _auto_ylim(dfs)
    fig, axes = plt.subplots(1, n, figsize=fig_size, sharey=True)

    if n == 1:
        axes = [axes]

    for i, (ax, df, lab) in enumerate(zip(axes, dfs, labels)):
        plot_inequality_on_ax(
            ax=ax,
            df=df,
            plot_label=lab,
            ylim=ylim,
            yticks=yticks,
            years_to_show=years_to_show,
            show_yaxis=(not keep_yaxis_first_only or i == 0)
        )


    if hide_all_yaxis:
        for ax in axes:
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)
            ax.set_ylabel("")

    # x-axis break marker
    if (x_break_between is not None) and (years_to_show is not None):
        y_left, y_right = x_break_between
        years = [int(y) for y in years_to_show]
        if (y_left in years) and (y_right in years):
            i_left = years.index(int(y_left))
            i_right = years.index(int(y_right))
            x_mid = (i_left + i_right) / 2.0
            for ax in axes:
                add_x_break_marker(ax, x_mid=x_mid, y_pos=0, dx=0.05, dy=0.02, lw=1.2)

    plt.subplots_adjust(wspace=0.08)
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':


    output_directory = str(figure_dir('Figure 3'))
    os.makedirs(output_directory, exist_ok=True)

    # =========================

    # =========================
    df = read_group_stats('national', 'accessibility')

    output_file = os.path.join(output_directory, 'acc_national_inequality.png')
    save_single_inequality_plot(
        df,
        output_path=output_file,
        plot_label='National',
        ylim=None,
        yticks=None,
        years_to_show=None
    )

    # =========================

    # =========================
    df = read_group_stats('Coastal_Inland', 'accessibility')

    df_inland = df[df['Coastal_Inland'] == 'Inland'].copy()
    df_coastal = df[df['Coastal_Inland'] == 'Coastal'].copy()

    output_file = os.path.join(output_directory, 'acc_Inland_Coastal_inequality_panel.png')
    save_horizontal_panels(
        dfs=[df_coastal, df_inland],
        labels=['Eastern China', 'Non–Eastern China'],
        output_path=output_file,
        fig_size=(16, 6.5),
        ylim=None,
        yticks=None,
        years_to_show=None,
        keep_yaxis_first_only=True
    )

    # =========================

    # =========================
    df = read_group_stats('Urban_Rural', 'accessibility')

    years_to_show_ur = [2014, 2015, 2016, 2019, 2020, 2021]
    df = df[df['Year'].isin(years_to_show_ur)].copy()

    df_rural = df[df['Urban_Rural'] == 'Rural'].copy()
    df_urban = df[df['Urban_Rural'] == 'Urban'].copy()

    output_file = os.path.join(output_directory, 'acc_Rural_Urban_inequality_panel.png')
    save_horizontal_panels(
        dfs=[df_urban, df_rural],
        labels=['Urban', 'Rural'],
        output_path=output_file,
        fig_size=(16, 6.5),
        ylim=None,
        yticks=None,
        years_to_show=years_to_show_ur,
        keep_yaxis_first_only=True,
        x_break_between=(2016, 2019),
        hide_all_yaxis=True
    )

    # =========================

    # =========================
    df = read_group_stats('city_level', 'accessibility')

    CITY_LEVEL_ORDER = [
        'Mega City',
        'Super City',
        'Large City',
        'Medium/Small City'
    ]

    LABEL_MAP = {
        'Mega City': 'Mega Cities',
        'Super City': 'Super Cities',
        'Large City': 'Large Cities',
        'Medium/Small City': 'Medium & Small Cities'
    }

    dfs = []
    labels = []
    for lvl in CITY_LEVEL_ORDER:
        df_sub = df[df['city_level'] == lvl].copy()
        if df_sub.empty:
            raise ValueError(f"No data available for city_level = {lvl}")
        dfs.append(df_sub)
        labels.append(LABEL_MAP.get(lvl, lvl))

    output_file = os.path.join(output_directory, 'acc_city_level_inequality_panel.png')
    save_horizontal_panels(
        dfs=dfs,
        labels=labels,
        output_path=output_file,
        fig_size=(30, 6.5),
        ylim=None,
        yticks=None,
        years_to_show=None,
        keep_yaxis_first_only=True
    )
