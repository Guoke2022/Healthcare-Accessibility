# -*- coding = utf-8 -*-


import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import warnings
from config import PLOT_DPI
from utils.figure_data import read_stats, figure_dir, load_admin_level


warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None
pd.set_option('display.float_format', '{:.2f}'.format)


LABEL_FONTSIZE = 25
TICKS_FONTSIZE = 45
LEGEND_FONTSIZE = 50

COLOR_N = '#ffeaad'
COLOR_P = '#b1c1da'

def main() -> None:

    output_dir = str(figure_dir('Figure 2'))
    os.makedirs(output_dir, exist_ok=True)

    output_dir2 = str(figure_dir('Map_layers', 'accessibility_change_2014_2024'))
    os.makedirs(output_dir2, exist_ok=True)


    df_province = read_stats('provincial', 'accessibility')[['Year', '省级', 'pop_median']]

    df_province_14 = df_province[df_province['Year'] == 2014]
    df_province_24 = df_province[df_province['Year'] == 2024]
    df_province_24_14 = pd.merge(df_province_14, df_province_24, how='inner', on='省级')
    df_province_24_14 = df_province_24_14.rename(
        columns={'pop_median_x':'2014_median', 'pop_median_y':'2024_median'})[['省级', '2014_median', '2024_median']]

    df_province_24_14['24-14'] = df_province_24_14['2024_median'] - df_province_24_14['2014_median']
    print(f"Provinces with accessibility gains: {(df_province_24_14['24-14'] > 0).sum()}")


    sheng_shp = load_admin_level('province')
    sheng_shp = sheng_shp.merge(df_province_24_14[['省级', '24-14']], left_on='省', right_on='省级', how='left')


    output_path2 = os.path.join(output_dir2, 'acc_median_change_sheng.shp')
    sheng_shp.to_file(output_path2)


    sns.set_context("talk")
    plt.figure(figsize=(10, 6))

    all_vals = df_province_24_14['24-14']
    bins = np.histogram_bin_edges(all_vals, bins=6)


    counts_pos, edges_pos = np.histogram(df_province_24_14[df_province_24_14['24-14'] >= 0]['24-14'], bins=bins)
    plt.bar(
        edges_pos[:-1], counts_pos,
        width=np.diff(edges_pos),
        align="edge",
        color=COLOR_P,
        alpha=0.7,
        edgecolor="white",
        label="≥ 0"
    )

    #

    # plt.axvline(0, color="gray", linestyle="--", linewidth=1)


    plt.axhline(0, color="gray", linestyle="--", linewidth=1)


    plt.tick_params(axis='x', which='both', bottom=True, labelbottom=False)
    # plt.tick_params(axis='x', which='both', bottom=True, labelbottom=True, labelsize=TICKS_FONTSIZE)


    ax = plt.gca()
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['bottom'].set_color("gray")
    ax.spines['bottom'].set_linestyle("--")
    ax.spines['bottom'].set_linewidth(1)


    ax.tick_params(
        axis="x",
        colors="gray",
        width=1.5,
        length=6
    )

    ax.spines['top'].set_visible(False)
    print("Province x ticks:", np.round(ax.get_xticks(), 2).tolist())

    plt.yticks([])
    plt.ylabel("")
    sns.despine(left=True)


    # plt.legend(frameon=False, fontsize=LEGEND_FONTSIZE)
    plt.tight_layout()


    output_path = os.path.join(output_dir, 'acc_change_province.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_DPI, transparent=True, bbox_inches='tight')
    plt.close()


    df_city = read_stats('city', 'accessibility')[['Year', '地级', 'pop_median']]

    df_city_14 = df_city[df_city['Year'] == 2014]
    df_city_24 = df_city[df_city['Year'] == 2024]
    df_city_24_14 = pd.merge(df_city_14, df_city_24, how='inner', on='地级')
    df_city_24_14 = df_city_24_14.rename(
        columns={'pop_median_x':'2014_median', 'pop_median_y':'2024_median'})[['地级', '2014_median', '2024_median']]

    df_city_24_14['24-14'] = df_city_24_14['2024_median'] - df_city_24_14['2014_median']
    print(f"Cities with accessibility gains: {(df_city_24_14['24-14'] > 0).sum()}")


    shi_shp = load_admin_level('city')


    shi_shp = shi_shp.merge(df_city_24_14[['地级', '24-14']], on='地级', how='left')


    output_path2 = os.path.join(output_dir2, 'acc_median_change_city.shp')
    shi_shp.to_file(output_path2)


    sns.set_context("talk")
    plt.figure(figsize=(10, 6))

    all_vals = df_city_24_14['24-14']
    bins = np.histogram_bin_edges(all_vals, bins=13)


    counts_neg, edges_neg = np.histogram(df_city_24_14[df_city_24_14['24-14'] < 0]['24-14'], bins=bins)
    plt.bar(
        edges_neg[:-1], -counts_neg,
        width=np.diff(edges_neg),
        align="edge",
        color=COLOR_N,
        alpha=0.6,
        edgecolor="white",
        label="< 0"
    )


    counts_pos, edges_pos = np.histogram(df_city_24_14[df_city_24_14['24-14'] >= 0]['24-14'], bins=bins)
    plt.bar(
        edges_pos[:-1], counts_pos,
        width=np.diff(edges_pos),
        align="edge",
        color=COLOR_P,
        alpha=0.6,
        edgecolor="white",
        label="≥ 0"
    )

    #

    # plt.axvline(0, color="gray", linestyle="--", linewidth=1)


    plt.axhline(0, color="gray", linestyle="--", linewidth=1)


    plt.tick_params(axis='x', which='both', bottom=True, labelbottom=False)
    # plt.tick_params(axis='x', which='both', bottom=True, labelbottom=True, labelsize=TICKS_FONTSIZE)


    ax = plt.gca()
    ax.set_xticks([-10, 0, 10, 20, 30, 40])
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['bottom'].set_color("gray")
    ax.spines['bottom'].set_linestyle("--")
    ax.spines['bottom'].set_linewidth(1)


    ax.tick_params(
        axis="x",
        colors="gray",
        width=1.5,
        length=6
    )

    ax.spines['top'].set_visible(False)
    print("City x ticks:", np.round(ax.get_xticks(), 2).tolist())

    plt.yticks([])
    plt.ylabel("")
    sns.despine(left=True)


    # plt.legend(frameon=False, fontsize=LEGEND_FONTSIZE)
    plt.tight_layout()


    output_path = os.path.join(output_dir, 'acc_change_city.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_DPI, transparent=True, bbox_inches='tight')
    plt.close()


    df_xian = read_stats('county', 'accessibility')[['Year', '县级', 'pop_median']]

    df_xian_14 = df_xian[df_xian['Year'] == 2014]
    df_xian_24 = df_xian[df_xian['Year'] == 2024]
    df_xian_24_14 = pd.merge(df_xian_14, df_xian_24, how='inner', on='县级')
    df_xian_24_14 = df_xian_24_14.rename(
        columns={'pop_median_x':'2014_median', 'pop_median_y':'2024_median'})[['县级', '2014_median', '2024_median']]

    df_xian_24_14['24-14'] = df_xian_24_14['2024_median'] - df_xian_24_14['2014_median']
    print(f"Counties with accessibility gains: {(df_xian_24_14['24-14'] > 0).sum()}")


    xian_shp = load_admin_level('county')

    xian_shp = xian_shp.merge(df_xian_24_14[['县级', '24-14']], on='县级', how='left')


    output_path2 = os.path.join(output_dir2, 'acc_median_change_county.shp')
    xian_shp.to_file(output_path2)


    sns.set_context("talk")
    plt.figure(figsize=(10, 6))

    df_xian_24_14 = df_xian_24_14[(df_xian_24_14['24-14'] < 35) & (df_xian_24_14['24-14'] > -17)]

    all_vals = df_xian_24_14['24-14']
    bins = np.histogram_bin_edges(all_vals, bins=17)


    counts_neg, edges_neg = np.histogram(df_xian_24_14[df_xian_24_14['24-14'] < 0]['24-14'], bins=bins)
    plt.bar(
        edges_neg[:-1], -counts_neg,
        width=np.diff(edges_neg),
        align="edge",
        color=COLOR_N,
        alpha=0.6,
        edgecolor="white",
        label="< 0"
    )


    counts_pos, edges_pos = np.histogram(df_xian_24_14[df_xian_24_14['24-14'] >= 0]['24-14'], bins=bins)
    plt.bar(
        edges_pos[:-1], counts_pos,
        width=np.diff(edges_pos),
        align="edge",
        color=COLOR_P,
        alpha=0.6,
        edgecolor="white",
        label="≥ 0"
    )

    #

    # plt.axvline(0, color="gray", linestyle="--", linewidth=1)


    plt.axhline(0, color="gray", linestyle="--", linewidth=1)


    plt.tick_params(axis='x', which='both', bottom=True, labelbottom=False)
    # plt.tick_params(axis='x', which='both', bottom=True, labelbottom=True, labelsize=TICKS_FONTSIZE)


    ax = plt.gca()
    ax.set_xticks([-20, -10, 0, 10, 20, 30, 40, 50])
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['bottom'].set_color("gray")
    ax.spines['bottom'].set_linestyle("--")
    ax.spines['bottom'].set_linewidth(1)


    ax.tick_params(
        axis="x",
        colors="gray",
        width=1.5,
        length=6
    )

    ax.spines['top'].set_visible(False)
    print("County x ticks:", np.round(ax.get_xticks(), 2).tolist())

    plt.yticks([])
    plt.ylabel("")
    sns.despine(left=True)


    # plt.legend(frameon=False, fontsize=LEGEND_FONTSIZE)
    plt.tight_layout()


    output_path = os.path.join(output_dir, 'acc_change_county.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_DPI, transparent=True, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
