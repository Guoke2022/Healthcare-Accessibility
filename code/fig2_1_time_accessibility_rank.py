# -*- coding = utf-8 -*-

import seaborn as sns
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import warnings

from config import PLOT_DPI
from utils.figure_data import read_stats, figure_dir

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None
pd.set_option('display.float_format', '{:.2f}'.format)


SHENG_Chinese = ["北京市", "山西省", "内蒙古自治区", "吉林省", "黑龙江省", "安徽省", "江西省", "河南省", "湖北省",
                 "湖南省", "重庆市", "四川省", "贵州省", "云南省", "西藏自治区", "陕西省", "甘肃省", "青海省",
                 "宁夏回族自治区", "新疆维吾尔自治区", "辽宁省", "河北省", "天津市", "山东省", "江苏省", "上海市",
                 "浙江省", "福建省", "广东省", "广西壮族自治区", "海南省"]

SHENG_English = ["Beijing", "Shanxi", "Nei Mongol", "Jilin", "Heilongjiang", "Anhui", "Jiangxi", "Henan", "Hubei",
                 "Hunan", "Chongqing", "Sichuan", "Guizhou", "Yunnan", "Tibet", "Shaanxi", "Gansu", "Qinghai",
                 "Ningxia", "Xinjiang", "Liaoning", "Hebei", "Tianjin", "Shandong", "Jiangsu", "Shanghai",
                 "Zhejiang", "Fujian", "Guangdong", "Guangxi", "Hainan"]


province_map = dict(zip(SHENG_Chinese, SHENG_English))

LABEL_FONTSIZE = 22
TICKS_FONTSIZE = 26
LEGEND_FONTSIZE = 33

def main() -> None:

    df_time = read_stats('provincial', 'travel_time')


    df_time_avg = (
        df_time[(df_time['Year'] >= 2014) & (df_time['Year'] <= 2024)]
        .groupby('省级', as_index=False)['pop_median'].mean()
    )
    df_time_avg['省级英文'] = df_time_avg['省级'].map(province_map)
    df_time_avg = df_time_avg[['省级英文', 'pop_median']].sort_values('pop_median', ascending=False)


    # Display-only cap used in the manuscript panel; provincial ranking is calculated from the uncapped values above.
    df_time_avg.loc[df_time_avg['省级英文'] == "Tibet", 'pop_median'] = 120


    df_acc = read_stats('provincial', 'accessibility')


    df_acc_avg = (
        df_acc[(df_acc['Year'] >= 2014) & (df_acc['Year'] <= 2024)]
        .groupby('省级', as_index=False)['pop_median'].mean()
    )
    df_acc_avg['省级英文'] = df_acc_avg['省级'].map(province_map)
    df_acc_avg = df_acc_avg[['省级英文', 'pop_median']].sort_values('pop_median')


    plt.figure(figsize=(10, 14))

    bars = plt.barh(
        df_time_avg['省级英文'],
        df_time_avg['pop_median'],
        color="#55A868",
        left=0
    )
    # plt.xlabel("Median Travel Time", fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICKS_FONTSIZE)

    ax = plt.gca()


    ax.set_facecolor("#e2f0d950")


    ax.set_yticks([]); ax.set_yticklabels([])
    # ax.set_xticks([]); ax.set_xticklabels([])


    max_val_t = df_time_avg['pop_median'].max()
    ax.set_xlim(max_val_t * 1.1, 0)


    ax.set_ylim(-0.5, len(df_time_avg['省级英文']) - 0.5)


    for bar, name in zip(bars, df_time_avg['省级英文']):
        width = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ax.text(width + max_val_t * 0.01, y, name,
                va='center', ha='right', fontsize=LABEL_FONTSIZE, clip_on=False)


    hatch_map = {
        "Shanxi": "//",
        "Xinjiang": "\\\\"
    }
    for bar, name in zip(bars, df_time_avg['省级英文']):
        if name in hatch_map:
            bar.set_hatch(hatch_map[name])
            bar.set_edgecolor("white")
            bar.set_linewidth(1)


    for spine in ['top', 'left', 'right', 'bottom']:
        ax.spines[spine].set_visible(False)

    ax.text(
        0.28, 0.96,
        "Median Travel Time",
        fontsize=LEGEND_FONTSIZE,
        fontweight="bold",
        style="italic",
        color="#55A868",
        ha="center", va="bottom",
        transform=ax.transAxes
    )

    plt.tight_layout()
    output_path1 = str(figure_dir('Figure 2') / 'time_rank.png')
    plt.savefig(output_path1, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()


    plt.figure(figsize=(10, 14))

    bars = plt.barh(df_acc_avg['省级英文'], df_acc_avg['pop_median'], color="#4C72B0")
    ax = plt.gca()


    ax.set_facecolor("#e8f4fc80")


    ax.set_yticks([]); ax.set_yticklabels([])
    # ax.set_xticks([]); ax.set_xticklabels([])

    ax.set_xlim(0, 34)

    ax.set_ylim(-0.5, len(df_time_avg['省级英文']) - 0.5)


    ax.tick_params(axis="y", length=0)


    ax.set_yticks(range(len(df_acc_avg)))
    ax.set_yticklabels(range(len(df_acc_avg), 0, -1), fontsize=LABEL_FONTSIZE, ha="center")


    ax.tick_params(axis="y", pad=20)


    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", labelsize=TICKS_FONTSIZE)


    for bar, name in zip(bars, df_acc_avg['省级英文']):
        width = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        plt.text(width + max(df_acc_avg['pop_median']) * 0.01,
                 y, name,
                 va='center', ha='left', fontsize=LABEL_FONTSIZE)


    hatch_map = {
        "Shanxi": "//",
        "Xinjiang": "\\\\"
    }
    for bar, name in zip(bars, df_acc_avg['省级英文']):
        if name in hatch_map:
            bar.set_hatch(hatch_map[name])
            bar.set_edgecolor("white")
            bar.set_linewidth(1)


    for spine in ['top', 'left', 'right', 'bottom']:
        ax.spines[spine].set_visible(False)

    ax.text(
        0.70, 0.002,
        "Median Accessibility",
        fontsize=LEGEND_FONTSIZE,
        fontweight="bold",
        style="italic",
        color="#4C72B0",
        ha="center", va="bottom",
        transform=ax.transAxes
    )


    output_dir = str(figure_dir('Figure 2'))
    os.makedirs(output_dir, exist_ok=True)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'acc_rank.png')
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
