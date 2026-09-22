# -*- coding: utf-8 -*-
"""Figure 1.1 travel-time threshold trends for National and all provinces."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from config import PLOT_DPI
from utils.figure_data import figure_dir, read_stats

PROVINCE_EN = {
    "北京市": "Beijing", "山西省": "Shanxi", "内蒙古自治区": "Nei Mongol", "吉林省": "Jilin",
    "黑龙江省": "Heilongjiang", "安徽省": "Anhui", "江西省": "Jiangxi", "河南省": "Henan",
    "湖北省": "Hubei", "湖南省": "Hunan", "重庆市": "Chongqing", "四川省": "Sichuan",
    "贵州省": "Guizhou", "云南省": "Yunnan", "西藏自治区": "Tibet", "陕西省": "Shaanxi",
    "甘肃省": "Gansu", "青海省": "Qinghai", "宁夏回族自治区": "Ningxia", "新疆维吾尔自治区": "Xinjiang",
    "辽宁省": "Liaoning", "河北省": "Hebei", "天津市": "Tianjin", "山东省": "Shandong",
    "江苏省": "Jiangsu", "上海市": "Shanghai", "浙江省": "Zhejiang", "福建省": "Fujian",
    "广东省": "Guangdong", "广西壮族自治区": "Guangxi", "海南省": "Hainan",
}


def plot_panel(data, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    linewidth = 10
    alpha = 0.25
    tick_fs = 40
    title_fs = 80

    line1, = ax.plot(data["Year"], data["pop_pct_0_30"], marker="o", label="<30 mins", linewidth=linewidth, color="#eeeed6")
    line2, = ax.plot(data["Year"], data["pop_pct_lt_60"], marker="s", label="<60 mins", linewidth=linewidth, color="#43a2ca")
    line3, = ax.plot(data["Year"], data["pop_pct_lt_90"], marker="^", label="<90 mins", linewidth=linewidth, color="#5ca18f")

    ax.fill_between(data["Year"], data["pop_pct_lt_60"], data["pop_pct_lt_90"], color=line3.get_color(), alpha=alpha)
    ax.fill_between(data["Year"], data["pop_pct_0_30"], data["pop_pct_lt_60"], color=line2.get_color(), alpha=alpha)
    ax.fill_between(data["Year"], data["pop_pct_0_30"], color=line1.get_color(), alpha=alpha)

    if title == "Tibet":
        ax.set_ylim(0.0, 0.6)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6], ["0%", "20%", "40%", "60%"], fontsize=tick_fs)
    else:
        ax.set_ylim(0.2, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["20%", "40%", "60%", "80%", "100%"], fontsize=tick_fs)
    ax.set_xlim(2014, None)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.grid(axis="x", linestyle="-", linewidth=2, alpha=0.8)
    ax.tick_params(axis="x", labelbottom=False)
    ax.tick_params(axis="y", labelleft=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title(title, fontsize=title_fs)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out_dir = figure_dir("Figure 1", "province_time_proportion")

    national = read_stats("national", "travel_time")
    plot_panel(national, "National", out_dir / "National.png")

    provincial = read_stats("provincial", "travel_time")
    for province, data in provincial.groupby("省级", sort=False):
        title = PROVINCE_EN.get(str(province), str(province))
        out = out_dir / f"{title}_travel_time_trends.png"
        plot_panel(data.sort_values("Year"), title, out)


if __name__ == "__main__":
    main()
