# -*- coding: utf-8 -*-
"""Figure 5.2 accessibility scale-reversal boxplot.

Separated from 5_4_see_cie_descriptives.py so the descriptive analysis script
writes tabular outputs, while this script renders the manuscript figure from the
released/derived summary table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import FIGURE5_ROOT, PLOT_DPI, SEE_CIE_DESCRIPTIVE_ROOT, CITY_ORDER_4
from utils.extended_analysis import set_nature_style

LEVEL_LABELS = ["Small & Medium\nCities", "Large Cities", "Super Cities", "Mega Cities"]


def _boxplot_colored(ax, data, positions, color, *, vert=True, width=.28):
    return ax.boxplot(
        data, positions=positions, widths=width, vert=vert,
        showfliers=False, patch_artist=True, whis=1.5,
        boxprops=dict(facecolor=color, edgecolor="black", linewidth=.9, alpha=.85),
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color="black", linewidth=.9),
        capprops=dict(color="black", linewidth=.9),
    )


def main() -> None:
    path = SEE_CIE_DESCRIPTIVE_ROOT / "city_accessibility_absolute_relative_change.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing prerequisite table: {path}. Run 5_4_see_cie_descriptives.py first."
        )

    acc_city = pd.read_csv(path, encoding="utf-8-sig")
    plt = set_nature_style()

    fig = plt.figure(figsize=(6, 4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.22, 1.22], wspace=.05)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1], sharey=axL)
    pos_h = np.arange(len(CITY_ORDER_4)) + 1

    data_abs = [
        pd.to_numeric(acc_city.loc[acc_city["city_level"] == level, "abs_gain"], errors="coerce").dropna().values
        for level in CITY_ORDER_4
    ]
    data_rel = [
        pd.to_numeric(acc_city.loc[acc_city["city_level"] == level, "rel_plot"], errors="coerce").dropna().values
        for level in CITY_ORDER_4
    ]

    _boxplot_colored(axL, data_abs, pos_h, "#83aaaf", vert=False, width=.30)
    bpR = _boxplot_colored(axR, data_rel, pos_h, "#cfa0a2", vert=False, width=.30)

    axL.set_xlabel("Absolute change in accessibility")
    axR.set_xlabel("Relative change in accessibility (%)")
    axL.set_yticks(pos_h)
    axL.set_yticklabels(LEVEL_LABELS)
    axR.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False, length=0)
    axR.spines["left"].set_visible(False)
    axL.set_ylim(.4, len(CITY_ORDER_4) + .6)

    whisk_x = []
    for w in bpR["whiskers"]:
        whisk_x.extend(w.get_xdata())
    whisk_x = np.asarray(whisk_x, dtype=float)
    whisk_x = whisk_x[np.isfinite(whisk_x)]
    if len(whisk_x):
        xmin, xmax = float(whisk_x.min()), float(whisk_x.max())
        pad = .06 * (xmax - xmin) if xmax > xmin else 1.0
        axR.set_xlim(xmin - pad, xmax + pad)

    fig.subplots_adjust(left=.22, right=.98, bottom=.25, top=.95)
    FIGURE5_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE5_ROOT / "Fig_acc_scale_reversal_mirrored_BOX.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"fig5_2 done -> {FIGURE5_ROOT / 'Fig_acc_scale_reversal_mirrored_BOX.png'}")


if __name__ == "__main__":
    main()
