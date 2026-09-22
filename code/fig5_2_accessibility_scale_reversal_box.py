"""Figure 5b: absolute and relative city-level accessibility gains by city size."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import BASE_YEAR, END_YEAR, SEE_CIE_PANEL_ROOT, FIGURE5_ROOT, PLOT_DPI, CITY_ORDER_4
from utils.extended_analysis import read_csv_robust, set_nature_style

LEVEL_LABELS = ["Small & Medium\nCities", "Large Cities", "Super Cities", "Mega Cities"]


def _boxplot_colored(ax, data, positions, color, *, width=0.28):
    return ax.boxplot(
        data,
        positions=positions,
        widths=width,
        vert=False,
        showfliers=False,
        patch_artist=True,
        whis=1.5,
        boxprops=dict(facecolor=color, edgecolor="black", linewidth=0.9, alpha=0.85),
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color="black", linewidth=0.9),
        capprops=dict(color="black", linewidth=0.9),
    )


def load_accessibility_changes() -> pd.DataFrame:
    path = SEE_CIE_PANEL_ROOT / f"city_{BASE_YEAR}_{END_YEAR}_index.csv"
    df = read_csv_robust(path)
    df = df[df["city_level"].isin(CITY_ORDER_4)].copy()
    df["abs_gain"] = pd.to_numeric(df["acc_delta"], errors="coerce")
    baseline = pd.to_numeric(df[f"acc_{BASE_YEAR}"], errors="coerce")
    df["rel_gain_pct"] = np.where(baseline.notna() & (baseline != 0), df["abs_gain"] / baseline * 100.0, np.nan)

    valid = df["rel_gain_pct"].dropna().to_numpy()
    if len(valid):
        lo = float(np.quantile(valid, 0.01))
        hi = float(np.quantile(valid, 0.95))
        df["rel_plot"] = df["rel_gain_pct"].clip(lower=lo, upper=hi)
    else:
        df["rel_plot"] = np.nan
    return df


def main() -> None:
    data = load_accessibility_changes()
    plt = set_nature_style()

    fig = plt.figure(figsize=(6, 4))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.22, 1.22], wspace=0.05)
    ax_left = fig.add_subplot(grid[0, 0])
    ax_right = fig.add_subplot(grid[0, 1], sharey=ax_left)
    positions = np.arange(len(CITY_ORDER_4)) + 1

    absolute = [
        pd.to_numeric(data.loc[data["city_level"] == level, "abs_gain"], errors="coerce").dropna().values
        for level in CITY_ORDER_4
    ]
    relative = [
        pd.to_numeric(data.loc[data["city_level"] == level, "rel_plot"], errors="coerce").dropna().values
        for level in CITY_ORDER_4
    ]

    _boxplot_colored(ax_left, absolute, positions, "#83aaaf", width=0.30)
    right_box = _boxplot_colored(ax_right, relative, positions, "#cfa0a2", width=0.30)

    ax_left.set_xlabel("Absolute change in accessibility")
    ax_right.set_xlabel("Relative change in accessibility (%)")
    ax_left.set_yticks(positions)
    ax_left.set_yticklabels(LEVEL_LABELS)
    ax_right.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False, length=0)
    ax_right.spines["left"].set_visible(False)
    ax_left.set_ylim(0.4, len(CITY_ORDER_4) + 0.6)

    whiskers = []
    for artist in right_box["whiskers"]:
        whiskers.extend(artist.get_xdata())
    whiskers = np.asarray(whiskers, dtype=float)
    whiskers = whiskers[np.isfinite(whiskers)]
    if len(whiskers):
        xmin, xmax = float(whiskers.min()), float(whiskers.max())
        pad = 0.06 * (xmax - xmin) if xmax > xmin else 1.0
        ax_right.set_xlim(xmin - pad, xmax + pad)

    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.25, top=0.95)
    FIGURE5_ROOT.mkdir(parents=True, exist_ok=True)
    out = FIGURE5_ROOT / "Fig_acc_scale_reversal_mirrored_BOX.png"
    fig.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
