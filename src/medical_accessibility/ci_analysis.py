from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from .metrics import concentration_index_weighted


YEAR_COLOR = {
    2014: "#084081",
    2015: "#2b8cbe",
    2016: "#4eb3d3",
    2017: "#7bccc4",
    2018: "#a8ddb5",
    2019: "#d9f0a3",
    2020: "#fee391",
    2021: "#fec44f",
    2022: "#fe9929",
    2023: "#e69f00",
}


@dataclass(frozen=True)
class AnalysisSpec:
    name: str
    rank_var: str
    y_var: str = "acc"
    w_var: str = "pop"
    valid_rank_rule: str = "gt0"
    xlabel: str = ""
    xlim: tuple[float, float] | None = None


def load_yearly_file(input_dir: Path, year: int) -> pd.DataFrame:
    candidates = [
        input_dir / f"national_{year}.csv",
        input_dir / f"national_{year}_sample.csv",
    ]
    for file_path in candidates:
        if file_path.exists():
            return pd.read_csv(file_path)
    raise FileNotFoundError(f"Missing input file for year {year} in {input_dir}")


def filter_analysis_frame(
    df: pd.DataFrame,
    rank_var: str,
    y_var: str,
    w_var: str,
    valid_rank_rule: str,
) -> pd.DataFrame:
    data = df.copy()
    data = data.replace([np.inf, -np.inf], np.nan)

    data[rank_var] = pd.to_numeric(data.get(rank_var), errors="coerce")
    data[y_var] = pd.to_numeric(data.get(y_var), errors="coerce")
    data[w_var] = pd.to_numeric(data.get(w_var), errors="coerce")
    data = data.dropna(subset=[rank_var, y_var, w_var])
    data = data[data[w_var] > 0].copy()

    if valid_rank_rule == "gt0":
        data = data[data[rank_var] > 0].copy()
    elif valid_rank_rule == "ge0":
        data = data[data[rank_var] >= 0].copy()
    else:
        raise ValueError("valid_rank_rule must be 'gt0' or 'ge0'")

    return data


def compute_ci_series(
    input_dir: Path,
    years: Iterable[int],
    spec: AnalysisSpec,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for year in years:
        try:
            df = load_yearly_file(input_dir, year)
        except FileNotFoundError:
            rows.append({"year": year, "CI": np.nan})
            continue

        df = filter_analysis_frame(
            df=df,
            rank_var=spec.rank_var,
            y_var=spec.y_var,
            w_var=spec.w_var,
            valid_rank_rule=spec.valid_rank_rule,
        )
        if df.shape[0] < 2:
            rows.append({"year": year, "CI": np.nan})
            continue

        ci = concentration_index_weighted(
            df=df,
            rank_var=spec.rank_var,
            y_var=spec.y_var,
            w_var=spec.w_var,
        )
        rows.append({"year": year, "CI": ci})

    return pd.DataFrame(rows)


def plot_concentration_curves(
    input_dir: Path,
    ci_df: pd.DataFrame,
    output_path: Path,
    spec: AnalysisSpec,
    years: Iterable[int],
) -> None:
    plt.figure(figsize=(8.2, 8))
    ci_lookup = dict(zip(ci_df["year"], ci_df["CI"]))

    for year in years:
        try:
            df = load_yearly_file(input_dir, year)
        except FileNotFoundError:
            continue

        df = filter_analysis_frame(
            df=df,
            rank_var=spec.rank_var,
            y_var=spec.y_var,
            w_var=spec.w_var,
            valid_rank_rule=spec.valid_rank_rule,
        )
        if df.empty:
            continue

        df = df.sort_values(by=spec.rank_var, kind="mergesort").reset_index(drop=True)
        w = df[spec.w_var].to_numpy(dtype=float)
        y = df[spec.y_var].to_numpy(dtype=float)
        total_weight = w.sum()
        total_y = np.sum(w * y)
        if total_weight <= 0 or total_y <= 0:
            continue

        cum_pop = np.cumsum(w) / total_weight
        cum_y = np.cumsum(w * y) / total_y
        ci_val = ci_lookup.get(year, np.nan)
        label = f"{year} (CI={ci_val:.3f})" if pd.notna(ci_val) else str(year)

        plt.plot(
            cum_pop,
            cum_y,
            color=YEAR_COLOR.get(year, "#808080"),
            linewidth=4,
            label=label,
        )

    ax = plt.gca()
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.2)
    plt.xlabel(spec.xlabel)
    plt.ylabel("Cumulative share of accessibility")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=400)
    plt.close()


def plot_ci_trend(
    ci_df: pd.DataFrame,
    output_path: Path,
    xlim: tuple[float, float] | None = None,
) -> None:
    data = ci_df.sort_values(by="year").dropna(subset=["CI"]).copy()
    years = data["year"].to_numpy()
    ci_vals = data["CI"].to_numpy()

    plt.figure(figsize=(2.8, 3.4))
    ax = plt.gca()
    ax.plot(ci_vals, years, linestyle="-", color="gray", alpha=0.4, linewidth=2)

    for year, ci in zip(years, ci_vals):
        color = YEAR_COLOR.get(year, "#bdbdbd") if year in {2014, 2023} else "#d9d9d9"
        ax.hlines(y=year, xmin=0, xmax=ci, colors=color, linewidth=4, zorder=2)
        ax.scatter(
            [ci],
            [year],
            s=80 if year in {2014, 2023} else 45,
            c=[YEAR_COLOR.get(year, "gray") if year in {2014, 2023} else "gray"],
            edgecolors="white",
            linewidths=1,
            zorder=3,
        )
        if year in {2014, 2023}:
            ax.text(ci + 0.01, year, f"CI={ci:.3f}", va="center", ha="left", fontsize=10)

    ax.invert_yaxis()
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xticks([])
    ax.set_ylabel("")
    ax.set_yticks([year for year in [2014, 2017, 2020, 2023] if year in years])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=400, transparent=True, bbox_inches="tight")
    plt.close()


def plot_weighted_kde(
    input_dir: Path,
    output_path: Path,
    x_var: str,
    x_label: str,
    years: Iterable[int],
    filter_fn,
    bw_adjust: float,
    formatter=None,
) -> None:
    import seaborn as sns

    plt.figure(figsize=(10, 5))
    highlight_years = {2014, 2023}

    for year in years:
        try:
            df = load_yearly_file(input_dir, year)
        except FileNotFoundError:
            continue

        df = filter_fn(df)
        if df.empty:
            continue

        is_highlight = year in highlight_years
        sns.kdeplot(
            data=df,
            x=x_var,
            weights="pop",
            clip=(0, df[x_var].max()),
            label=str(year),
            bw_adjust=bw_adjust,
            color=YEAR_COLOR.get(year, "#bdbdbd"),
            linewidth=5 if is_highlight else 2,
            linestyle="-" if is_highlight else "--",
            alpha=1.0,
            zorder=10 if is_highlight else 3,
        )

    ax = plt.gca()
    if formatter is not None:
        ax.xaxis.set_major_formatter(formatter)
    plt.xlabel(x_label)
    plt.ylabel("Density")
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=400, transparent=True, bbox_inches="tight")
    plt.close()


def make_gdp_formatter():
    return mticker.FuncFormatter(lambda x, _: f"{x / 10000:.0f}")
