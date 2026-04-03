from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .inequality import mean_log_deviation, p_high_low_ratio, theil_index, weighted_gini


def weighted_percentile(data, weights, q):
    """Compute a weighted percentile for one-dimensional numeric arrays."""
    sorter = np.argsort(data)
    data_sorted = data[sorter]
    weights_sorted = weights[sorter]
    cumsum = np.cumsum(weights_sorted)
    return np.interp(q / 100 * cumsum[-1], cumsum, data_sorted)


def calculate_stats(df: pd.DataFrame, value_col: str = 'acc', weight_col: str = 'pop') -> dict[str, float]:
    """Summarize weighted accessibility statistics for one table or subgroup."""
    df = df[[value_col, weight_col]].copy()
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    df[weight_col] = pd.to_numeric(df[weight_col], errors='coerce')
    df = df.dropna(subset=[value_col, weight_col])
    df = df[df[weight_col] > 0].copy()
    if df.empty:
        return {
            'min': np.nan,
            'max': np.nan,
            'grid_num': 0,
            'pop_num': 0,
            'pop_mean': np.nan,
            'pop_25%': np.nan,
            'pop_median': np.nan,
            'pop_75%': np.nan,
            'pop_std': np.nan,
            'pop_gini': np.nan,
            'pop_theil': np.nan,
            'pop_MLD': np.nan,
            'p90_p10': np.nan,
            'p80_p20': np.nan,
        }

    total_grids = len(df)
    total_pop = df[weight_col].sum()
    values = df[value_col].to_numpy()
    weights = df[weight_col].to_numpy()
    return {
        'min': df[value_col].min(),
        'max': df[value_col].max(),
        'grid_num': total_grids,
        'pop_num': total_pop,
        'pop_mean': np.average(values, weights=weights),
        'pop_25%': weighted_percentile(values, weights, 25),
        'pop_median': weighted_percentile(values, weights, 50),
        'pop_75%': weighted_percentile(values, weights, 75),
        'pop_std': np.sqrt(np.cov(values, aweights=weights)),
        'pop_gini': weighted_gini(values, weights),
        'pop_theil': theil_index(values, weights),
        'pop_MLD': mean_log_deviation(values, weights),
        'p90_p10': p_high_low_ratio(values, weights, 0.9, 0.1),
        'p80_p20': p_high_low_ratio(values, weights, 0.8, 0.2),
    }


def process_yearly_files(
    input_dir: Path,
    years: Iterable[str | int],
    output_csv: Path,
    groupby_col: str | None = None,
    value_col: str = 'acc',
    weight_col: str = 'pop',
) -> pd.DataFrame:
    rows = []
    for year in years:
        candidates = [
            input_dir / f'national_{year}.csv',
            input_dir / f'national_{year}_sample.csv',
        ]
        file_path = next((path for path in candidates if path.exists()), None)
        if file_path is None:
            continue
        df = pd.read_csv(file_path)
        if groupby_col:
            if groupby_col not in df.columns:
                continue
            grouped = df.groupby(groupby_col)
            rows.extend(
                {'Year': year, groupby_col: key, **calculate_stats(group, value_col=value_col, weight_col=weight_col)}
                for key, group in grouped
            )
        else:
            rows.append({'Year': year, **calculate_stats(df, value_col=value_col, weight_col=weight_col)})

    results_df = pd.DataFrame(rows)
    if not results_df.empty and 'grid_num' in results_df.columns:
        results_df = results_df.astype({'grid_num': int, 'pop_num': int})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    return results_df
