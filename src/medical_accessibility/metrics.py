from __future__ import annotations

import numpy as np
import pandas as pd


def concentration_index_weighted(
    df: pd.DataFrame,
    rank_var: str,
    y_var: str,
    w_var: str,
) -> float:
    """Compute the weighted concentration index."""
    sub = df[[rank_var, y_var, w_var]].copy()
    sub = sub.replace([np.inf, -np.inf], np.nan)
    sub = sub.dropna(subset=[rank_var, y_var, w_var])

    sub[w_var] = pd.to_numeric(sub[w_var], errors="coerce")
    sub = sub.dropna(subset=[w_var])
    sub = sub[sub[w_var] > 0]
    if sub.shape[0] < 2:
        return np.nan

    sub[rank_var] = pd.to_numeric(sub[rank_var], errors="coerce")
    sub[y_var] = pd.to_numeric(sub[y_var], errors="coerce")
    sub = sub.dropna(subset=[rank_var, y_var])
    if sub.shape[0] < 2:
        return np.nan

    sub = sub.sort_values(by=rank_var, kind="mergesort").reset_index(drop=True)

    w = sub[w_var].to_numpy(dtype=float)
    y = sub[y_var].to_numpy(dtype=float)
    total_weight = w.sum()
    if total_weight <= 0:
        return np.nan

    mean_y = np.sum(w * y) / total_weight
    if mean_y == 0 or np.isnan(mean_y):
        return np.nan

    cum_w = np.cumsum(w)
    rank = (cum_w - 0.5 * w) / total_weight
    rank_mean = np.sum(w * rank) / total_weight
    cov = np.sum(w * (y - mean_y) * (rank - rank_mean)) / total_weight
    return float(2.0 * cov / mean_y)

