"""Module utilities for concentration index."""


from __future__ import annotations

import numpy as np
import pandas as pd


def _aggregate_rank_ties(df: pd.DataFrame, rank_var: str, y_var: str, w_var: str) -> pd.DataFrame:
    """Helper for _aggregate_rank_ties."""
    sub = df[[rank_var, y_var, w_var]].copy().replace([np.inf, -np.inf], np.nan)
    for c in [rank_var, y_var, w_var]:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna(subset=[rank_var, y_var, w_var])
    sub = sub[sub[w_var] > 0]
    if sub.empty:
        return pd.DataFrame(columns=[rank_var, "w", "yw", "y"])

    sub["_yw"] = sub[y_var] * sub[w_var]
    g = sub.groupby(rank_var, sort=True, as_index=False).agg(w=(w_var, "sum"), yw=("_yw", "sum"))
    g = g[g["w"] > 0].copy()
    g["y"] = g["yw"] / g["w"]
    return g.sort_values(rank_var, kind="mergesort").reset_index(drop=True)


def concentration_index_weighted(df: pd.DataFrame, rank_var: str, y_var: str, w_var: str) -> float:
    """Helper for concentration_index_weighted."""
    g = _aggregate_rank_ties(df, rank_var, y_var, w_var)
    if len(g) < 2:
        return np.nan

    w = g["w"].to_numpy(dtype=np.float64)
    y = g["y"].to_numpy(dtype=np.float64)
    W = float(w.sum())
    if W <= 0:
        return np.nan
    mu = float(np.sum(w * y) / W)
    if not np.isfinite(mu) or mu == 0:
        return np.nan


    cum_w = np.cumsum(w)
    r = (cum_w - 0.5 * w) / W
    r_bar = float(np.sum(w * r) / W)
    cov_wr = float(np.sum(w * (y - mu) * (r - r_bar)) / W)
    return float(2.0 * cov_wr / mu)


def concentration_curve_weighted(df: pd.DataFrame, rank_var: str, y_var: str, w_var: str):
    """Helper for concentration_curve_weighted."""
    g = _aggregate_rank_ties(df, rank_var, y_var, w_var)
    if len(g) < 2:
        return None
    W = float(g["w"].sum())
    Y = float(g["yw"].sum())
    if W <= 0 or Y <= 0:
        return None
    x = np.r_[0.0, np.cumsum(g["w"].to_numpy(dtype=np.float64)) / W]
    y = np.r_[0.0, np.cumsum(g["yw"].to_numpy(dtype=np.float64)) / Y]
    return x, y


if __name__ == "__main__":

    d = pd.DataFrame({"rank": [1, 1, 2, 2], "y": [1, 9, 2, 8], "w": [1, 1, 1, 1]})
    a = concentration_index_weighted(d, "rank", "y", "w")
    b = concentration_index_weighted(d.iloc[[1, 0, 3, 2]], "rank", "y", "w")
    print("tie-order invariant:", a, b)
