from __future__ import annotations

import numpy as np


def weighted_gini(x, w):
    """Compute the weighted Gini coefficient."""
    x = np.asarray(x)
    w = np.asarray(w)
    mask = ~(np.isnan(x) | np.isnan(w))
    x = x[mask]
    w = w[mask]
    if x.size == 0 or w.sum() == 0:
        return np.nan
    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    w_sorted = w[sorted_idx]
    cumw = np.cumsum(w_sorted)
    cumxw = np.cumsum(x_sorted * w_sorted)
    total_w = cumw[-1]
    total_xw = cumxw[-1]
    if total_w <= 0 or total_xw <= 0:
        return np.nan
    return 1 - 2 * np.sum(cumxw * w_sorted) / (total_xw * total_w)


def theil_index(x, w):
    """Compute the weighted Theil index."""
    x = np.asarray(x)
    w = np.asarray(w)
    mask = (x > 0) & (w > 0) & (~np.isnan(x)) & (~np.isnan(w))
    x = x[mask]
    w = w[mask]
    if x.size == 0 or np.sum(w) == 0:
        return np.nan
    weighted_mean = np.average(x, weights=w)
    if weighted_mean == 0:
        return np.nan
    return np.sum(w * (x / weighted_mean) * np.log(x / weighted_mean)) / np.sum(w)


def mean_log_deviation(x, w):
    """Compute the weighted mean log deviation."""
    x = np.asarray(x)
    w = np.asarray(w)
    mask = (x > 0) & (w > 0) & (~np.isnan(x)) & (~np.isnan(w))
    x = x[mask]
    w = w[mask]
    if x.size == 0 or np.sum(w) == 0:
        return np.nan
    weighted_mean = np.average(x, weights=w)
    return np.sum(w * np.log(weighted_mean / x)) / np.sum(w)


def weighted_quantile(x, w, q):
    x = np.asarray(x)
    w = np.asarray(w)
    mask = (~np.isnan(x)) & (~np.isnan(w)) & (w > 0)
    x = x[mask]
    w = w[mask]
    if x.size == 0 or np.sum(w) == 0:
        return np.nan
    idx = np.argsort(x)
    x_sorted = x[idx]
    w_sorted = w[idx]
    cw = np.cumsum(w_sorted)
    cw = cw / cw[-1]
    return x_sorted[np.searchsorted(cw, q, side="left")]


def p_high_low_ratio(x, w, q_high=0.9, q_low=0.1):
    x = np.asarray(x)
    w = np.asarray(w)
    mask = (~np.isnan(x)) & (~np.isnan(w)) & (w > 0) & (x > 0)
    x = x[mask]
    w = w[mask]
    if x.size == 0:
        return np.nan
    p_high = weighted_quantile(x, w, q_high)
    p_low = weighted_quantile(x, w, q_low)
    if np.isnan(p_low) or p_low == 0:
        return np.nan
    return p_high / p_low
