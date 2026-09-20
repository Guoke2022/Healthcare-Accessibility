# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np

from utils.inequality_schema import ATKINSON_EPSILON


def _valid_nonnegative(x, w):
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0) & (x >= 0)
    return x[mask], w[mask]


def weighted_gini(x, w):
    """Population-weighted Gini, retaining accessibility=0 observations."""
    x, w = _valid_nonnegative(x, w)
    if x.size == 0 or w.sum() <= 0:
        return np.nan
    order = np.argsort(x, kind="mergesort")
    x, w = x[order], w[order]
    xw = x * w
    total_w = float(w.sum())
    total_xw = float(xw.sum())
    if total_xw <= 0:
        return 0.0
    cumxw = np.cumsum(xw)
    prev_cumxw = np.r_[0.0, cumxw[:-1]]
    gini = 1.0 - np.sum(w * (cumxw + prev_cumxw)) / (total_w * total_xw)
    return float(np.clip(gini, 0.0, 1.0))


def theil_index(x, w):
    """Population-weighted Theil T, retaining accessibility=0 observations.

    Zero values remain in total population and in the arithmetic mean. Their
    summand is evaluated by the mathematical limit ``r*log(r) -> 0`` as r -> 0.
    """
    x, w = _valid_nonnegative(x, w)
    if x.size == 0 or w.sum() <= 0:
        return np.nan
    total_w = float(w.sum())
    mean = float(np.sum(w * x) / total_w)
    if mean <= 0:
        return 0.0
    r = x / mean
    # Evaluate the zero case by its limit without changing the analysis universe.
    # np.log(..., where=...) avoids evaluating log(0); zero observations remain
    # in total_w and in mean and contribute exactly 0 to r*log(r).
    log_r = np.zeros_like(r)
    np.log(r, out=log_r, where=r != 0)
    terms = r * log_r
    value = float(np.sum(w * terms) / total_w)
    return 0.0 if abs(value) < 1e-15 else value


def atkinson_index(x, w, epsilon: float = ATKINSON_EPSILON):

    x, w = _valid_nonnegative(x, w)
    if x.size == 0 or w.sum() <= 0:
        return np.nan
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon < 0 or epsilon >= 1:
        raise ValueError("Atkinson epsilon must satisfy 0 <= epsilon < 1 when zero accessibility is retained")
    total_w = float(w.sum())
    mean = float(np.sum(w * x) / total_w)
    if mean <= 0:
        return 0.0
    if epsilon == 0:
        return 0.0
    power = 1.0 - epsilon
    ede = float((np.sum(w * np.power(x, power)) / total_w) ** (1.0 / power))
    value = 1.0 - ede / mean
    return float(np.clip(value, 0.0, 1.0))


def atkinson_05(x, w):

    return atkinson_index(x, w, epsilon=ATKINSON_EPSILON)


def weighted_quantile(x, w, q):
    """Left-continuous weighted empirical quantile; q in [0,1]."""
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if not 0 <= q <= 1:
        raise ValueError(f"q must be in [0, 1], got {q}")
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[mask], w[mask]
    if x.size == 0 or w.sum() <= 0:
        return np.nan
    order = np.argsort(x, kind="mergesort")
    x, w = x[order], w[order]
    cw = np.cumsum(w) / np.sum(w)
    return float(x[np.searchsorted(cw, q, side="left")])


def p_high_low_ratio(x, w, q_high=0.9, q_low=0.1):
    x, w = _valid_nonnegative(x, w)
    if x.size == 0:
        return np.nan
    p_high = weighted_quantile(x, w, q_high)
    p_low = weighted_quantile(x, w, q_low)
    if not np.isfinite(p_low) or p_low <= 0:
        return np.nan
    return float(p_high / p_low)


def squared_coeff_variation(x, w):
    x, w = _valid_nonnegative(x, w)
    if x.size == 0 or w.sum() <= 0:
        return np.nan
    mean = np.average(x, weights=w)
    if mean <= 0:
        return 0.0
    var = np.average((x - mean) ** 2, weights=w)
    return float(var / (mean ** 2))


def palma_ratio(x, w):
    x, w = _valid_nonnegative(x, w)
    if x.size == 0 or w.sum() <= 0:
        return np.nan
    order = np.argsort(x, kind="mergesort")
    x, w = x[order], w[order]
    total_pop = float(w.sum())
    total_acc = float(np.sum(x * w))
    if total_acc <= 0:
        return np.nan
    cum_pop = np.cumsum(w) / total_pop
    cum_acc = np.cumsum(x * w) / total_acc
    low40_share = np.interp(0.4, cum_pop, cum_acc)
    top10_share = 1 - np.interp(0.9, cum_pop, cum_acc)
    if low40_share <= 0:
        return np.nan
    return float(top10_share / low40_share)


def ttvi(x, w, tail_pct):
    x, w = _valid_nonnegative(x, w)
    if x.size == 0 or w.sum() <= 0:
        return np.nan
    if not 0 < tail_pct < 0.5:
        raise ValueError("tail_pct must be between 0 and 0.5")
    mean_all = np.average(x, weights=w)
    var_all = np.average((x - mean_all) ** 2, weights=w)
    if var_all <= 0:
        return 0.0
    low_thresh = weighted_quantile(x, w, tail_pct)
    high_thresh = weighted_quantile(x, w, 1 - tail_pct)
    tail = (x <= low_thresh) | (x >= high_thresh)
    if not np.any(tail):
        return np.nan
    var_tail = np.average((x[tail] - mean_all) ** 2, weights=w[tail])
    return float(var_tail / var_all)


def weighted_percentile(data, weights, q):
    data = np.asarray(data, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(data) & np.isfinite(weights) & (weights > 0)
    data, weights = data[valid], weights[valid]
    if data.size == 0:
        return np.nan
    sorter = np.argsort(data, kind="mergesort")
    data_sorted = data[sorter]
    weights_sorted = weights[sorter]
    cumsum = np.cumsum(weights_sorted)
    return float(np.interp(q / 100 * cumsum[-1], cumsum, data_sorted))
