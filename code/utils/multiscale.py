# -*- coding: utf-8 -*-
"""Module utilities for multiscale."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import re

import numpy as np
import pandas as pd

from config import (
    ACCESSIBILITY_ROOT,
    MULTISCALE_ROOT,
    VALID_TIME_MIN,
)
from config import (
    PREPARED_ROOT,
    ONLY_PROVINCES,
    REQUIRE_COMPLETE_RUN_WHEN_UNFILTERED,
)

NEW_TO_OLD = {
    "population": "pop",
    "accessibility": "acc",
    "nearest_hospital_time_min": "travel_time",
}

BASE_GRID_COLS = [
    "grid_id",
    "population",
    "lon",
    "lat",
    "accessibility",
    "nearest_hospital_time_min",
    "n_reachable_hospitals",
    "grid_snap_distance_km",
    "grid_snap_kind",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def accessibility_grid_dir(year: int, scope: str, profile: str) -> Path:
    return ACCESSIBILITY_ROOT / str(year) / scope / profile / "grids"


def matched_parts_dir(year: int, scope: str, profile: str) -> Path:
    return MULTISCALE_ROOT / str(year) / scope / profile / "parts"


def discover_parts(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(folder.glob("province_*.parquet"))


PROVINCE_PART_RE = re.compile(r"^province_(\d{3})_(.+)\.parquet$")


def expected_province_keys() -> set[tuple[int, str]]:
    """Helper for expected_province_keys."""
    provinces_path = PREPARED_ROOT / "common" / "provinces.parquet"
    if not provinces_path.exists():
        raise FileNotFoundError(f"缺少 0_1 省级基准表：{provinces_path}")
    df = pd.read_parquet(provinces_path, columns=["province_id", "province_name"])
    if df[["province_id", "province_name"]].isna().any().any():
        raise ValueError(f"省级基准表存在缺失 province_id/province_name：{provinces_path}")
    keys = set(zip(df["province_id"].astype(int), df["province_name"].astype(str)))
    if ONLY_PROVINCES is not None:
        selected = {str(x) for x in ONLY_PROVINCES}
        keys = {k for k in keys if k[1] in selected}
        missing_names = selected - {k[1] for k in keys}
        if missing_names:
            raise ValueError(f"ONLY_PROVINCES 中有名称不在 0_1 省级基准表：{sorted(missing_names)}")
    return keys


def province_keys_from_parts(parts: Iterable[Path], label: str = "province parts") -> set[tuple[int, str]]:
    """Helper for province_keys_from_parts."""
    keys = []
    malformed = []
    for path in parts:
        m = PROVINCE_PART_RE.match(Path(path).name)
        if not m:
            malformed.append(Path(path).name)
            continue
        keys.append((int(m.group(1)), m.group(2)))
    if malformed:
        raise RuntimeError(f"{label} 存在无法解析的省级分片文件名：{malformed[:10]}")
    if len(keys) != len(set(keys)):
        from collections import Counter
        dup = [k for k, n in Counter(keys).items() if n > 1]
        raise RuntimeError(f"{label} 存在重复省级分片：{dup[:10]}")
    return set(keys)


def validate_complete_province_parts(
    parts: Iterable[Path],
    label: str,
    *,
    require_complete: bool | None = None,
) -> set[tuple[int, str]]:
    """Helper for validate_complete_province_parts."""


    parts = list(parts)
    if not parts:
        raise FileNotFoundError(f"{label}: 未找到任何 province_*.parquet")
    actual = province_keys_from_parts(parts, label)
    if require_complete is None:
        require_complete = REQUIRE_COMPLETE_RUN_WHEN_UNFILTERED or ONLY_PROVINCES is not None
    if require_complete:
        expected = expected_province_keys()
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise RuntimeError(
                f"{label} 省级分片集合不完整/不一致，拒绝继续。"
                f"\nexpected={len(expected)}, actual={len(actual)}"
                f"\nmissing={missing}"
                f"\nextra={extra}"
            )
    return actual


def parquet_columns(path: Path) -> list[str]:
    """Helper for parquet_columns."""
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError(
            "读取新代码 Parquet 需要 pyarrow；请执行：pip install pyarrow"
        ) from e
    return pq.ParquetFile(path).schema.names


def read_new_grid_part(path: Path) -> pd.DataFrame:
    available = parquet_columns(path)
    required = [
        "grid_id", "population", "lon", "lat",
        "accessibility", "nearest_hospital_time_min",
    ]
    missing = [c for c in required if c not in available]
    if missing:
        raise KeyError(
            f"{path} 缺少 1_2 应输出的字段 {missing}；实际字段={available}"
        )
    usecols = [c for c in BASE_GRID_COLS if c in available]
    df = pd.read_parquet(path, columns=usecols)
    return df.rename(columns=NEW_TO_OLD)


def weighted_percentile(data, weights, q):
    data = np.asarray(data, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(data) & np.isfinite(weights) & (weights > 0)
    data = data[valid]
    weights = weights[valid]
    if len(data) == 0 or weights.sum() <= 0:
        return np.nan

    sorter = np.argsort(data, kind="mergesort")
    data_sorted = data[sorter]
    weights_sorted = weights[sorter]
    cumsum = np.cumsum(weights_sorted, dtype=np.float64)
    target = q / 100.0 * cumsum[-1]
    return float(np.interp(target, cumsum, data_sorted))


def weighted_std(data, weights):
    data = np.asarray(data, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(data) & np.isfinite(weights) & (weights > 0)
    data = data[valid]
    weights = weights[valid]
    if len(data) == 0 or weights.sum() <= 0:
        return np.nan
    mean = np.average(data, weights=weights)
    return float(np.sqrt(np.average((data - mean) ** 2, weights=weights)))


def _inequality_functions():
    """Return the formal inequality functions from the single project implementation."""
    try:
        from utils.inequality_metrics import weighted_gini, theil_index, atkinson_05, p_high_low_ratio
    except ImportError as e:
        raise ImportError("Cannot import utils.inequality_metrics; formal inequality metrics are unavailable.") from e
    return weighted_gini, theil_index, atkinson_05, p_high_low_ratio

def _sorted_left_quantile(x_sorted: np.ndarray, w_sorted: np.ndarray, q: float) -> float:
    """Left-continuous weighted empirical quantile from already-sorted values."""
    if x_sorted.size == 0 or w_sorted.sum() <= 0:
        return np.nan
    cw = np.cumsum(w_sorted, dtype=np.float64) / np.sum(w_sorted)
    idx = int(np.searchsorted(cw, q, side="left"))
    if idx >= x_sorted.size:
        idx = x_sorted.size - 1
    return float(x_sorted[idx])


def _gini_from_sorted_nonnegative(x_sorted: np.ndarray, w_sorted: np.ndarray) -> float:
    """Exact weighted-Gini formula from already-sorted nonnegative values.

    This is algebraically identical to ``utils.inequality_metrics.weighted_gini``;
    only the redundant internal argsort is removed.
    """
    if x_sorted.size == 0 or w_sorted.sum() <= 0:
        return np.nan
    xw = x_sorted * w_sorted
    total_w = float(w_sorted.sum())
    total_xw = float(xw.sum())
    if total_xw <= 0:
        return 0.0
    cumxw = np.cumsum(xw)
    prev_cumxw = np.r_[0.0, cumxw[:-1]]
    gini = 1.0 - np.sum(w_sorted * (cumxw + prev_cumxw)) / (total_w * total_xw)
    return float(np.clip(gini, 0.0, 1.0))


def _distribution_stats_single_sort(values, weights) -> dict:
    """Compute weighted distribution statistics with a single stable sort.

    Gini, Theil T and Atkinson (ε=0.5) all use finite accessibility >= 0
    with finite positive population weights. Zero accessibility is therefore
    retained consistently across all formal inequality outcomes.
    """
    x0 = np.asarray(values, dtype=np.float64)
    w0 = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(x0) & np.isfinite(w0) & (w0 > 0)
    x = x0[valid]
    w = w0[valid]

    empty = {
        "pop_mean": np.nan,
        "pop_25%": np.nan,
        "pop_median": np.nan,
        "pop_75%": np.nan,
        "pop_std": np.nan,
        "pop_gini": np.nan,
        "pop_theil": np.nan,
        "pop_atkinson_05": np.nan,
        "zero_access_pop_pct": np.nan,
        "p90_p10": np.nan,
        "p80_p20": np.nan,
    }
    if x.size == 0 or w.sum() <= 0:
        return empty

    mean = float(np.average(x, weights=w))
    std = float(np.sqrt(np.average((x - mean) ** 2, weights=w)))

    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ws = w[order]
    cumsum = np.cumsum(ws, dtype=np.float64)
    total_w = float(cumsum[-1])

    def interp_pct(q_pct: float) -> float:
        return float(np.interp(q_pct / 100.0 * total_w, cumsum, xs))

    out = {
        "pop_mean": mean,
        "pop_25%": interp_pct(25.0),
        "pop_median": interp_pct(50.0),
        "pop_75%": interp_pct(75.0),
        "pop_std": std,
    }

    nonneg_unsorted = x >= 0
    xi = x[nonneg_unsorted]
    wi = w[nonneg_unsorted]
    nonneg_sorted = xs >= 0
    xis = xs[nonneg_sorted]
    wis = ws[nonneg_sorted]

    if xi.size == 0 or wi.sum() <= 0:
        out.update({k: empty[k] for k in (
            "pop_gini", "pop_theil", "pop_atkinson_05",
            "zero_access_pop_pct", "p90_p10", "p80_p20",
        )})
        return out

    _, theil_index, atkinson_05, _ = _inequality_functions()
    zero_access_pop_pct = float(wi[xi == 0].sum() / wi.sum() * 100.0)
    p90 = _sorted_left_quantile(xis, wis, 0.9)
    p10 = _sorted_left_quantile(xis, wis, 0.1)
    p80 = _sorted_left_quantile(xis, wis, 0.8)
    p20 = _sorted_left_quantile(xis, wis, 0.2)

    out.update({
        "pop_gini": _gini_from_sorted_nonnegative(xis, wis),
        "pop_theil": float(theil_index(xi, wi)),
        "pop_atkinson_05": float(atkinson_05(xi, wi)),
        "zero_access_pop_pct": zero_access_pop_pct,
        "p90_p10": float(p90 / p10) if np.isfinite(p10) and p10 > 0 else np.nan,
        "p80_p20": float(p80 / p20) if np.isfinite(p20) and p20 > 0 else np.nan,
    })
    return out

def _safe_inequality(values, weights):
    """Inequality-only helper used by callers/tests."""
    stats = _distribution_stats_single_sort(values, weights)
    return {
        key: stats[key]
        for key in (
            "pop_gini", "pop_theil", "pop_atkinson_05",
            "zero_access_pop_pct", "p90_p10", "p80_p20",
        )
    }

def _snap_invalid_pop_pct(df: pd.DataFrame, pop_all: np.ndarray) -> float:
    """Helper for _snap_invalid_pop_pct."""


    if "grid_snap_distance_km" not in df.columns:
        return np.nan
    d = pd.to_numeric(df["grid_snap_distance_km"], errors="coerce").to_numpy(dtype=np.float64)
    invalid = ~np.isfinite(d)
    total = float(pop_all.sum())
    return float(pop_all[invalid].sum() / total) if total > 0 else np.nan


def calculate_travel_time_stats(df: pd.DataFrame) -> dict:
    """Helper for calculate_travel_time_stats."""


    if len(df) == 0:
        return {}

    pop_all = pd.to_numeric(df["pop"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    time_all = pd.to_numeric(df["travel_time"], errors="coerce").to_numpy(dtype=np.float64)

    pop_valid_base = np.isfinite(pop_all) & (pop_all >= 0)
    pop_all = np.where(pop_valid_base, pop_all, 0.0)

    total_grids = int(len(df))
    total_pop = float(pop_all.sum())
    snap_invalid_pop_pct = _snap_invalid_pop_pct(df, pop_all)

    reachable = np.isfinite(time_all) & (time_all >= VALID_TIME_MIN)
    t = time_all[reachable]
    w = pop_all[reachable]
    positive_w = w > 0
    t = t[positive_w]
    w = w[positive_w]

    reachable_pop = float(w.sum())
    reachable_grid_num = int(reachable.sum())

    def pop_for(mask):
        return float(pop_all[mask].sum())

    time_0_30 = reachable & (time_all <= 30)
    time_30_60 = reachable & (time_all > 30) & (time_all <= 60)
    time_60_90 = reachable & (time_all > 60) & (time_all <= 90)
    time_lt_60 = reachable & (time_all <= 60)
    time_lt_90 = reachable & (time_all <= 90)

    def pct(v):
        return v / total_pop if total_pop > 0 else np.nan

    pop_num_0_30 = pop_for(time_0_30)
    pop_num_30_60 = pop_for(time_30_60)
    pop_num_60_90 = pop_for(time_60_90)
    pop_num_lt_60 = pop_for(time_lt_60)
    pop_num_lt_90 = pop_for(time_lt_90)

    stats = {
        "pop_num_0_30": pop_num_0_30,
        "pop_pct_0_30": pct(pop_num_0_30),
        "pop_num_30_60": pop_num_30_60,
        "pop_pct_30_60": pct(pop_num_30_60),
        "pop_num_60_90": pop_num_60_90,
        "pop_pct_60_90": pct(pop_num_60_90),
        "pop_num_lt_60": pop_num_lt_60,
        "pop_pct_lt_60": pct(pop_num_lt_60),
        "pop_num_lt_90": pop_num_lt_90,
        "pop_pct_lt_90": pct(pop_num_lt_90),
        "min": float(np.min(t)) if len(t) else np.nan,
        "max": float(np.max(t)) if len(t) else np.nan,
        "grid_num": total_grids,
        "pop_num": total_pop,
        "snap_invalid_pop_pct": snap_invalid_pop_pct,
        "reachable_grid_num": reachable_grid_num,
        "reachable_pop_num": reachable_pop,
        "reachable_pop_pct": pct(reachable_pop),
        "unreachable_pop_pct": 1.0 - pct(reachable_pop) if total_pop > 0 else np.nan,
        "inequality_population_universe": "reachable_population_only",
    }
    stats.update(_distribution_stats_single_sort(t, w))
    return stats


def theil_city_decomposition(
    df: pd.DataFrame,
    *,
    value_col: str = "acc",
    weight_col: str = "pop",
    city_col: str = "city_name_norm",
) -> dict:
    """Exact Theil-T decomposition using the same all-nonnegative population universe.

    Zero-access observations stay in city and national population denominators and
    means. Terms involving ``x log x`` are evaluated by their zero limit. Cities with
    zero total accessibility contribute zero to access-share-weighted log terms while
    remaining in the population denominator, preserving the exact identity.
    """
    empty = {
        "pop_theil_within_city": np.nan,
        "pop_theil_between_city": np.nan,
        "pop_theil_decomp_sum": np.nan,
        "pop_theil_decomp_residual": np.nan,
        "pop_theil_within_city_share_pct": np.nan,
        "pop_theil_between_city_share_pct": np.nan,
        "pop_theil_city_count": 0,
    }
    if city_col not in df.columns or value_col not in df.columns or weight_col not in df.columns:
        return empty

    x = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=np.float64)
    w = pd.to_numeric(df[weight_col], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(w) & (w > 0) & (x >= 0)
    if not np.any(valid):
        return empty

    x = x[valid]
    w = w[valid]
    city = df.loc[valid, city_col].astype("string").fillna("<MISSING_CITY>").to_numpy()
    total_pop = float(w.sum())
    total_access = float(np.sum(w * x))
    if total_pop <= 0:
        return empty
    if total_access <= 0:
        return {
            **empty,
            "pop_theil_within_city": 0.0,
            "pop_theil_between_city": 0.0,
            "pop_theil_decomp_sum": 0.0,
            "pop_theil_decomp_residual": 0.0,
            "pop_theil_city_count": int(pd.Series(city).nunique(dropna=False)),
        }

    overall_mean = total_access / total_pop
    # Keep all zero-accessibility observations in the population universe.
    # log(0) is never evaluated; the corresponding x*log(x) limit is exactly 0.
    log_x = np.zeros_like(x)
    np.log(x, out=log_x, where=x != 0)
    wx_log_x_sum = float(np.sum((w * x) * log_x))
    total_theil = wx_log_x_sum / total_access - np.log(overall_mean)

    codes, city_labels = pd.factorize(city, sort=False)
    city_pop = np.bincount(codes, weights=w).astype(np.float64, copy=False)
    city_access = np.bincount(codes, weights=w * x).astype(np.float64, copy=False)
    city_mean = np.divide(city_access, city_pop, out=np.zeros_like(city_access), where=city_pop != 0)
    access_share = city_access / total_access

    log_city_mean = np.zeros_like(city_mean)
    np.log(city_mean, out=log_city_mean, where=city_mean != 0)
    weighted_log_city_mean = float(np.sum(access_share * log_city_mean))
    within = wx_log_x_sum / total_access - weighted_log_city_mean
    between = weighted_log_city_mean - np.log(overall_mean)

    tol = 1e-12
    if abs(within) < tol: within = 0.0
    if abs(between) < tol: between = 0.0
    decomp_sum = within + between
    residual = total_theil - decomp_sum
    if abs(residual) < tol: residual = 0.0
    within_share = within / total_theil * 100.0 if total_theil > tol else np.nan
    between_share = between / total_theil * 100.0 if total_theil > tol else np.nan
    return {
        "pop_theil_within_city": float(within),
        "pop_theil_between_city": float(between),
        "pop_theil_decomp_sum": float(decomp_sum),
        "pop_theil_decomp_residual": float(residual),
        "pop_theil_within_city_share_pct": float(within_share) if np.isfinite(within_share) else np.nan,
        "pop_theil_between_city_share_pct": float(between_share) if np.isfinite(between_share) else np.nan,
        "pop_theil_city_count": int(len(city_labels)),
    }

def calculate_accessibility_stats(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {}

    pop_all = pd.to_numeric(df["pop"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    acc_all = pd.to_numeric(df["acc"], errors="coerce").to_numpy(dtype=np.float64)

    valid = np.isfinite(acc_all) & np.isfinite(pop_all) & (pop_all > 0)
    a = acc_all[valid]
    w = pop_all[valid]

    total_grids = int(len(df))
    pop_nonnegative = np.where(np.isfinite(pop_all) & (pop_all >= 0), pop_all, 0)
    total_pop = float(pop_nonnegative.sum())
    snap_invalid_pop_pct = _snap_invalid_pop_pct(df, pop_nonnegative)

    stats = {
        "min": float(np.min(a)) if len(a) else np.nan,
        "max": float(np.max(a)) if len(a) else np.nan,
        "grid_num": total_grids,
        "pop_num": total_pop,
        "snap_invalid_pop_pct": snap_invalid_pop_pct,
        "valid_acc_grid_num": int(valid.sum()),
        "valid_acc_pop_num": float(w.sum()),
        "valid_acc_pop_pct": float(w.sum() / total_pop) if total_pop > 0 else np.nan,
        "inequality_population_universe": "finite_acc_ge_0_and_positive_population_for_all_formal_inequality_metrics",
    }
    stats.update(_distribution_stats_single_sort(a, w))
    stats.update(theil_city_decomposition(df))

    # On 2_2 matched data city_name_norm is available, so the decomposition total
    # should reproduce the formal Theil up to floating-point noise.
    if np.isfinite(stats.get("pop_theil", np.nan)) and np.isfinite(stats.get("pop_theil_decomp_sum", np.nan)):
        stats["pop_theil_decomp_residual"] = float(
            stats["pop_theil"] - stats["pop_theil_decomp_sum"]
        )
        if abs(stats["pop_theil_decomp_residual"]) < 1e-12:
            stats["pop_theil_decomp_residual"] = 0.0
    return stats


def read_matched_year(
    year: int,
    scope: str,
    profile: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    parts = discover_parts(matched_parts_dir(year, scope, profile))
    if not parts:
        raise FileNotFoundError(
            f"未找到匹配结果：{matched_parts_dir(year, scope, profile)}\n"
            f"请先运行 2_1_multiscale_match.py"
        )

    frames = []
    for p in parts:
        if columns is None:
            frames.append(pd.read_parquet(p))
            continue

        available = parquet_columns(p)
        missing = [c for c in columns if c not in available]
        if missing:
            raise KeyError(f"{p.name} 缺少字段 {missing}")
        frames.append(pd.read_parquet(p, columns=columns))

    return pd.concat(frames, ignore_index=True)


def stats_output_dir(scope: str, profile: str, kind: str) -> Path:
    from config import ANALYSIS_ROOT
    out = ANALYSIS_ROOT / scope / profile / kind
    out.mkdir(parents=True, exist_ok=True)
    return out
