#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recompute the public three-component Shapley decomposition from eight scenario statistics.

This lightweight script is used by ``reproduce.py``.  It deliberately starts from
``scenario_stats.csv`` rather than the released ``shapley_summary.csv`` so that the
public workflow independently recalculates the decomposition before plotting it.
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

FACTORS = ("road", "population", "hospital")
FACTOR_BITS = {"road": 0, "population": 1, "hospital": 2}
SCENARIOS = tuple(f"A{r}{p}{h}" for r in (0, 1) for p in (0, 1) for h in (0, 1))
BASE_CODE = "A000"
TARGET_CODE = "A111"

OUTCOMES = {
    "accessibility": "pop_median",
    "gini": "pop_gini",
    "theil": "pop_theil",
    "atkinson_05": "pop_atkinson_05",
}

FACTOR_DISPLAY = {
    "road": "Road",
    "population": "Population",
    "hospital": "Hospital supply",
}


def state_code(active: set[str]) -> str:
    """Return the Axyz scenario code for a set of updated factors."""
    bits = ["0", "0", "0"]
    for factor in active:
        if factor not in FACTOR_BITS:
            raise KeyError(f"Unknown Shapley factor: {factor}")
        bits[FACTOR_BITS[factor]] = "1"
    return "A" + "".join(bits)


def exact_shapley(values: dict[str, float]) -> dict[str, float]:
    """Exact three-player Shapley values averaged over all 3! update orders."""
    missing = sorted(set(SCENARIOS) - set(values))
    if missing:
        raise ValueError(f"Missing Shapley scenarios: {missing}")

    phi = {factor: 0.0 for factor in FACTORS}
    permutations = list(itertools.permutations(FACTORS))
    for order in permutations:
        active: set[str] = set()
        current = float(values[state_code(active)])
        for factor in order:
            active.add(factor)
            updated = float(values[state_code(active)])
            phi[factor] += updated - current
            current = updated
    return {factor: value / len(permutations) for factor, value in phi.items()}


def _single_unique(df: pd.DataFrame, column: str, default):
    if column not in df.columns:
        return default
    vals = df[column].dropna().unique().tolist()
    if not vals:
        return default
    if len(vals) != 1:
        raise ValueError(f"Expected one unique {column}, found: {vals}")
    return vals[0]


def validate_scenario_stats(stats: pd.DataFrame) -> None:
    """Validate the compact public scenario table before decomposition."""
    required = {"scenario", *OUTCOMES.values()}
    missing_cols = sorted(required - set(stats.columns))
    if missing_cols:
        raise ValueError(f"scenario_stats.csv is missing required columns: {missing_cols}")

    codes = stats["scenario"].astype(str)
    if codes.duplicated().any():
        dup = sorted(codes[codes.duplicated(keep=False)].unique().tolist())
        raise ValueError(f"Duplicate Shapley scenario rows: {dup}")

    actual = set(codes)
    expected = set(SCENARIOS)
    if actual != expected:
        raise ValueError(
            "Shapley scenario set must be exactly A000-A111; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )

    for col in OUTCOMES.values():
        x = pd.to_numeric(stats[col], errors="coerce")
        if not np.isfinite(x).all():
            bad = stats.loc[~np.isfinite(x), ["scenario", col]].to_dict("records")
            raise ValueError(f"Non-finite values in {col}: {bad}")


def decompose_scenario_stats(stats: pd.DataFrame) -> pd.DataFrame:
    """Return a manuscript-compatible exact Shapley summary from scenario statistics."""
    validate_scenario_stats(stats)
    indexed = stats.set_index(stats["scenario"].astype(str))

    base_year = int(indexed.loc[BASE_CODE, "road_year"]) if "road_year" in indexed.columns else 2014
    target_year = int(indexed.loc[TARGET_CODE, "road_year"]) if "road_year" in indexed.columns else 2024
    profile = str(_single_unique(stats, "profile", ""))
    service_scope = str(_single_unique(stats, "service_scope", ""))

    rows: list[dict[str, object]] = []
    for outcome, column in OUTCOMES.items():
        values = {code: float(indexed.loc[code, column]) for code in SCENARIOS}
        contributions = exact_shapley(values)
        total_change = values[TARGET_CODE] - values[BASE_CODE]
        contribution_sum = sum(contributions.values())
        residual = contribution_sum - total_change
        if not np.isclose(residual, 0.0, rtol=1e-12, atol=1e-12):
            raise RuntimeError(
                f"Shapley efficiency check failed for {outcome}: "
                f"sum(phi)-total={residual:.16g}"
            )

        improvement_sign = 1.0 if outcome == "accessibility" else -1.0
        for factor in FACTORS:
            share = (
                100.0 * contributions[factor] / total_change
                if abs(total_change) > 1e-15
                else np.nan
            )
            rows.append(
                {
                    "outcome": outcome,
                    "factor": factor,
                    "factor_display": FACTOR_DISPLAY[factor],
                    "contribution_abs": contributions[factor],
                    "contribution_to_improvement": contributions[factor] * improvement_sign,
                    "share_pct": share,
                    "baseline_value": values[BASE_CODE],
                    "target_value": values[TARGET_CODE],
                    "total_change": total_change,
                    "total_improvement": total_change * improvement_sign,
                    "sum_check_residual": residual,
                    "method": f"exact_shapley_{math.factorial(len(FACTORS))}_permutation_average",
                    "shapley_mode": "three_component",
                    "base_year": base_year,
                    "target_year": target_year,
                    "profile": profile,
                    "service_scope": service_scope,
                }
            )
    return pd.DataFrame(rows)


def compare_with_reference(
    recomputed: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-10,
) -> None:
    """Verify that recomputed values agree with the released decomposition summary."""
    key = ["outcome", "factor"]
    numeric = [
        "contribution_abs",
        "share_pct",
        "baseline_value",
        "target_value",
        "total_change",
    ]
    left = recomputed[key + numeric].copy()
    right = reference[key + numeric].copy()
    merged = left.merge(right, on=key, how="outer", suffixes=("_new", "_ref"), indicator=True)
    if not merged["_merge"].eq("both").all():
        bad = merged.loc[~merged["_merge"].eq("both"), key + ["_merge"]]
        raise ValueError(f"Released Shapley summary row mismatch:\n{bad.to_string(index=False)}")

    failures: list[str] = []
    for col in numeric:
        a = pd.to_numeric(merged[f"{col}_new"], errors="coerce").to_numpy(float)
        b = pd.to_numeric(merged[f"{col}_ref"], errors="coerce").to_numpy(float)
        ok = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
        if not ok.all():
            for i in np.where(~ok)[0]:
                row = merged.iloc[int(i)]
                failures.append(
                    f"{row['outcome']}/{row['factor']} {col}: "
                    f"recomputed={a[i]:.12g}, reference={b[i]:.12g}"
                )
    if failures:
        raise ValueError("Recomputed Shapley values differ from the released summary:\n  - " + "\n  - ".join(failures))


def main() -> None:
    # Import the shared configuration only for filesystem resolution.  The numerical
    # functions above remain importable for unit tests without requiring raw data.
    from config import RESULT_ROOT

    scenario_path = RESULT_ROOT / "4_2_shapley_decomposition" / "scenario_stats.csv"
    reference_path = RESULT_ROOT / "4_2_shapley_decomposition" / "shapley_summary.csv"
    if not scenario_path.exists():
        raise FileNotFoundError(f"Missing public Shapley scenario table: {scenario_path}")

    stats = pd.read_csv(scenario_path, encoding="utf-8-sig")
    recomputed = decompose_scenario_stats(stats)

    if reference_path.exists():
        reference = pd.read_csv(reference_path, encoding="utf-8-sig")
        compare_with_reference(recomputed, reference)

    generated_dir_path = RESULT_ROOT / "4_2_shapley_decomposition" / "shapley_summary_recomputed.csv"
    generated_root_path = RESULT_ROOT / "shapley_summary.csv"
    recomputed.to_csv(generated_dir_path, index=False, encoding="utf-8-sig")
    recomputed.to_csv(generated_root_path, index=False, encoding="utf-8-sig")



if __name__ == "__main__":
    main()
