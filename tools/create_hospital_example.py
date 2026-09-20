#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a deterministic hospital-record example from a local annual hospital CSV.

This utility is intentionally separate from the reviewer-facing reproduction workflow.
It never modifies the source file.  It samples rows only after basic structural checks
and writes the public example using the canonical column order documented in
``data/example/hospital_records_2024_schema.csv``.

Example
-------
python tools/create_hospital_example.py \
    --input "C:/path/to/private/beds/2024.csv" \
    --output "data/example/hospital_records_2024_sample_local.csv" \
    --n 100 --seed 20260920
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_COLUMNS = [
    "name",
    "beds",
    "lng",
    "lat",
    "province",
    "region",
    "area",
    "grade",
    "type",
    "construction_time",
    "3A_year",
]

# These patterns are used only to keep obviously out-of-scope records out of an
# illustrative public sample.  They are not a substitute for the study's documented
# record-level eligibility review and are not used by the analytical pipeline.
OBVIOUS_OUT_OF_SCOPE_PATTERNS = [
    re.compile(r"中国人民解放军|解放军|人民武装警察|武警|联勤保障部队"),
    re.compile(r"妇幼保健院|妇幼保健中心|妇幼健康服务中心"),
]


def read_csv_robust(path: Path) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeError(
        f"Could not decode {path} with utf-8-sig, utf-8, or gb18030. "
        + " | ".join(errors)
    )


def obvious_out_of_scope_mask(names: pd.Series) -> pd.Series:
    text = names.fillna("").astype(str)
    mask = pd.Series(False, index=text.index)
    for pattern in OBVIOUS_OUT_OF_SCOPE_PATTERNS:
        mask |= text.str.contains(pattern, regex=True, na=False)
    return mask


def prepare_sample(
    df: pd.DataFrame,
    *,
    n: int,
    seed: int,
    keep_obvious_out_of_scope: bool,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if n < 1:
        raise ValueError("--n must be >= 1")

    required = {"name", "beds", "lng", "lat"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input hospital table is missing required columns: {missing}")

    work = df.copy()
    start_rows = len(work)

    for col in ("beds", "lng", "lat"):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    valid = (
        work["name"].notna()
        & work["name"].astype(str).str.strip().ne("")
        & np.isfinite(work["beds"])
        & np.isfinite(work["lng"])
        & np.isfinite(work["lat"])
        & work["beds"].ge(0)
        & work["lng"].between(-180, 180)
        & work["lat"].between(-90, 90)
    )
    invalid_rows = int((~valid).sum())
    work = work.loc[valid].copy()

    scope_mask = obvious_out_of_scope_mask(work["name"])
    obvious_scope_rows = int(scope_mask.sum())
    if not keep_obvious_out_of_scope:
        work = work.loc[~scope_mask].copy()

    if work.empty:
        raise ValueError("No eligible rows remain after basic example-generation checks")

    # Always emit the canonical example schema.  Optional columns absent from the
    # source are added as empty values so the public example remains structurally
    # consistent across local source-table versions.
    for col in SCHEMA_COLUMNS:
        if col not in work.columns:
            work[col] = pd.NA

    sample_n = min(n, len(work))
    sample = work.sample(n=sample_n, random_state=seed, replace=False).copy()
    sample = sample[SCHEMA_COLUMNS].reset_index(drop=True)

    stats = {
        "source_rows": int(start_rows),
        "invalid_required_rows": invalid_rows,
        "obvious_out_of_scope_rows_detected": obvious_scope_rows,
        "candidate_rows_after_filters": int(len(work)),
        "sample_rows": int(len(sample)),
    }
    return sample, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Local annual hospital CSV, e.g. the private 2024 table.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hospital_records_2024_sample_local.csv"),
        help="Output CSV. A *_local.csv name is recommended until the sample has been manually reviewed.",
    )
    parser.add_argument("--n", type=int, default=100, help="Maximum number of records to sample (default: 100).")
    parser.add_argument("--seed", type=int, default=20260920, help="Deterministic pandas random_state (default: 20260920).")
    parser.add_argument(
        "--keep-obvious-out-of-scope",
        action="store_true",
        help=(
            "Do not remove records whose names clearly indicate military/armed-police or maternal-and-child-health institutions. "
            "By default these obvious cases are omitted from the illustrative sample only."
        ),
    )
    args = parser.parse_args()

    src = args.input.expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    out = args.output.expanduser().resolve()
    if src == out:
        raise ValueError("Refusing to overwrite the source hospital table")

    df = read_csv_robust(src)
    sample, stats = prepare_sample(
        df,
        n=args.n,
        seed=args.seed,
        keep_obvious_out_of_scope=args.keep_obvious_out_of_scope,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False, encoding="utf-8-sig")

    print("Hospital example generated for manual review")
    print(f"  source: {src}")
    print(f"  output: {out}")
    print(f"  seed: {args.seed}")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    if stats["obvious_out_of_scope_rows_detected"] and args.keep_obvious_out_of_scope:
        print("  WARNING: obvious out-of-scope names were retained because --keep-obvious-out-of-scope was set.")
    print("Review the sampled rows manually before replacing the public example file.")


if __name__ == "__main__":
    main()
