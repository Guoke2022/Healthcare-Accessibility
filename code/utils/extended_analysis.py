# -*- coding: utf-8 -*-
"""Module utilities for extended analysis."""
from __future__ import annotations

from pathlib import Path
import math
import re
import numpy as np
import pandas as pd

from config import (
    SERVICE_SCOPE, PROFILE, CITY_LEVEL_MERGE_MAP, CITY_ORDER_4,
)
from utils.multiscale import stats_output_dir


def ensure_exists(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label}不存在：{path}")
    return path


def norm6(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return pd.NA
    s = re.sub(r"\.0$", "", s)
    digits = re.sub(r"\D", "", s)
    return digits.zfill(6) if digits else pd.NA


def read_csv_robust(path: Path, **kwargs) -> pd.DataFrame:
    last = None
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as e:
            last = e
    raise RuntimeError(f"无法识别 CSV 编码：{path}") from last


def find_excel_sheet_with_columns(path: Path, required: set[str], preferred: str | None = None) -> str:
    ensure_exists(path, "Excel")
    xl = pd.ExcelFile(path)
    order = ([preferred] if preferred and preferred in xl.sheet_names else []) + [s for s in xl.sheet_names if s != preferred]
    for sheet in order:
        cols = set(pd.read_excel(path, sheet_name=sheet, nrows=0).columns.astype(str))
        if required.issubset(cols):
            return sheet
    raise KeyError(f"{path} 中没有同时包含这些字段的工作表：{sorted(required)}；sheets={xl.sheet_names}")


def stats_path(kind: str, group: str) -> Path:
    if kind not in {"accessibility", "travel_time"}:
        raise ValueError(kind)
    suffix = "acc" if kind == "accessibility" else "travel_time"
    return stats_output_dir(SERVICE_SCOPE, PROFILE, kind) / f"{group}_{suffix}_stats.csv"


def read_stats(kind: str, group: str) -> pd.DataFrame:
    path = stats_path(kind, group)
    ensure_exists(path, f"2_2 {group} {kind}统计")
    df = read_csv_robust(path)
    for c in ["县级码", "地级码", "省级码"]:
        if c in df.columns:
            df[c] = df[c].map(norm6)
    return df


def compute_deltas(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    out = df.copy().sort_values([id_col, "Year"])
    pairs = {
        "pop_median": "acc_delta",
        "pop_gini": "gini_delta",
        "pop_theil": "theil_delta",
        "pop_atkinson_05": "atkinson_05_delta",
        "zero_access_pop_pct": "zero_access_pop_pct_delta",
        "p90_p10": "p90_p10_delta",
        "p80_p20": "p80_p20_delta",
    }
    for src, dst in pairs.items():
        if src in out.columns:
            out[dst] = out.groupby(id_col, dropna=False)[src].diff()
    return out


def merge_city_levels(series: pd.Series) -> pd.Series:
    return series.replace(CITY_LEVEL_MERGE_MAP)


def safe_ratio(num, den):
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    return np.where(np.isfinite(den) & (den != 0), num / den, np.nan)


def zscore_inplace(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in [x for x in cols if x in out.columns]:
        s = pd.to_numeric(out[c], errors="coerce")
        sd = s.std(skipna=True)
        out[c] = (s - s.mean(skipna=True)) / sd if pd.notna(sd) and sd != 0 else s
    return out


# Publication-wide regression significance convention.
# Keep this as the single source of truth for regression tables/figures.
SIG_P_ONE_STAR = 0.10
SIG_P_TWO_STARS = 0.05
SIG_P_THREE_STARS = 0.01
SIGNIFICANCE_NOTE = "* P < 0.10; ** P < 0.05; *** P < 0.01."


def p_to_star(p) -> str:
    """Map an exact p-value to the manuscript-wide significance stars."""
    if p is None or not np.isfinite(float(p)):
        return ""
    p = float(p)
    if p < SIG_P_THREE_STARS:
        return "***"
    if p < SIG_P_TWO_STARS:
        return "**"
    if p < SIG_P_ONE_STAR:
        return "*"
    return ""


def summary_col_standard_stars(results, **kwargs):
    """statsmodels ``summary_col`` using the manuscript-wide star thresholds.

    statsmodels' built-in ``stars=True`` uses *<0.10, **<0.05, ***<0.01.
    This wrapper keeps that same manuscript-wide convention explicit and shared
    with figures and post-estimation tables, while leaving estimation results and
    exact p-values untouched.
    """
    from statsmodels.iolib import summary2 as s2

    original = s2._col_params

    def _col_params_standard(result, float_format='%.4f', stars=True, include_r2=False):
        res = s2.summary_params(result)
        for col in res.columns[:2]:
            res[col] = res[col].apply(lambda x: float_format % x)
        res.iloc[:, 1] = '(' + res.iloc[:, 1] + ')'
        if stars:
            coef_col = res.columns[0]
            pvals = res.iloc[:, 3]
            # Append one star at each nested threshold, matching p_to_star().
            for cutoff in (SIG_P_ONE_STAR, SIG_P_TWO_STARS, SIG_P_THREE_STARS):
                idx = pvals < cutoff
                res.loc[idx, coef_col] = res.loc[idx, coef_col] + '*'
        res = res.iloc[:, :2].stack(**s2.FUTURE_STACK)
        if include_r2:
            r2 = pd.Series({
                ('R-squared', ''): getattr(result, 'rsquared', np.nan),
                ('R-squared Adj.', ''): getattr(result, 'rsquared_adj', np.nan),
            })
            if r2.notnull().any():
                r2 = r2.apply(lambda x: float_format % x)
                res = pd.concat([res, r2], axis=0)
        res = pd.DataFrame(res)
        res.columns = [str(result.model.endog_names)]
        return res

    try:
        s2._col_params = _col_params_standard
        return s2.summary_col(results, **kwargs)
    finally:
        s2._col_params = original


def set_nature_style():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
    })
    return plt


def city_level_from_name(names: pd.Series) -> pd.Series:
    from utils.city_levels import (
        Megacity_list, Supercity_list, Type_I_Large_Cities_list,
        Type_II_Large_Cities_list, Medium_sized_cites_list,
    )
    mapping = {
        **{c: "Mega City" for c in Megacity_list},
        **{c: "Super City" for c in Supercity_list},
        **{c: "Type I Large City" for c in Type_I_Large_Cities_list},
        **{c: "Type II Large City" for c in Type_II_Large_Cities_list},
        **{c: "Medium-sized City" for c in Medium_sized_cites_list},
    }
    out = names.map(mapping)
    out = out.where(out.notna(), np.where(names.notna(), "Small City", pd.NA))
    return pd.Series(out, index=names.index, dtype="string")

# =============================================================================
# Extended-analysis lineage guard
# =============================================================================

EXTENDED_LINEAGE_SCHEMA_VERSION = 1


def _existing_shapefile_sidecars(path: Path) -> list[Path]:
    out = []
    for ext in [path.suffix, ".shx", ".dbf", ".prj", ".cpg", ".qix"]:
        p = path.with_suffix(ext)
        if p.exists() and p not in out:
            out.append(p)
    return out


def extended_lineage_source_files() -> list[Path]:
    """Files whose current content defines whether downstream outputs are reusable.

    This is intentionally broad.  Extended analyses are much cheaper than 1_1,
    so conservative invalidation is preferred over silently combining a new core
    run with stale regression/descriptive outputs.
    """
    from config import (
        ANALYSIS_ROOT, BASE_YEAR, END_YEAR, YEARS, HOSPITAL_DATA_DIR, COUNTY_SHP,
        CITY_DATABASE_XLSX, POP_FLOW_CSV, POP_FLOW_XLSX, GDP_CITY_CSV,
        GDP_CITY_RAW_CSV, CODE_ROOT, SERVICE_SCOPE, PROFILE,
    )

    files: list[Path] = []
    for y in YEARS:
        p = HOSPITAL_DATA_DIR / f"{y}.csv"
        if p.exists():
            files.append(p)

    for p in [CITY_DATABASE_XLSX, POP_FLOW_CSV, POP_FLOW_XLSX, GDP_CITY_CSV, GDP_CITY_RAW_CSV]:
        p = Path(p)
        if p.exists():
            files.append(p)

    files.extend(_existing_shapefile_sidecars(Path(COUNTY_SHP)))

    # Downstream analyses read the 2_2 statistics through read_stats(). Include all current
    # CSVs in this scope/profile so any core-result change invalidates the marker.
    stats_root = Path(ANALYSIS_ROOT) / SERVICE_SCOPE / PROFILE
    if stats_root.exists():
        files.extend(sorted(stats_root.rglob("*.csv")))

    # Code/config changes that affect extended-analysis semantics also invalidate.
    code_root = Path(CODE_ROOT)
    for name in [
        "config.py", "4_1_city_dynamics.py", "prepare_hospital_expansion.py",
        "build_see_cie_regression_panel.py", "run_see_cie_regressions.py",
        "run_province_fe_robustness.py", "run_spatial_robustness.py",
    ]:
        p = code_root / name
        if p.exists():
            files.append(p)
    this_file = Path(__file__).resolve()
    if this_file.exists():
        files.append(this_file)

    # Stable deduplication.
    seen = set(); out = []
    for p in files:
        rp = Path(p).resolve()
        if rp not in seen:
            seen.add(rp); out.append(rp)
    return out


def extended_lineage_fingerprint() -> str:
    from config import BASE_YEAR, END_YEAR, SERVICE_SCOPE, PROFILE, MAX_ANALYSIS_TIME_MIN
    from utils.cache import build_fingerprint
    return build_fingerprint(
        config={
            "stage": "extended_analysis_lineage",
            "schema_version": EXTENDED_LINEAGE_SCHEMA_VERSION,
            "base_year": BASE_YEAR,
            "end_year": END_YEAR,
            "service_scope": SERVICE_SCOPE,
            "profile": PROFILE,
            "search_threshold_min": float(MAX_ANALYSIS_TIME_MIN),
        },
        files=extended_lineage_source_files(),
    )


def extended_lineage_marker_path() -> Path:
    from config import RESULT_ROOT
    return Path(RESULT_ROOT) / "_extended_analysis_lineage.json"


def write_extended_lineage_marker(*, completed_stages: list[str] | None = None, run_id: str | None = None) -> Path:
    from datetime import datetime
    from utils.cache import json_dump
    p = extended_lineage_marker_path()
    json_dump({
        "schema_version": EXTENDED_LINEAGE_SCHEMA_VERSION,
        "lineage_fingerprint": extended_lineage_fingerprint(),
        "completed_stages": completed_stages or [],
        "run_id": run_id,
        "completed_at": datetime.now().isoformat(),
    }, p)
    return p


def extended_lineage_status() -> tuple[bool, str, str | None, str | None]:
    """Return (is_current, reason, marker_fp, current_fp)."""
    from utils.cache import json_load
    p = extended_lineage_marker_path()
    if not p.exists():
        return False, "missing_extended_lineage_marker", None, None
    try:
        marker = json_load(p)
    except Exception:
        return False, "unreadable_extended_lineage_marker", None, None
    marker_fp = marker.get("lineage_fingerprint")
    try:
        current_fp = extended_lineage_fingerprint()
    except Exception as e:
        return False, f"cannot_compute_current_lineage:{type(e).__name__}", marker_fp, None
    if marker_fp != current_fp:
        return False, "extended_lineage_changed", marker_fp, current_fp
    return True, "extended_lineage_current", marker_fp, current_fp
