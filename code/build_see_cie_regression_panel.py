# -*- coding: utf-8 -*-
"""Build the analysis-ready city-level SEE/CIE regression panel from upstream expansion and covariate data."""


from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    BASE_YEAR, END_YEAR, SOCIO_END_YEAR, CHANGE_YEARS,
    SEE_CIE_ANNUAL_ROOT, SEE_CIE_PANEL_ROOT, CITY_DYNAMICS_ROOT,
    CITY_DATABASE_XLSX, CITY_DATABASE_SHEET, CITY_LEVEL_MERGE_MAP,
)
from utils.extended_analysis import (
    read_csv_robust,
    read_stats,
    find_excel_sheet_with_columns,
    ensure_exists,
)

ACC_BASE_COL = f"acc_{BASE_YEAR}"
GINI_BASE_COL = f"gini_{BASE_YEAR}"
THEIL_BASE_COL = f"theil_{BASE_YEAR}"
ATKINSON_BASE_COL = f"atkinson_05_{BASE_YEAR}"
GDP_BASE_COL = f"GDP_{BASE_YEAR}"
RESPOP_BASE_COL = f"ResPop_{BASE_YEAR}"

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
FISCAL_REVENUE_RAW_COL = "地方财政一般预算内收入(万元)"
FISCAL_BASE_COL = f"FiscalRevenue_{BASE_YEAR}"
FISCAL_PC_BASE_COL = f"FiscalRevenue_pc_{BASE_YEAR}"
LN_FISCAL_PC_BASE_COL = f"ln_FiscalRevenue_pc_{BASE_YEAR}"


def aggregate_expansion() -> pd.DataFrame:
    frames = []
    keep = [
        "省级", "地级", "city_level",
        "new_hosp_beds", "expanded_beds", "decreased_beds", "closed_beds",
    ]
    for year in CHANGE_YEARS:
        df = read_csv_robust(SEE_CIE_ANNUAL_ROOT / f"city_SEE_CIE_{year}.csv")
        df["city_level"] = df["city_level"].replace(CITY_LEVEL_MERGE_MAP)
        df["year"] = year
        frames.append(df[[c for c in keep + ["year"] if c in df.columns]])

    all_df = pd.concat(frames, ignore_index=True)


    # SEE = new - closed；CIE = increase - decrease。
    out = all_df.groupby(["省级", "地级", "city_level"], as_index=False).agg(
        new_hosp_beds=("new_hosp_beds", "sum"),
        expanded_beds=("expanded_beds", "sum"),
        decreased_beds=("decreased_beds", "sum"),
        closed_beds=("closed_beds", "sum"),
    )
    component_cols = ["new_hosp_beds", "expanded_beds", "decreased_beds", "closed_beds"]
    out[component_cols] = out[component_cols].fillna(0)
    out["net_SEE_beds"] = out["new_hosp_beds"] - out["closed_beds"]
    out["net_CIE_beds"] = out["expanded_beds"] - out["decreased_beds"]
    out["net_total_beds"] = out["net_SEE_beds"] + out["net_CIE_beds"]

    # Accounting closure QC。
    expected_total = (
        out["new_hosp_beds"] + out["expanded_beds"]
        - out["decreased_beds"] - out["closed_beds"]
    )
    if not np.allclose(
        pd.to_numeric(out["net_total_beds"], errors="coerce").fillna(0).to_numpy(dtype=float),
        pd.to_numeric(expected_total, errors="coerce").fillna(0).to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-9,
    ):
        raise AssertionError(
            "SEE/CIE accounting closure failed: net SEE + net CIE != total net bed change"
        )


    pop = read_stats("accessibility", "city")
    pop = (
        pop[pop["Year"].between(BASE_YEAR, END_YEAR)]
        .groupby("地级", as_index=False)["pop_num"]
        .mean()
        .rename(columns={"pop_num": "city_pop"})
    )
    out = out.merge(pop, on="地级", how="left")
    out["city_SEE"] = out["net_SEE_beds"] / out["city_pop"] * 10000
    out["city_CIE"] = out["net_CIE_beds"] / out["city_pop"] * 10000
    out["city_TotalNetExpansion"] = out["city_SEE"] + out["city_CIE"]
    out["Dominance"] = out["city_SEE"].fillna(0) - out["city_CIE"].fillna(0)
    return out


def _load_baseline_fiscal_revenue() -> pd.DataFrame:
    """Helper for _load_baseline_fiscal_revenue."""


    ensure_exists(CITY_DATABASE_XLSX, "城市数据库（财政收入控制变量）")
    fiscal_sheet = find_excel_sheet_with_columns(
        CITY_DATABASE_XLSX,
        {"年份", "城市", FISCAL_REVENUE_RAW_COL},
        CITY_DATABASE_SHEET,
    )
    fiscal = pd.read_excel(
        CITY_DATABASE_XLSX,
        sheet_name=fiscal_sheet,
        usecols=lambda c: c in {"年份", "城市", FISCAL_REVENUE_RAW_COL},
    )
    fiscal["年份"] = pd.to_numeric(fiscal["年份"], errors="coerce")
    fiscal[FISCAL_REVENUE_RAW_COL] = pd.to_numeric(
        fiscal[FISCAL_REVENUE_RAW_COL], errors="coerce"
    )
    fiscal["城市"] = fiscal["城市"].astype("string").str.strip()
    fiscal = fiscal[fiscal["年份"].eq(BASE_YEAR)].copy()

    duplicate_counts = fiscal.groupby("城市", dropna=False).size()
    duplicate_counts = duplicate_counts[duplicate_counts > 1]
    if len(duplicate_counts):
        print(
            f"Warning: {FISCAL_REVENUE_RAW_COL} 在 {BASE_YEAR} 年有 "
            f"{len(duplicate_counts)} 个城市出现重复记录；按城市取数值均值。"
        )

    fiscal = (
        fiscal.groupby("城市", as_index=False, dropna=False)[FISCAL_REVENUE_RAW_COL]
        .mean()
        .rename(columns={FISCAL_REVENUE_RAW_COL: FISCAL_BASE_COL})
    )
    return fiscal


def add_controls(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()


    changes = read_csv_robust(CITY_DYNAMICS_ROOT / "pop_acc_changes.csv")
    ctrl = [
        c for c in [
            "城市",
            GDP_BASE_COL,
            "GDP_growth",
            "GDP_growth_pct",
            "Ppo_NetIn",
            "Ppo_NetIn_rate",
            RESPOP_BASE_COL,
            "ResPop_growth",
            "ResPop_growth_rate",
        ]
        if c in changes.columns
    ]
    out = (
        out.merge(
            changes[ctrl].drop_duplicates("城市"),
            left_on="地级",
            right_on="城市",
            how="left",
        )
        .drop(columns=["城市"], errors="ignore")
    )


    ensure_exists(CITY_DATABASE_XLSX, "城市数据库（人口密度控制变量）")
    density_sheet = find_excel_sheet_with_columns(
        CITY_DATABASE_XLSX,
        {"年份", "城市", "人口密度(人／平方公里)"},
        CITY_DATABASE_SHEET,
    )
    density = pd.read_excel(
        CITY_DATABASE_XLSX,
        sheet_name=density_sheet,
        usecols=lambda c: c in {"年份", "城市", "人口密度(人／平方公里)"},
    )
    density["年份"] = pd.to_numeric(density["年份"], errors="coerce")
    density["人口密度(人／平方公里)"] = pd.to_numeric(
        density["人口密度(人／平方公里)"], errors="coerce"
    )
    density["城市"] = density["城市"].astype("string").str.strip()
    density = (
        density[density["年份"].between(BASE_YEAR, END_YEAR)]
        .groupby("城市", as_index=False)
        .agg(pop_density_mean=("人口密度(人／平方公里)", "mean"))
    )
    out = (
        out.merge(density, left_on="地级", right_on="城市", how="left")
        .drop(columns=["城市"], errors="ignore")
    )

    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    fiscal = _load_baseline_fiscal_revenue()
    out = (
        out.merge(fiscal, left_on="地级", right_on="城市", how="left")
        .drop(columns=["城市"], errors="ignore")
    )


    fiscal_num = pd.to_numeric(out[FISCAL_BASE_COL], errors="coerce")
    res_pop = pd.to_numeric(out.get(RESPOP_BASE_COL), errors="coerce")
    valid_pc = np.isfinite(fiscal_num) & np.isfinite(res_pop) & (res_pop > 0)
    out[FISCAL_PC_BASE_COL] = np.where(valid_pc, fiscal_num / res_pop, np.nan)

    fiscal_pc = pd.to_numeric(out[FISCAL_PC_BASE_COL], errors="coerce")
    out[LN_FISCAL_PC_BASE_COL] = np.where(
        np.isfinite(fiscal_pc) & (fiscal_pc > 0),
        np.log(fiscal_pc),
        np.nan,
    )
    return out


def add_baseline_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    acc = read_stats("accessibility", "city")
    metric_map = {
        "pop_median": "acc",
        "pop_gini": "gini",
        "pop_theil": "theil",
        "pop_atkinson_05": "atkinson_05",
        "zero_access_pop_pct": "zero_access_pop_pct",
        "p90_p10": "p90_p10",
        "p80_p20": "p80_p20",
    }
    available = [c for c in metric_map if c in acc.columns]
    base = (
        acc[acc["Year"].eq(BASE_YEAR)][["地级", *available]]
        .drop_duplicates("地级")
        .copy()
    )
    end = (
        acc[acc["Year"].eq(END_YEAR)][["地级", *available]]
        .drop_duplicates("地级")
        .copy()
    )
    base = base.rename(columns={c: f"{metric_map[c]}_{BASE_YEAR}" for c in available})
    end = end.rename(columns={c: f"{metric_map[c]}_{END_YEAR}" for c in available})
    out = df.merge(base, on="地级", how="left").merge(end, on="地级", how="left")

    delta_names = {
        "acc": "acc_delta",
        "gini": "gini_delta",
        "theil": "theil_delta",
        "atkinson_05": "atkinson_05_delta",
        "zero_access_pop_pct": "zero_access_pop_pct_delta",
        "p90_p10": "p90_p10_delta",
        "p80_p20": "p80_p20_delta",
    }
    for stem, dst in delta_names.items():
        c0, c1 = f"{stem}_{BASE_YEAR}", f"{stem}_{END_YEAR}"
        if c0 in out.columns and c1 in out.columns:
            out[dst] = (
                pd.to_numeric(out[c1], errors="coerce")
                - pd.to_numeric(out[c0], errors="coerce")
            )
    return out


def main():
    SEE_CIE_PANEL_ROOT.mkdir(parents=True, exist_ok=True)
    df = add_baseline_outcomes(add_controls(aggregate_expansion()))

    out = SEE_CIE_PANEL_ROOT / f"city_{BASE_YEAR}_{END_YEAR}_index.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    miss = pd.DataFrame(
        {
            "column": df.columns,
            "missing_n": [int(df[c].isna().sum()) for c in df.columns],
            "missing_pct": [float(df[c].isna().mean()) for c in df.columns],
        }
    )
    miss.to_csv(
        SEE_CIE_PANEL_ROOT / "panel_missingness.csv",
        index=False,
        encoding="utf-8-sig",
    )


    fiscal_qc_cols = [
        c for c in [
            "省级",
            "地级",
            RESPOP_BASE_COL,
            FISCAL_BASE_COL,
            FISCAL_PC_BASE_COL,
            LN_FISCAL_PC_BASE_COL,
        ]
        if c in df.columns
    ]
    df[fiscal_qc_cols].to_csv(
        SEE_CIE_PANEL_ROOT / "fiscal_control_qc.csv",
        index=False,
        encoding="utf-8-sig",
    )

    n_fiscal = int(pd.to_numeric(df[FISCAL_BASE_COL], errors="coerce").notna().sum())
    n_fiscal_pc = int(pd.to_numeric(df[LN_FISCAL_PC_BASE_COL], errors="coerce").notna().sum())
    print(f"SEE/CIE regression panel: {len(df):,} cities -> {out}")
    print(
        f"Fiscal control | {FISCAL_BASE_COL} available={n_fiscal:,}/{len(df):,}; "
        f"{LN_FISCAL_PC_BASE_COL} available={n_fiscal_pc:,}/{len(df):,}"
    )


if __name__ == "__main__":
    main()
