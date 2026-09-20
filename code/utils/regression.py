"""Shared data preparation for SEE/CIE regression and robustness analyses."""
from __future__ import annotations

import pandas as pd

from config import BASE_YEAR, END_YEAR, SEE_CIE_PANEL_ROOT, CITY_ORDER_4
from utils.extended_analysis import read_csv_robust, zscore_inplace

ACC_BASE_COL = f"acc_{BASE_YEAR}"
GINI_BASE_COL = f"gini_{BASE_YEAR}"
THEIL_BASE_COL = f"theil_{BASE_YEAR}"
ATKINSON_BASE_COL = f"atkinson_05_{BASE_YEAR}"
GDP_BASE_COL = f"GDP_{BASE_YEAR}"
RESPOP_BASE_COL = f"ResPop_{BASE_YEAR}"
FISCAL_BASE_COL = f"ln_FiscalRevenue_pc_{BASE_YEAR}"


def load_standardized_see_cie_panel() -> pd.DataFrame:
    """Load the released city panel and apply the standardization used by all models."""
    path = SEE_CIE_PANEL_ROOT / f"city_{BASE_YEAR}_{END_YEAR}_index.csv"
    df = read_csv_robust(path)
    if FISCAL_BASE_COL not in df.columns:
        raise KeyError(f"Missing required fiscal-capacity column: {FISCAL_BASE_COL}")

    df = df[df["city_level"].isin(CITY_ORDER_4)].copy()
    df["city_level_4"] = pd.Categorical(df["city_level"], categories=CITY_ORDER_4, ordered=True)
    for col in ["city_SEE", "city_CIE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["city_TotalExpansion"] = df["city_SEE"] + df["city_CIE"]

    continuous = [
        "city_SEE", "city_CIE", "city_TotalExpansion", ACC_BASE_COL,
        GINI_BASE_COL, THEIL_BASE_COL, ATKINSON_BASE_COL, GDP_BASE_COL,
        "GDP_growth", "GDP_growth_pct", "Ppo_NetIn", "Ppo_NetIn_rate",
        RESPOP_BASE_COL, "ResPop_growth", "ResPop_growth_rate",
        "pop_density_mean", FISCAL_BASE_COL,
    ]
    inequality = [
        "gini_delta", "theil_delta", "atkinson_05_delta",
        GINI_BASE_COL, THEIL_BASE_COL, ATKINSON_BASE_COL,
    ]
    df = zscore_inplace(df, continuous)
    df = zscore_inplace(df, inequality)
    return df
