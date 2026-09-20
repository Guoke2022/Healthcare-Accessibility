# -*- coding: utf-8 -*-
"""4_1 Prepare city socioeconomic controls and accessibility-change covariates.

This is a deterministic data-preparation stage for the SEE/CIE regression pipeline.
Exploratory city clustering is intentionally excluded from the formal workflow.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    ANALYSIS_YEARS, BASE_YEAR, END_YEAR, SOCIO_END_YEAR,
    POP_FLOW_CSV, POP_FLOW_XLSX, POP_FLOW_SHEET,
    GDP_CITY_CSV, GDP_CITY_RAW_CSV, CITY_DATABASE_XLSX, CITY_DATABASE_SHEET,
    CITY_DYNAMICS_ROOT,
)
from utils.extended_analysis import (
    ensure_exists, read_csv_robust, find_excel_sheet_with_columns,
    read_stats, norm6,
)

ACC_BASE_COL = f"acc_{BASE_YEAR}"
TIME_BASE_COL = f"time_{BASE_YEAR}"
RESPOP_BASE_COL = f"ResPop_{BASE_YEAR}"
GDP_BASE_COL = f"GDP_{BASE_YEAR}"


def _normalize_pop_flow(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "year" in x.columns and "年份" not in x.columns:
        x = x.rename(columns={"year": "年份"})
    required = {"年份", "城市", "常住人口数(万人)", "户籍人口数(万人)"}
    missing = required - set(x.columns)
    if missing:
        raise KeyError(f"人口流动表缺少字段：{sorted(missing)}")
    x["年份"] = pd.to_numeric(x["年份"], errors="coerce").astype("Int64")
    for c in ["常住人口数(万人)", "户籍人口数(万人)"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    if "人口净流入(万人)" not in x.columns:
        x["人口净流入(万人)"] = x["常住人口数(万人)"] - x["户籍人口数(万人)"]
    else:
        x["人口净流入(万人)"] = pd.to_numeric(x["人口净流入(万人)"], errors="coerce")
    x = x.sort_values([c for c in ["省份", "城市", "年份"] if c in x.columns])
    if "常住人口增长率" not in x.columns:
        x["常住人口增长率"] = x.groupby("城市")["常住人口数(万人)"].pct_change()
    else:
        x["常住人口增长率"] = pd.to_numeric(x["常住人口增长率"], errors="coerce")
    for c in ["城市代码", "省份代码"]:
        if c in x.columns:
            x[c] = x[c].map(norm6)
    return x[x["年份"].between(BASE_YEAR, SOCIO_END_YEAR)].copy()


def _require_year_coverage(df: pd.DataFrame, year_col: str, target_year: int, label: str) -> bool:
    years = pd.to_numeric(df[year_col], errors="coerce").dropna()
    max_year = int(years.max()) if len(years) else None
    if max_year is not None and max_year >= target_year:
        return True
    print(f"4_1 {label} only reaches {max_year}; need {target_year}, trying a fuller source...")
    return False


def load_population_flow() -> pd.DataFrame:
    if POP_FLOW_CSV.exists():
        x = _normalize_pop_flow(read_csv_robust(POP_FLOW_CSV))
        if _require_year_coverage(x, "年份", SOCIO_END_YEAR, f"population flow CSV {POP_FLOW_CSV.name}"):
            print(f"4_1 population flow: {POP_FLOW_CSV}")
            return x
    ensure_exists(POP_FLOW_XLSX, "人口流动数据未覆盖目标末年；Excel")
    sheet = find_excel_sheet_with_columns(
        POP_FLOW_XLSX,
        {"年份", "城市", "常住人口数(万人)", "户籍人口数(万人)"},
        POP_FLOW_SHEET,
    )
    x = _normalize_pop_flow(pd.read_excel(POP_FLOW_XLSX, sheet_name=sheet))
    if not _require_year_coverage(x, "年份", SOCIO_END_YEAR, f"population flow Excel {POP_FLOW_XLSX.name}"):
        raise RuntimeError(f"人口流动数据没有覆盖 SOCIO_END_YEAR={SOCIO_END_YEAR}")
    print(f"4_1 population flow from Excel: {POP_FLOW_XLSX} | sheet={sheet}")
    return x


def _normalize_gdp(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    rename = {
        "年份": "year",
        "人均地区生产总值(元)": "GDP_per",
        "城市代码": "城市代码",
    }
    x = x.rename(columns={k: v for k, v in rename.items() if k in x.columns})
    required = {"year", "城市", "GDP_per"}
    missing = required - set(x.columns)
    if missing:
        raise KeyError(f"GDP 表缺少字段：{sorted(missing)}")
    x["year"] = pd.to_numeric(x["year"], errors="coerce").astype("Int64")
    x["GDP_per"] = pd.to_numeric(x["GDP_per"], errors="coerce")
    for c in ["城市代码", "地级码", "省份代码"]:
        if c in x.columns:
            x[c] = x[c].map(norm6)
    x = x[x["year"].between(BASE_YEAR, SOCIO_END_YEAR)].copy()
    x = x.sort_values([c for c in ["省份", "城市", "year"] if c in x.columns])
    x["GDP_per_rate"] = x.groupby("城市")["GDP_per"].pct_change()
    return x


def load_city_gdp() -> pd.DataFrame:
    for label, path in [("prepared GDP CSV", GDP_CITY_CSV), ("raw GDP CSV", GDP_CITY_RAW_CSV)]:
        if not path.exists():
            continue
        x = _normalize_gdp(read_csv_robust(path))
        if _require_year_coverage(x, "year", SOCIO_END_YEAR, f"{label} {path.name}"):
            print(f"4_1 GDP: {path}")
            return x
    ensure_exists(CITY_DATABASE_XLSX, "城市 GDP CSV 未覆盖目标末年；城市数据库")
    sheet = find_excel_sheet_with_columns(
        CITY_DATABASE_XLSX,
        {"年份", "城市", "人均地区生产总值(元)"},
        CITY_DATABASE_SHEET,
    )
    x = _normalize_gdp(pd.read_excel(CITY_DATABASE_XLSX, sheet_name=sheet))
    if not _require_year_coverage(x, "year", SOCIO_END_YEAR, f"city database {CITY_DATABASE_XLSX.name}"):
        raise RuntimeError(f"城市 GDP 数据没有覆盖 SOCIO_END_YEAR={SOCIO_END_YEAR}")
    print(f"4_1 GDP from city database: {CITY_DATABASE_XLSX} | sheet={sheet}")
    return x


def build_city_changes(pop: pd.DataFrame, gdp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    acc = read_stats("accessibility", "city")
    tim = read_stats("travel_time", "city")
    acc = acc[["Year", "地级", "pop_median"]].rename(columns={"pop_median": "acc_median"})
    tim = tim[["Year", "地级", "pop_median"]].rename(columns={"pop_median": "time_median"})


    panel = pop.merge(acc, left_on=["年份", "城市"], right_on=["Year", "地级"], how="left").drop(columns=["Year", "地级"])
    panel = panel.merge(tim, left_on=["年份", "城市"], right_on=["Year", "地级"], how="left").drop(columns=["Year", "地级"])

    acc0 = acc[acc["Year"] == BASE_YEAR].set_index("地级")["acc_median"]
    time0 = tim[tim["Year"] == BASE_YEAR].set_index("地级")["time_median"]

    panel[ACC_BASE_COL] = panel["城市"].map(acc0)
    panel[TIME_BASE_COL] = panel["城市"].map(time0)
    acc1 = acc[acc["Year"] == END_YEAR].set_index("地级")["acc_median"]
    time1 = tim[tim["Year"] == END_YEAR].set_index("地级")["time_median"]


    meta_cols = [c for c in ["省份", "城市", "城市代码", "所属地域", "胡焕庸线"] if c in pop.columns]
    rows = []
    for city, g in pop.groupby("城市", dropna=False):
        g = g.sort_values("年份")
        base_rows = g[g["年份"].eq(BASE_YEAR)]
        end_rows = g[g["年份"].eq(SOCIO_END_YEAR)]
        first_meta = base_rows.iloc[0] if len(base_rows) else g.iloc[0]
        base_row = base_rows.iloc[0] if len(base_rows) else None
        end_row = end_rows.iloc[-1] if len(end_rows) else None
        a0, a1 = acc0.get(city, np.nan), acc1.get(city, np.nan)
        t0, t1 = time0.get(city, np.nan), time1.get(city, np.nan)
        res0 = pd.to_numeric(base_row.get("常住人口数(万人)"), errors="coerce") if base_row is not None else np.nan
        res1 = pd.to_numeric(end_row.get("常住人口数(万人)"), errors="coerce") if end_row is not None else np.nan
        hukou_series = pd.to_numeric(g["户籍人口数(万人)"], errors="coerce")
        hukou_sum = hukou_series.sum(min_count=1)
        netin_series = pd.to_numeric(g["人口净流入(万人)"], errors="coerce")
        netin_sum = netin_series.sum(min_count=1)
        netin_mean = netin_series.mean()
        expected_social_years = SOCIO_END_YEAR - BASE_YEAR + 1
        observed_years = int(g["年份"].dropna().nunique())
        coverage_complete = bool(len(base_rows) and len(end_rows) and observed_years >= expected_social_years)
        row = {c: first_meta.get(c, pd.NA) for c in meta_cols}
        row.update({
            ACC_BASE_COL: a0,
            TIME_BASE_COL: t0,
            "十年间acc增长值": a1 - a0,
            "十年间acc增长率": (a1 - a0) / a0 if np.isfinite(a0) and a0 != 0 and np.isfinite(a1) else np.nan,
            "十年间time_median下降值": t0 - t1,
            "十年间time_median下降率": (t0 - t1) / t0 if np.isfinite(t0) and t0 != 0 and np.isfinite(t1) else np.nan,
            "十年间常住人口增长率": (res1 - res0) / res0 if np.isfinite(res0) and res0 != 0 and np.isfinite(res1) else np.nan,
            "十年间人口净增长值": res1 - res0 if np.isfinite(res0) and np.isfinite(res1) else np.nan,
            "十年间人口净增长率": (res1 - res0) / res0 if np.isfinite(res0) and res0 != 0 and np.isfinite(res1) else np.nan,
            "逐年人口增长率均值": pd.to_numeric(g["常住人口增长率"], errors="coerce").mean(),
            "累计人口净流入值": netin_sum,
            "年均人口净流入值": netin_mean,
            "十年间人口净流入值": netin_sum,
            "社会经济年份数": observed_years,
            "社会经济首末年完整": coverage_complete,
            "十年间净流入人口占比": netin_sum / hukou_sum if pd.notna(hukou_sum) and hukou_sum != 0 else np.nan,
            "十年间人口净流入率": netin_sum / hukou_sum if pd.notna(hukou_sum) and hukou_sum != 0 else np.nan,
            RESPOP_BASE_COL: res0,
            "ResPop_growth": res1 - res0 if np.isfinite(res0) and np.isfinite(res1) else np.nan,
            "ResPop_growth_rate": (res1 - res0) / res0 if np.isfinite(res0) and res0 != 0 and np.isfinite(res1) else np.nan,
            "Ppo_NetIn": netin_mean,
            "Ppo_NetIn_rate": netin_sum / hukou_sum if pd.notna(hukou_sum) and hukou_sum != 0 else np.nan,
        })
        rows.append(row)
    changes = pd.DataFrame(rows)


    gp = gdp[gdp["year"].isin([BASE_YEAR, SOCIO_END_YEAR])].pivot_table(index="城市", columns="year", values="GDP_per", aggfunc="first")
    if BASE_YEAR in gp.columns:
        gp = gp.rename(columns={BASE_YEAR: GDP_BASE_COL})
    else:
        gp[GDP_BASE_COL] = np.nan
    end_col = SOCIO_END_YEAR if SOCIO_END_YEAR in gp.columns else None
    gp["十年间人均GDP净增长值"] = gp[end_col] - gp[GDP_BASE_COL] if end_col is not None else np.nan
    gp["十年间人均GDP增长率"] = gp["十年间人均GDP净增长值"] / gp[GDP_BASE_COL]
    gp["GDP_growth"] = gp["十年间人均GDP净增长值"]
    gp["GDP_growth_pct"] = gp["十年间人均GDP增长率"] * 100
    gp = gp[[GDP_BASE_COL, "GDP_growth", "GDP_growth_pct", "十年间人均GDP净增长值", "十年间人均GDP增长率"]].reset_index()
    changes = changes.merge(gp, on="城市", how="left")
    return panel, changes


def main():
    CITY_DYNAMICS_ROOT.mkdir(parents=True, exist_ok=True)
    pop = load_population_flow(); gdp = load_city_gdp()
    pop.to_csv(CITY_DYNAMICS_ROOT / "population_flow_prepared.csv", index=False, encoding="utf-8-sig")
    gdp.to_csv(CITY_DYNAMICS_ROOT / "gdp_city_prepared.csv", index=False, encoding="utf-8-sig")
    panel, changes = build_city_changes(pop, gdp)
    panel.to_csv(CITY_DYNAMICS_ROOT / "pop_acc.csv", index=False, encoding="utf-8-sig")
    changes.to_csv(CITY_DYNAMICS_ROOT / "pop_acc_changes.csv", index=False, encoding="utf-8-sig")
    print(f"4_1 city changes: {len(changes):,} cities -> {CITY_DYNAMICS_ROOT}")


if __name__ == "__main__":
    main()
