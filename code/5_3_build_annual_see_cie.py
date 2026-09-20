# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd

from config import CHANGE_YEARS, HOSPITAL_CHANGES_ROOT, SEE_CIE_ANNUAL_ROOT
from utils.extended_analysis import read_csv_robust, read_stats, compute_deltas, city_level_from_name, norm6

OUTCOME_COLS = ["pop_median", "pop_gini", "pop_theil", "pop_atkinson_05", "zero_access_pop_pct", "p90_p10", "p80_p20"]
DELTA_COLS = ["acc_delta", "gini_delta", "theil_delta", "atkinson_05_delta", "zero_access_pop_pct_delta", "p90_p10_delta", "p80_p20_delta"]
RENAME_CURRENT = {
    "pop_median": "acc_median", "pop_gini": "acc_gini", "pop_theil": "acc_theil",
    "pop_atkinson_05": "acc_atkinson_05", "zero_access_pop_pct": "acc_zero_access_pop_pct",
    "p90_p10": "acc_p90_p10", "p80_p20": "acc_p80_p20",
}


def prepare_stats():
    city = compute_deltas(read_stats("accessibility", "city"), "地级")
    county = compute_deltas(read_stats("accessibility", "county"), "县级码")
    prov = compute_deltas(read_stats("accessibility", "provincial"), "省级")
    return city, county, prov


def build_one(year: int, city_stats: pd.DataFrame, county_stats: pd.DataFrame, prov_stats: pd.DataFrame):
    src = HOSPITAL_CHANGES_ROOT / "annual" / f"hosps_changed_type_{year}.csv"
    data = read_csv_robust(src)
    data["县级码"] = data["县级码"].map(norm6)

    keys = ["省级", "city_name_norm", "县级", "县级码", "县级类"]
    pop_cols = [c for c in ["county_pop", "city_pop", "county_pop_prev", "county_pop_curr", "city_pop_prev", "city_pop_curr", "population_denominator_years"] if c in data.columns]
    meta = data[keys + pop_cols].drop_duplicates(subset=["县级码", "city_name_norm", "县级"], keep="first")
    new = data[data["change_type"] == "new"].groupby(["city_name_norm", "县级码"], dropna=False).agg(new_hosp_num=("name", "size"), new_hosp_beds=("beds_added", "sum")).reset_index()

    intensive = data[data["change_type"] == "increase"].groupby(["city_name_norm", "县级码"], dropna=False).agg(expanded_hosp_num=("name", "size"), expanded_beds=("beds_added", "sum")).reset_index()
    decrease = data[data["change_type"] == "decrease"].assign(decreased_beds=lambda x: -pd.to_numeric(x["beds_added"], errors="coerce")).groupby(["city_name_norm", "县级码"], dropna=False).agg(decreased_hosp_num=("name", "size"), decreased_beds=("decreased_beds", "sum")).reset_index()
    closed = data[data["change_type"] == "closed"].assign(closed_beds=lambda x: -pd.to_numeric(x["beds_added"], errors="coerce")).groupby(["city_name_norm", "县级码"], dropna=False).agg(closed_hosp_num=("name", "size"), closed_beds=("closed_beds", "sum")).reset_index()
    county = meta.merge(new, on=["city_name_norm", "县级码"], how="left").merge(intensive, on=["city_name_norm", "县级码"], how="left").merge(decrease, on=["city_name_norm", "县级码"], how="left").merge(closed, on=["city_name_norm", "县级码"], how="left")


    component_cols = ["new_hosp_beds", "expanded_beds", "decreased_beds", "closed_beds"]
    mask = county[component_cols].notna().any(axis=1)
    county.loc[mask, component_cols] = county.loc[mask, component_cols].fillna(0)
    county.loc[mask, "net_SEE_beds"] = county.loc[mask, "new_hosp_beds"] - county.loc[mask, "closed_beds"]
    county.loc[mask, "net_CIE_beds"] = county.loc[mask, "expanded_beds"] - county.loc[mask, "decreased_beds"]
    county.loc[mask, "net_total_beds"] = county.loc[mask, "net_SEE_beds"] + county.loc[mask, "net_CIE_beds"]
    county.loc[mask, "Extensive_index"] = county.loc[mask, "net_SEE_beds"] / county.loc[mask, "city_pop"] * 10000
    county.loc[mask, "Intensive_index"] = county.loc[mask, "net_CIE_beds"] / county.loc[mask, "city_pop"] * 10000
    county.loc[mask, "Dominance"] = county.loc[mask, "Extensive_index"] - county.loc[mask, "Intensive_index"]
    county["地级"] = county["city_name_norm"]
    county["city_level"] = city_level_from_name(county["地级"])
    county.to_csv(SEE_CIE_ANNUAL_ROOT / f"county_SEE_CIE_{year}.csv", index=False, encoding="utf-8-sig")

    city = county.groupby(["省级", "地级", "city_level"], dropna=False, as_index=False).agg(
        new_hosp_beds=("new_hosp_beds", "sum"), expanded_beds=("expanded_beds", "sum"),
        decreased_beds=("decreased_beds", "sum"), closed_beds=("closed_beds", "sum"), city_pop=("city_pop", "first")
    )
    city["net_SEE_beds"] = city["new_hosp_beds"] - city["closed_beds"]
    city["net_CIE_beds"] = city["expanded_beds"] - city["decreased_beds"]
    city["net_total_beds"] = city["net_SEE_beds"] + city["net_CIE_beds"]
    city["SEE_city"] = city["net_SEE_beds"] / city["city_pop"] * 10000
    city["CIE_city"] = city["net_CIE_beds"] / city["city_pop"] * 10000
    m = city["SEE_city"].notna() | city["CIE_city"].notna(); city.loc[m, "Dominance_city"] = city.loc[m, "SEE_city"].fillna(0) - city.loc[m, "CIE_city"].fillna(0)

    st = city_stats[city_stats["Year"] == year].copy()
    need = ["地级"] + [c for c in OUTCOME_COLS + DELTA_COLS if c in st.columns]
    city = city.merge(st[need], on="地级", how="left").rename(columns=RENAME_CURRENT)


    target_current = [RENAME_CURRENT[c] for c in OUTCOME_COLS]
    source_all = [c for c in OUTCOME_COLS + DELTA_COLS if c in st.columns]
    missing = city["acc_median"].isna() if "acc_median" in city.columns else pd.Series(False, index=city.index)
    if missing.any():
        ps = prov_stats[prov_stats["Year"] == year].drop_duplicates("省级").set_index("省级")[source_all].rename(columns=RENAME_CURRENT)
        idx = city.loc[missing & city["地级"].isin(["北京市", "上海市", "天津市", "重庆市"]), "省级"]
        if len(idx):
            cols = [c for c in target_current + DELTA_COLS if c in ps.columns]
            city.loc[idx.index, cols] = ps.reindex(idx).loc[:, cols].to_numpy()
    missing = city["acc_median"].isna() if "acc_median" in city.columns else pd.Series(False, index=city.index)
    if missing.any():
        cs = county_stats[county_stats["Year"] == year].drop_duplicates("县级").set_index("县级")[source_all].rename(columns=RENAME_CURRENT)
        names = city.loc[missing, "地级"]
        cols = [c for c in target_current + DELTA_COLS if c in cs.columns]
        city.loc[missing, cols] = cs.reindex(names).loc[:, cols].to_numpy()

    city.to_csv(SEE_CIE_ANNUAL_ROOT / f"city_SEE_CIE_{year}.csv", index=False, encoding="utf-8-sig")
    return county, city


def main():
    SEE_CIE_ANNUAL_ROOT.mkdir(parents=True, exist_ok=True)
    city_stats, county_stats, prov_stats = prepare_stats()
    for year in CHANGE_YEARS:
        county, city = build_one(year, city_stats, county_stats, prov_stats)
        print(f"5_3 {year}: county={len(county):,}; city={len(city):,}")


if __name__ == "__main__":
    main()
