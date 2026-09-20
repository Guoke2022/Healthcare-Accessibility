# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from config import (
    BASE_YEAR, END_YEAR, CHANGE_YEARS, HOSPITAL_CHANGES_ROOT,
    HOSPITAL_DESCRIPTIVE_ROOT, CITY_DYNAMICS_ROOT, MAKE_PLOTS, PLOT_DPI, MODE_COASTAL,
)
from utils.extended_analysis import read_csv_robust, read_stats, set_nature_style


def _mw(a, b):
    a = pd.to_numeric(a, errors="coerce").dropna(); b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan
    r = mannwhitneyu(a, b, alternative="two-sided")
    return float(r.statistic), float(r.pvalue)


def annual_county_type_summary() -> pd.DataFrame:
    rows = []
    for year in CHANGE_YEARS:
        p = HOSPITAL_CHANGES_ROOT / "annual" / f"hosps_changed_type_{year}.csv"
        df = read_csv_robust(p)
        t = df["县级类"].astype("string")
        bed = pd.to_numeric(df.get(f"beds_{year}"), errors="coerce")

        active = ~df["change_type"].astype("string").eq("closed")
        is_dist = t.eq("市辖区") & active
        is_county = t.isin(["县", "县级市"]) & active
        rows.append({
            "year": year,
            "Districts_hospital_n": int(is_dist.sum()),
            "Counties_hospital_n": int(is_county.sum()),
            "Districts_beds": float(bed[is_dist].sum(min_count=1)),
            "Counties_beds": float(bed[is_county].sum(min_count=1)),
        })
    return pd.DataFrame(rows)


def build_overall_city_index() -> pd.DataFrame:
    hosp = read_csv_robust(HOSPITAL_CHANGES_ROOT / "hosps_changed_type.csv")
    pop = read_stats("accessibility", "city")
    pop_avg = pop[pop["Year"].between(BASE_YEAR, END_YEAR)].groupby("地级", as_index=False)["pop_num"].mean().rename(columns={"pop_num": "pop_avg_10yr"})


    valid = hosp[hosp["change_type"] != "new"].groupby("city_name_norm").size().rename("hosp_2014").reset_index()
    valid = valid[valid["hosp_2014"] > 0]
    new = hosp[hosp["change_type"] == "new"].groupby("city_name_norm", as_index=False)["beds_added"].sum().rename(columns={"beds_added": "new_hosp_beds"})

    #   net SEE = new beds - closed beds
    #   net CIE = increased beds - decreased beds

    old = hosp[hosp["change_type"] == "increase"].groupby("city_name_norm", as_index=False)["beds_added"].sum().rename(columns={"beds_added": "expanded_beds"})
    reduced = hosp[hosp["change_type"] == "decrease"].assign(decreased_beds=lambda x: -pd.to_numeric(x["beds_added"], errors="coerce")).groupby("city_name_norm", as_index=False)["decreased_beds"].sum()
    closed = hosp[hosp["change_type"] == "closed"].assign(closed_beds=lambda x: -pd.to_numeric(x["beds_added"], errors="coerce")).groupby("city_name_norm", as_index=False)["closed_beds"].sum()
    meta = hosp.groupby("city_name_norm", as_index=False).agg(省份=("省级", "first"))
    out = valid.merge(meta, on="city_name_norm", how="left").merge(pop_avg, left_on="city_name_norm", right_on="地级", how="inner").merge(new, on="city_name_norm", how="left").merge(old, on="city_name_norm", how="left").merge(reduced, on="city_name_norm", how="left").merge(closed, on="city_name_norm", how="left")
    out = out[pd.to_numeric(out["pop_avg_10yr"], errors="coerce") > 0].copy()
    component_cols = ["new_hosp_beds", "expanded_beds", "decreased_beds", "closed_beds"]
    out[component_cols] = out[component_cols].fillna(0)
    out["net_SEE_beds"] = out["new_hosp_beds"] - out["closed_beds"]
    out["net_CIE_beds"] = out["expanded_beds"] - out["decreased_beds"]
    out["net_total_beds"] = out["net_SEE_beds"] + out["net_CIE_beds"]
    out["Extensive_index"] = out["net_SEE_beds"] / out["pop_avg_10yr"] * 10000
    out["Intensive_index"] = out["net_CIE_beds"] / out["pop_avg_10yr"] * 10000
    out["Dominance"] = out["Extensive_index"] - out["Intensive_index"]
    out["Mode"] = np.where(out["Dominance"] > 0, "SEE-dominant", "CIE-dominant")
    out["Coastal_Inland"] = np.where(out["省份"].isin(MODE_COASTAL), "Coastal", "Inland")

    changes_path = CITY_DYNAMICS_ROOT / "pop_acc_changes.csv"
    if changes_path.exists():
        changes = read_csv_robust(changes_path)
        out = out.merge(changes, left_on="city_name_norm", right_on="城市", how="left", suffixes=("", "_changes"))
        if "省份_changes" in out.columns:
            out["省份"] = out["省份"].fillna(out["省份_changes"])
            out = out.drop(columns=["省份_changes"])

    acc = read_stats("accessibility", "city")
    cols = ["pop_median", "pop_gini", "pop_theil", "pop_atkinson_05"]
    a0 = acc[acc["Year"] == BASE_YEAR][["地级"] + cols].rename(columns={c: f"{c}_{BASE_YEAR}" for c in cols})
    a1 = acc[acc["Year"] == END_YEAR][["地级"] + cols].rename(columns={c: f"{c}_{END_YEAR}" for c in cols})
    ch = a0.merge(a1, on="地级", how="inner")
    for c in cols:
        ch[f"d_{c}"] = ch[f"{c}_{END_YEAR}"] - ch[f"{c}_{BASE_YEAR}"]
    out = out.merge(ch, left_on="city_name_norm", right_on="地级", how="left", suffixes=("", "_acc"))
    return out


def mode_summary(df: pd.DataFrame):
    rows = []
    for metric in ["d_pop_median", "d_pop_gini", "d_pop_theil", "d_pop_atkinson_05"]:
        if metric not in df.columns:
            continue
        desc = df.groupby("Mode")[metric].agg(
            N="count", mean="mean", median="median",
            q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75),
            improve_rate=lambda x: (x < 0).mean() if metric != "d_pop_median" else np.nan,
        ).reset_index()
        desc.insert(0, "metric", metric); rows.append(desc)
    summary = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


    region_rows = []
    tmp = df.copy(); tmp["Mode_Region"] = tmp["Mode"].astype(str) + " - " + tmp["Coastal_Inland"].astype(str)
    for metric in ["d_pop_median", "d_pop_gini", "d_pop_theil", "d_pop_atkinson_05"]:
        if metric not in tmp.columns: continue
        d = tmp.groupby("Mode_Region")[metric].agg(
            N="count", mean="mean", median="median",
            q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75),
            improve_rate=lambda x: (x < 0).mean() if metric != "d_pop_median" else np.nan,
        ).reset_index(); d.insert(0,"metric",metric); region_rows.append(d)
    region_summary = pd.concat(region_rows, ignore_index=True) if region_rows else pd.DataFrame()

    tests = []
    def add_test(metric, a_label, b_label, a, b):
        u,p=_mw(a,b); tests.append({"metric":metric,"comparison":f"{a_label} vs {b_label}","U":u,"p":p})
    for metric in ["d_pop_median", "d_pop_gini", "d_pop_theil", "d_pop_atkinson_05"]:
        if metric not in df.columns: continue
        add_test(metric,"SEE-dominant","CIE-dominant",df.loc[df["Mode"]=="SEE-dominant",metric],df.loc[df["Mode"]=="CIE-dominant",metric])

        pairs=[
            ("SEE-dominant - Coastal","SEE-dominant - Inland"),
            ("CIE-dominant - Coastal","CIE-dominant - Inland"),
            ("SEE-dominant - Coastal","CIE-dominant - Coastal"),
            ("SEE-dominant - Inland","CIE-dominant - Inland"),
        ]
        for a,b in pairs:
            add_test(metric,a,b,tmp.loc[tmp["Mode_Region"]==a,metric],tmp.loc[tmp["Mode_Region"]==b,metric])
    return summary, region_summary, pd.DataFrame(tests)


def make_plots(annual: pd.DataFrame, city: pd.DataFrame):
    if not MAKE_PLOTS:
        return
    plt = set_nature_style()
    for a, b, ylabel, name in [
        ("Districts_hospital_n", "Counties_hospital_n", "Number of hospitals", f"hospital_by_county_type_{CHANGE_YEARS[0]}_{CHANGE_YEARS[-1]}.png"),
        ("Districts_beds", "Counties_beds", "Hospital beds", f"hospital_beds_by_county_type_{CHANGE_YEARS[0]}_{CHANGE_YEARS[-1]}.png"),
    ]:
        fig, ax = plt.subplots(figsize=(4.8, 3.2)); ax.plot(annual["year"], annual[a], label="Districts", linewidth=1.5); ax.plot(annual["year"], annual[b], label="Counties", linewidth=1.5); ax.set_xlabel("Year"); ax.set_ylabel(ylabel); ax.set_xticks(annual["year"]); ax.legend(); fig.tight_layout(); fig.savefig(HOSPITAL_DESCRIPTIVE_ROOT / name, dpi=PLOT_DPI, bbox_inches="tight"); plt.close(fig)

    for metric, ylabel, name in [
        ("d_pop_median", "ΔAccessibility", "mode_accessibility_box.png"),
        ("d_pop_gini", "ΔGini", "mode_gini_box.png"),
        ("d_pop_theil", "ΔTheil T", "mode_theil_box.png"),
        ("d_pop_atkinson_05", "ΔAtkinson (ε=0.5)", "mode_atkinson_05_box.png"),
    ]:
        if metric not in city.columns:
            continue
        groups = [pd.to_numeric(city.loc[city["Mode"] == m, metric], errors="coerce").dropna().values for m in ["SEE-dominant", "CIE-dominant"]]
        fig, ax = plt.subplots(figsize=(4.8, 3.6)); ax.boxplot(groups, tick_labels=["SEE-dominant", "CIE-dominant"], showfliers=False); ax.set_ylabel(ylabel); fig.tight_layout(); fig.savefig(HOSPITAL_DESCRIPTIVE_ROOT / name, dpi=PLOT_DPI, bbox_inches="tight"); plt.close(fig)

    tmp = city.copy(); tmp["Mode_Region"] = tmp["Mode"].astype(str) + " - " + tmp["Coastal_Inland"].astype(str)
    order = ["SEE-dominant - Coastal", "SEE-dominant - Inland", "CIE-dominant - Coastal", "CIE-dominant - Inland"]
    for metric, ylabel, name in [
        ("d_pop_median", "ΔAccessibility", "mode_region_accessibility_box.png"),
        ("d_pop_gini", "ΔGini", "mode_region_gini_box.png"),
        ("d_pop_theil", "ΔTheil T", "mode_region_theil_box.png"),
        ("d_pop_atkinson_05", "ΔAtkinson (ε=0.5)", "mode_region_atkinson_05_box.png"),
    ]:
        if metric not in tmp.columns: continue
        groups=[pd.to_numeric(tmp.loc[tmp["Mode_Region"]==g,metric],errors="coerce").dropna().values for g in order]
        fig,ax=plt.subplots(figsize=(7.2,3.8)); ax.boxplot(groups,tick_labels=["SEE-C","SEE-I","CIE-C","CIE-I"],showfliers=False); ax.set_ylabel(ylabel); fig.tight_layout(); fig.savefig(HOSPITAL_DESCRIPTIVE_ROOT/name,dpi=PLOT_DPI,bbox_inches="tight"); plt.close(fig)


def main():
    HOSPITAL_DESCRIPTIVE_ROOT.mkdir(parents=True, exist_ok=True)
    annual = annual_county_type_summary(); annual.to_csv(HOSPITAL_DESCRIPTIVE_ROOT / "county_type_inventory_trends.csv", index=False, encoding="utf-8-sig")
    city = build_overall_city_index(); city.to_csv(HOSPITAL_DESCRIPTIVE_ROOT / "city_expansion_mode_index.csv", index=False, encoding="utf-8-sig")

    city.to_csv(HOSPITAL_DESCRIPTIVE_ROOT / "city_indexs.csv", index=False, encoding="utf-8-sig")
    summary, region_summary, tests = mode_summary(city)
    summary.to_csv(HOSPITAL_DESCRIPTIVE_ROOT / "mode_outcome_summary.csv", index=False, encoding="utf-8-sig")
    region_summary.to_csv(HOSPITAL_DESCRIPTIVE_ROOT / "mode_region_outcome_summary.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(HOSPITAL_DESCRIPTIVE_ROOT / "mode_mannwhitney_tests.csv", index=False, encoding="utf-8-sig")
    make_plots(annual, city)
    print(f"5_2 done -> {HOSPITAL_DESCRIPTIVE_ROOT}")


if __name__ == "__main__":
    main()
