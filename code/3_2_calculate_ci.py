# -*- coding: utf-8 -*-
"""3_2 CI calculation: national, regional, and city-size CI summary tables.

Figure rendering is intentionally separated into ``fig4_2_ci_plots.py``.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from config import CI_YEARS, CI_MATCHED_ROOT, CI_ANALYSIS_ROOT, SERVICE_SCOPES, PROFILES, REGIONS, CITY_LEVEL_MERGE_MAP, MEGA_LABEL, SUPER_LABEL, COMBINED_GROUP_NAME
from utils.cache import clean_directory
from utils.concentration_index import concentration_index_weighted
from utils.multiscale import validate_complete_province_parts


def validate_ci_inputs(scope, profile):

    for year in CI_YEARS:
        folder=CI_MATCHED_ROOT/str(year)/scope/profile/"parts"
        parts=sorted(folder.glob("province_*.parquet")) if folder.exists() else []
        validate_complete_province_parts(parts, f"3_1 input for 3_2 {year}/{scope}/{profile}")


def read_ci_year(year,scope,profile,columns):
    folder=CI_MATCHED_ROOT/str(year)/scope/profile/"parts"; parts=sorted(folder.glob("province_*.parquet")) if folder.exists() else []
    if not parts: raise FileNotFoundError(f"未找到 3_1 CI parquet：{folder}")
    return pd.concat([pd.read_parquet(p,columns=columns) for p in parts],ignore_index=True)


def clean_sample(df,rank_var,rule="gt0"):
    sub=df[[rank_var,"acc","pop"]].copy().replace([np.inf,-np.inf],np.nan)
    for c in [rank_var,"acc","pop"]: sub[c]=pd.to_numeric(sub[c],errors="coerce")
    sub=sub.dropna(subset=[rank_var,"acc","pop"]); sub=sub[sub["pop"]>0]
    if rule=="gt0": sub=sub[sub[rank_var]>0]
    elif rule=="ge0": sub=sub[sub[rank_var]>=0]
    else: raise ValueError(rule)
    return sub


def count_cities_in_ci(df, rank_var, rule="gt0"):

    required=[rank_var,"acc","pop","地级码_use"]
    missing=[c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"CI 城市计数缺字段：{missing}")
    sub=df[required].copy().replace([np.inf,-np.inf],np.nan)
    for c in [rank_var,"acc","pop"]:
        sub[c]=pd.to_numeric(sub[c],errors="coerce")
    sub=sub.dropna(subset=[rank_var,"acc","pop","地级码_use"])
    sub=sub[sub["pop"]>0]
    if rule=="gt0":
        sub=sub[sub[rank_var]>0]
    elif rule=="ge0":
        sub=sub[sub[rank_var]>=0]
    else:
        raise ValueError(rule)
    return int(sub["地级码_use"].astype("string").nunique())


def ci_value(df,rank_var,rule="gt0"):
    sub=clean_sample(df,rank_var,rule)
    return concentration_index_weighted(sub,rank_var,"acc","pop") if len(sub)>=2 else np.nan


def curve(df,rank_var,rule="gt0"):
    sub=clean_sample(df,rank_var,rule)
    return concentration_curve_weighted(sub,rank_var,"acc","pop") if len(sub)>=2 else None


def merge_city_level(s): return s.replace(CITY_LEVEL_MERGE_MAP)
def group_mask(s,g): return s.isin([MEGA_LABEL,SUPER_LABEL]) if g==COMBINED_GROUP_NAME else s.eq(g)


def compute_all(scope,profile,out_dir):
    ng=[]; nm=[]
    rg=[]; rs=[]; cg=[]; cs=[]
    rm=[]; rms=[]; cm=[]; cms=[]
    all_groups=set()

    for year in CI_YEARS:
        df=read_ci_year(year,scope,profile,["pop","acc","GDP_per","Minority_rate","Eastern","city_level","地级码_use"])
        pop_all=pd.to_numeric(df["pop"],errors="coerce")
        pop_total=float(pop_all[(pop_all>0)&np.isfinite(pop_all)].sum())

        # National CI: GDP rank and ethnic-minority-share rank.
        gdp_sub=clean_sample(df,"GDP_per","gt0")
        min_sub=clean_sample(df,"Minority_rate","ge0")
        gdp_ci=concentration_index_weighted(gdp_sub,"GDP_per","acc","pop") if len(gdp_sub)>=2 else np.nan
        min_ci=concentration_index_weighted(min_sub,"Minority_rate","acc","pop") if len(min_sub)>=2 else np.nan
        gdp_pop=float(gdp_sub["pop"].sum()) if len(gdp_sub) else 0.0
        min_pop=float(min_sub["pop"].sum()) if len(min_sub) else 0.0
        ng.append({"year":year,"CI":gdp_ci,"pop_in_CI":gdp_pop,"pop_total":pop_total,"pop_coverage":gdp_pop/pop_total if pop_total>0 else np.nan})
        nm.append({"year":year,"CI":min_ci,"pop_in_CI":min_pop,"pop_total":pop_total,"pop_coverage":min_pop/pop_total if pop_total>0 else np.nan})
        print(f"3_2 {year}: GDP_CI={gdp_ci:.6f} (pop coverage={gdp_pop/pop_total:.3%}), Minority_CI={min_ci:.6f} (pop coverage={min_pop/pop_total:.3%})" if pop_total>0 else f"3_2 {year}: no population")

        # Eastern / Non-Eastern subgroup CI under both rankings.
        base=df[["Eastern","pop"]].copy().replace([np.inf,-np.inf],np.nan)
        base["pop"]=pd.to_numeric(base["pop"],errors="coerce")
        base=base.dropna(subset=["Eastern","pop"])
        base=base[(base["pop"]>0)&base["Eastern"].isin(REGIONS)]
        total=float(base["pop"].sum())
        for region in REGIONS:
            pop_total_region=float(base.loc[base["Eastern"].eq(region),"pop"].sum())
            region_df=df[df["Eastern"].eq(region)]

            gsub=clean_sample(region_df,"GDP_per","gt0")
            gci=concentration_index_weighted(gsub,"GDP_per","acc","pop") if len(gsub)>=2 else np.nan
            gpop=float(gsub["pop"].sum()) if len(gsub) else 0.0
            rg.append({"year":year,"region":region,"CI":gci})
            rs.append({
                "year":year,"region":region,
                "n_cities_in_CI":count_cities_in_ci(region_df,"GDP_per","gt0"),
                "pop_in_CI":gpop,"pop_total_region":pop_total_region,
                "pop_share_region":pop_total_region/total if total>0 else np.nan,
                "excluded_pop_due_to_gdp":pop_total_region-gpop,
            })

            msub=clean_sample(region_df,"Minority_rate","ge0")
            mci=concentration_index_weighted(msub,"Minority_rate","acc","pop") if len(msub)>=2 else np.nan
            mpop=float(msub["pop"].sum()) if len(msub) else 0.0
            rm.append({"year":year,"region":region,"CI":mci})
            rms.append({
                "year":year,"region":region,
                "n_cities_in_CI":count_cities_in_ci(region_df,"Minority_rate","ge0"),
                "pop_in_CI":mpop,"pop_total_region":pop_total_region,
                "pop_share_region":pop_total_region/total if total>0 else np.nan,
                "excluded_pop_due_to_minority":pop_total_region-mpop,
            })

        # City-size subgroup CI under both rankings.
        df["city_level_merged"]=merge_city_level(df["city_level"])
        groups=[g for g in df["city_level_merged"].dropna().unique().tolist() if str(g).strip()]
        if any(g in {MEGA_LABEL,SUPER_LABEL} for g in groups):
            groups.append(COMBINED_GROUP_NAME)
        groups=list(dict.fromkeys(groups))
        all_groups.update(groups)

        base=df[["city_level_merged","pop"]].copy().replace([np.inf,-np.inf],np.nan)
        base["pop"]=pd.to_numeric(base["pop"],errors="coerce")
        base=base.dropna(subset=["city_level_merged","pop"])
        base=base[base["pop"]>0]
        total=float(base["pop"].sum())
        for g in groups:
            mask=group_mask(df["city_level_merged"],g)
            pop_total_group=float(base.loc[group_mask(base["city_level_merged"],g),"pop"].sum())
            group_df=df[mask]

            gsub=clean_sample(group_df,"GDP_per","gt0")
            gci=concentration_index_weighted(gsub,"GDP_per","acc","pop") if len(gsub)>=2 else np.nan
            gpop=float(gsub["pop"].sum()) if len(gsub) else 0.0
            cg.append({"year":year,"city_level":g,"CI":gci})
            cs.append({
                "year":year,"city_level":g,
                "n_cities_in_CI":count_cities_in_ci(group_df,"GDP_per","gt0"),
                "pop_in_CI":gpop,"pop_total_group":pop_total_group,
                "pop_share_group":pop_total_group/total if total>0 else np.nan,
                "excluded_pop_due_to_gdp":pop_total_group-gpop,
            })

            msub=clean_sample(group_df,"Minority_rate","ge0")
            mci=concentration_index_weighted(msub,"Minority_rate","acc","pop") if len(msub)>=2 else np.nan
            mpop=float(msub["pop"].sum()) if len(msub) else 0.0
            cm.append({"year":year,"city_level":g,"CI":mci})
            cms.append({
                "year":year,"city_level":g,
                "n_cities_in_CI":count_cities_in_ci(group_df,"Minority_rate","ge0"),
                "pop_in_CI":mpop,"pop_total_group":pop_total_group,
                "pop_share_group":pop_total_group/total if total>0 else np.nan,
                "excluded_pop_due_to_minority":pop_total_group-mpop,
            })

    ng=pd.DataFrame(ng)
    nm=pd.DataFrame(nm)
    rg=pd.DataFrame(rg).sort_values(["region","year"])
    rs=pd.DataFrame(rs).sort_values(["region","year"])
    cg=pd.DataFrame(cg).sort_values(["city_level","year"])
    cs=pd.DataFrame(cs).sort_values(["city_level","year"])
    rm=pd.DataFrame(rm).sort_values(["region","year"])
    rms=pd.DataFrame(rms).sort_values(["region","year"])
    cm=pd.DataFrame(cm).sort_values(["city_level","year"])
    cms=pd.DataFrame(cms).sort_values(["city_level","year"])

    ng.to_csv(out_dir/"acc_CI_results_by_GDP.csv",index=False,encoding="utf-8-sig")
    nm.to_csv(out_dir/"acc_CI_results_by_Minority.csv",index=False,encoding="utf-8-sig")

    rg.to_csv(out_dir/"acc_CI_results_by_GDP_region.csv",index=False,encoding="utf-8-sig")
    rs.to_csv(out_dir/"Eastern_NotEastern_city_pop_summary.csv",index=False,encoding="utf-8-sig")
    cg.to_csv(out_dir/"acc_CI_results_by_GDP_city_level.csv",index=False,encoding="utf-8-sig")
    cs.to_csv(out_dir/"city_level_city_pop_summary.csv",index=False,encoding="utf-8-sig")

    rm.to_csv(out_dir/"acc_CI_results_by_Minority_region.csv",index=False,encoding="utf-8-sig")
    rms.to_csv(out_dir/"Eastern_NotEastern_city_pop_summary_by_Minority.csv",index=False,encoding="utf-8-sig")
    cm.to_csv(out_dir/"acc_CI_results_by_Minority_city_level.csv",index=False,encoding="utf-8-sig")
    cms.to_csv(out_dir/"city_level_city_pop_summary_by_Minority.csv",index=False,encoding="utf-8-sig")

    return ng,nm,rg,cg,rm,cm,all_groups

def main():
    for scope in SERVICE_SCOPES:
        for profile in PROFILES:
            validate_ci_inputs(scope,profile)
            out=CI_ANALYSIS_ROOT/scope/profile
            clean_directory(out)
            print(f"\n{'='*90}\n3_2 CI | {scope} | {profile}\n{'='*90}")
            compute_all(scope,profile,out)
            print(f"3_2 done -> {out}")

if __name__=="__main__": main()
