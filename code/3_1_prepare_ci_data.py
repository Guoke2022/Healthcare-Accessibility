# -*- coding: utf-8 -*-
"""Module utilities for 3 1 prepare ci data."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from config import CI_YEARS, GDP_XLSX, GDP_SHEET, MINORITY_CSV, CI_MATCHED_ROOT, MUNICIPALITIES, LEGACY_MINORITY_GT_ZERO, ETHNIC_THRESHOLD_PCT
from config import SERVICE_SCOPES, PROFILES
from utils.multiscale import matched_parts_dir, discover_parts, validate_complete_province_parts
from utils.cache import build_fingerprint, prepare_stage_directory
from config import RUN_POLICY


def norm6(x):
    if pd.isna(x): return pd.NA
    s=str(x).strip()
    if not s or s.lower()=="nan": return pd.NA
    s=s.split(".")[0].strip()
    return s.zfill(6) if s else pd.NA


def county_to_pref(x):
    c=norm6(x)
    return pd.NA if pd.isna(c) or c=="000000" else c[:4]+"00"


def build_pref_use(df):
    county=df["县级码"].map(norm6)
    pref_raw=df["地级码"].map(norm6) if "地级码" in df.columns else pd.Series(pd.NA,index=df.index,dtype="object")
    pref_use=pref_raw.copy(); need=pref_use.isna()|pref_use.eq("000000"); pref_use.loc[need]=county.loc[need].map(county_to_pref)
    return county,pref_raw.replace("000000",pd.NA),pref_use.replace("000000",pd.NA)


def _assert_unique_value(df, keys, value_col, label):
    dup = df[df.duplicated(keys, keep=False)].copy()
    if dup.empty:
        return
    conflicts = dup.groupby(keys, dropna=False)[value_col].nunique(dropna=True)
    conflicts = conflicts[conflicts > 1]
    if len(conflicts):
        raise ValueError(f"{label} 在键 {keys} 上存在冲突值，不能用 drop_duplicates 静默取第一条；示例={conflicts.head(10).to_dict()}")


GDP_REQUIRED_COLS = {"年份", "省份", "城市", "城市代码", "人均地区生产总值(元)"}


def _resolve_gdp_sheet() -> str:
    """Helper for _resolve_gdp_sheet."""
    return GDP_SHEET


def load_gdp_tables():
    if not GDP_XLSX.exists(): raise FileNotFoundError(f"GDP Excel 不存在：{GDP_XLSX}")
    sheet = _resolve_gdp_sheet()
    x=pd.read_excel(GDP_XLSX,sheet_name=sheet,usecols=["年份","省份","城市","城市代码","人均地区生产总值(元)"],dtype={"城市代码":str}).rename(columns={"年份":"year","省份":"prov_name","城市":"city_name","城市代码":"code","人均地区生产总值(元)":"GDP_per"})
    x["year"]=pd.to_numeric(x["year"],errors="coerce").astype("Int64"); x["code"]=x["code"].map(norm6); x["GDP_per"]=pd.to_numeric(x["GDP_per"],errors="coerce"); x=x[x["year"].isin(CI_YEARS)].copy()
    gdp=x[["year","code","GDP_per"]].dropna(subset=["year","code"])
    _assert_unique_value(gdp,["year","code"],"GDP_per","GDP")
    gdp=gdp.drop_duplicates(["year","code"])
    names=x[["code","prov_name","city_name"]].dropna(subset=["code"])
    _assert_unique_value(names,["code"],"city_name","GDP city name")
    names=names.drop_duplicates("code")
    return gdp,names,sheet


def load_minority_table():
    if not MINORITY_CSV.exists(): raise FileNotFoundError(f"少数民族 CSV 不存在：{MINORITY_CSV}")
    x=pd.read_csv(MINORITY_CSV,dtype={"县级码":str})
    if "少数民族人口比重（%）" not in x.columns: raise KeyError(f"{MINORITY_CSV} 缺少列：少数民族人口比重（%）")
    x["县级码"]=x["县级码"].map(norm6); x=x.rename(columns={"少数民族人口比重（%）":"Minority_rate"}); x["Minority_rate"]=pd.to_numeric(x["Minority_rate"],errors="coerce")
    x=x[x["Minority_rate"]>0].copy() if LEGACY_MINORITY_GT_ZERO else x[x["Minority_rate"]>=0].copy()
    x=x[["县级码","Minority_rate"]].dropna(subset=["县级码"])
    _assert_unique_value(x,["县级码"],"Minority_rate","Minority")
    return x.drop_duplicates("县级码")


def enrich_part(df,year,gdp,names,minority):
    missing={"pop","acc","县级码","省级"}-set(df.columns)
    if missing: raise KeyError(f"2_1 matched parquet 缺少 CI 必需字段：{sorted(missing)}")
    out=df[pd.to_numeric(df["pop"],errors="coerce")>0].copy(); out["year"]=year; out["is_muni"]=out["省级"].isin(MUNICIPALITIES)
    county,pref_raw,pref_use=build_pref_use(out); out["县级码"]=county; out["地级码_raw"]=pref_raw; out["地级码_use"]=pref_use
    out=out.merge(names.rename(columns={"code":"地级码_use"}),on="地级码_use",how="left")
    out=out.merge(gdp.rename(columns={"code":"地级码_use"}),on=["year","地级码_use"],how="left"); out["gdp_source"]=np.where(out["GDP_per"].notna(),"pref_code","missing")
    out["muni_code"]=np.where(out["is_muni"]&out["地级码_use"].notna(),out["地级码_use"].astype("string").str[:2]+"0000",pd.NA)
    out=out.merge(gdp.rename(columns={"code":"muni_code","GDP_per":"GDP_per_muni"}),on=["year","muni_code"],how="left")
    need=out["is_muni"]&out["GDP_per"].isna()&out["GDP_per_muni"].notna(); out.loc[need,"GDP_per"]=out.loc[need,"GDP_per_muni"]; out.loc[need,"gdp_source"]="muni_prov_code"
    out=out.merge(names.rename(columns={"code":"muni_code","prov_name":"prov_name_muni","city_name":"city_name_muni"}),on="muni_code",how="left")
    need=out["is_muni"]&out["city_name"].isna()&out["city_name_muni"].notna(); out.loc[need,"city_name"]=out.loc[need,"city_name_muni"]
    need=out["is_muni"]&out["prov_name"].isna()&out["prov_name_muni"].notna(); out.loc[need,"prov_name"]=out.loc[need,"prov_name_muni"]
    out=out.merge(minority,on="县级码",how="left")
    out["Ethnic_Han"]=np.where(out["Minority_rate"].notna(),np.where(out["Minority_rate"]<=ETHNIC_THRESHOLD_PCT,"Han","Ethnic"),pd.NA)
    if "Eastern_Region" in out.columns:
        out["Eastern"] = out["Eastern_Region"]
    elif "Coastal_Inland" in out.columns:
        out["Eastern"] = out["Coastal_Inland"].replace({"Coastal":"Eastern","Inland":"NotEastern"})
    else:
        out["Eastern"] = pd.NA
    out.drop(columns=["GDP_per_muni","prov_name_muni","city_name_muni","muni_code","is_muni"],inplace=True,errors="ignore")
    return out


def main():
    gdp,names,gdp_sheet=load_gdp_tables(); minority=load_minority_table(); print(f"GDP rows={len(gdp):,}; sheet={gdp_sheet}; Minority counties={len(minority):,}")
    for scope in SERVICE_SCOPES:
        for profile in PROFILES:
            for year in CI_YEARS:
                src_parts=discover_parts(matched_parts_dir(year,scope,profile))
                if not src_parts: raise FileNotFoundError(f"未找到 2_1 matched parquet：{matched_parts_dir(year,scope,profile)}")
                validate_complete_province_parts(
                    src_parts,
                    f"2_1 input for 3_1 {year}/{scope}/{profile}",
                )
                stage_dir=CI_MATCHED_ROOT/str(year)/scope/profile
                fp=build_fingerprint(config={"stage":"3_1_prepare_ci_data","year":year,"scope":scope,"profile":profile,"legacy_minority_gt_zero":LEGACY_MINORITY_GT_ZERO,"ethnic_threshold_pct":ETHNIC_THRESHOLD_PCT,"gdp_sheet":gdp_sheet},files=[*src_parts,GDP_XLSX,MINORITY_CSV,Path(__file__).resolve(),Path(__file__).resolve().parent / "utils" / "multiscale.py"])
                status=prepare_stage_directory(stage_dir,fp,run_policy=RUN_POLICY,payload={"stage":"3_1_prepare_ci_data","year":year}); out_dir=stage_dir/"parts"; out_dir.mkdir(parents=True,exist_ok=True)
                n=n_gdp=n_min=0; pop_total=pop_gdp_missing=pop_min_missing=0.0; print(f"\n=== 3_1 CI prepare | {year} | {scope} | {profile} | {len(src_parts)} parts ===")
                for i,src in enumerate(src_parts,1):
                    dst=out_dir/src.name
                    if status=="reused_same_input" and dst.exists():
                        q=pd.read_parquet(dst,columns=["pop","GDP_per","Minority_rate"]); print(f"[{i}/{len(src_parts)}] {src.name}: reuse")
                    else:
                        q=enrich_part(pd.read_parquet(src),year,gdp,names,minority); q.to_parquet(dst,index=False,compression="zstd"); print(f"[{i}/{len(src_parts)}] {src.name}: {len(q):,} rows")
                    pop=pd.to_numeric(q["pop"],errors="coerce").fillna(0).clip(lower=0); n+=len(q); n_gdp+=int(q["GDP_per"].isna().sum()); n_min+=int(q["Minority_rate"].isna().sum()); pop_total+=float(pop.sum()); pop_gdp_missing+=float(pop[q["GDP_per"].isna()].sum()); pop_min_missing+=float(pop[q["Minority_rate"].isna()].sum())
                validate_complete_province_parts(
                    discover_parts(out_dir),
                    f"3_1 output {year}/{scope}/{profile}",
                )
                coverage={"year":year,"scope":scope,"profile":profile,"gdp_sheet":gdp_sheet,"n_rows":n,"population":pop_total,"gdp_missing_rows":n_gdp,"gdp_missing_row_pct":n_gdp/n if n else np.nan,"gdp_missing_population":pop_gdp_missing,"gdp_missing_population_pct":pop_gdp_missing/pop_total if pop_total else np.nan,"minority_missing_rows":n_min,"minority_missing_row_pct":n_min/n if n else np.nan,"minority_missing_population":pop_min_missing,"minority_missing_population_pct":pop_min_missing/pop_total if pop_total else np.nan}
                pd.DataFrame([coverage]).to_csv(stage_dir/"coverage_qc.csv",index=False,encoding="utf-8-sig")
                print(f"3_1 done {year}: rows={n:,}, GDP_missing_pop={coverage['gdp_missing_population_pct']:.3%}, Minority_missing_pop={coverage['minority_missing_population_pct']:.3%}" if n else f"3_1 done {year}: no rows")

if __name__=="__main__": main()
