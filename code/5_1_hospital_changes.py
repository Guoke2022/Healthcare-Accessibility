# -*- coding: utf-8 -*-

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from config import (
    BASE_YEAR, END_YEAR, CHANGE_YEARS, HOSPITAL_DATA_DIR, COUNTY_SHP,
    HOSPITAL_CHANGES_ROOT, MUNICIPALITIES, ALLOW_FUZZY_HOSPITAL_MATCH, HOSPITAL_FUZZY_MATCH_MAX_KM,
    HOSPITAL_NAME_SIMILARITY_MIN, HOSPITAL_EXACT_MATCH_REVIEW_KM,
)
from utils.extended_analysis import ensure_exists, read_csv_robust, read_stats, norm6

META_COLS = ["name", "grade", "type", "province", "region", "area", "construction_time", "3A_year", "lng", "lat"]
EARTH_RADIUS_KM = 6371.0


def normalize_hospital_name(x) -> str:
    if pd.isna(x): return ""
    s = unicodedata.normalize("NFKC", str(x)).lower().strip()
    return re.sub(r"[\s·•・,，。.;；:：()（）\[\]【】{}<>《》'\"“”‘’_-]+", "", s)


def haversine_km(lon1, lat1, lon2, lat2) -> float:
    if not all(np.isfinite([lon1, lat1, lon2, lat2])): return np.nan
    a1, a2 = np.radians(lat1), np.radians(lat2); dlat = a2-a1; dlon = np.radians(lon2-lon1)
    a = np.sin(dlat/2)**2 + np.cos(a1)*np.cos(a2)*np.sin(dlon/2)**2
    return float(2*EARTH_RADIUS_KM*np.arcsin(np.sqrt(np.clip(a, 0, 1))))


def load_hospitals(year: int) -> pd.DataFrame:
    path = HOSPITAL_DATA_DIR / f"{year}.csv"; ensure_exists(path, f"{year} 医院表")
    df = read_csv_robust(path).copy(); required = {"name", "beds", "lng", "lat"}; missing = required-set(df.columns)
    if missing: raise KeyError(f"{path} 缺少字段：{sorted(missing)}")
    df["name"] = df["name"].astype("string").str.strip(); df["name_norm"] = df["name"].map(normalize_hospital_name)
    df["province_norm"] = df["province"].astype("string").str.strip().fillna("") if "province" in df.columns else ""
    for c in ["beds", "lng", "lat"]: df[c] = pd.to_numeric(df[c], errors="coerce")
    bad_beds = ~np.isfinite(df["beds"].to_numpy(dtype=float))
    if bad_beds.any():
        sample_cols = [c for c in ["name", "province", "region", "beds"] if c in df.columns]
        sample = df.loc[bad_beds, sample_cols].head(10).to_dict("records")
        raise ValueError(
            f"{path} contains {int(bad_beds.sum())} hospital rows with missing/non-finite beds. "
            "SEE/CIE requires complete annual bed capacity; refusing to convert unknown beds "
            f"to unchanged/zero. Sample={sample}"
        )
    for c in META_COLS:
        if c not in df.columns: df[c] = pd.NA
    df["_row_id"] = np.arange(len(df), dtype=np.int64)
    dup = int(df["name_norm"].duplicated(keep=False).sum())
    if dup: print(f"QC {year}: {dup} rows share a normalized hospital name; spatial one-to-one matching will be used.")
    return df


def _pair_exact_name(prev: pd.DataFrame, curr: pd.DataFrame):
    pairs, used_p, used_c = [], set(), set()
    prev_keys = set(zip(prev["name_norm"], prev["province_norm"])); curr_keys = set(zip(curr["name_norm"], curr["province_norm"]))
    common = sorted((prev_keys & curr_keys) - {("", "")})
    for name, province in common:
        pidx = prev.index[prev["name_norm"].eq(name) & prev["province_norm"].eq(province)].tolist(); cidx = curr.index[curr["name_norm"].eq(name) & curr["province_norm"].eq(province)].tolist()
        if len(pidx) == len(cidx) == 1:
            cand = [(pidx[0], cidx[0])]
        else:
            cost = np.full((len(pidx), len(cidx)), 1e9, dtype=float)
            for i, pi in enumerate(pidx):
                for j, ci in enumerate(cidx):
                    d = haversine_km(prev.at[pi,"lng"], prev.at[pi,"lat"], curr.at[ci,"lng"], curr.at[ci,"lat"])
                    cost[i,j] = d if np.isfinite(d) else 1e6 + abs(i-j)
            rr, cc = linear_sum_assignment(cost); cand = [(pidx[i], cidx[j]) for i,j in zip(rr,cc)]
        for pi, ci in cand:
            d = haversine_km(prev.at[pi,"lng"], prev.at[pi,"lat"], curr.at[ci,"lng"], curr.at[ci,"lat"])
            pairs.append((pi, ci, "exact_name", d, 1.0)); used_p.add(pi); used_c.add(ci)
    return pairs, used_p, used_c


def _xyz(df: pd.DataFrame) -> np.ndarray:
    lon = np.radians(pd.to_numeric(df["lng"], errors="coerce").to_numpy(float)); lat = np.radians(pd.to_numeric(df["lat"], errors="coerce").to_numpy(float))
    c = np.cos(lat); return np.column_stack([c*np.cos(lon), c*np.sin(lon), np.sin(lat)])


def _pair_fuzzy(prev: pd.DataFrame, curr: pd.DataFrame, used_p: set, used_c: set):
    pidx = [i for i in prev.index if i not in used_p and np.isfinite(prev.at[i,"lng"]) and np.isfinite(prev.at[i,"lat"])]
    cidx = [i for i in curr.index if i not in used_c and np.isfinite(curr.at[i,"lng"]) and np.isfinite(curr.at[i,"lat"])]
    if not pidx or not cidx: return []
    p = prev.loc[pidx]; c = curr.loc[cidx]; tree = cKDTree(_xyz(p)); chord = 2*np.sin((HOSPITAL_FUZZY_MATCH_MAX_KM/EARTH_RADIUS_KM)/2)
    candidates = []
    for j, neigh in enumerate(tree.query_ball_point(_xyz(c), r=chord)):
        ci = cidx[j]
        for ii in neigh:
            pi = pidx[ii]

            pp, cp = prev.at[pi,"province"], curr.at[ci,"province"]
            if pd.notna(pp) and pd.notna(cp) and str(pp).strip() != str(cp).strip(): continue
            sim = SequenceMatcher(None, prev.at[pi,"name_norm"], curr.at[ci,"name_norm"]).ratio()
            if sim < HOSPITAL_NAME_SIMILARITY_MIN: continue
            d = haversine_km(prev.at[pi,"lng"], prev.at[pi,"lat"], curr.at[ci,"lng"], curr.at[ci,"lat"])
            score = (1-sim) + d/max(HOSPITAL_FUZZY_MATCH_MAX_KM, 1e-9)
            candidates.append((score, -sim, d, pi, ci))
    out=[]; up=set(); uc=set()
    for _, negsim, d, pi, ci in sorted(candidates):
        if pi in up or ci in uc: continue
        sim=-negsim; out.append((pi, ci, "fuzzy_spatial_name", d, sim)); up.add(pi); uc.add(ci)
    return out


def compute_hospital_changes(prev: pd.DataFrame, curr: pd.DataFrame, prev_year: int, curr_year: int) -> pd.DataFrame:
    bp, bc = f"beds_{prev_year}", f"beds_{curr_year}"; cols = META_COLS + ["name_prev", bp, bc, "beds_added", "change_type", "match_method", "match_distance_km", "name_similarity"]
    pairs, used_p, used_c = _pair_exact_name(prev, curr)
    fuzzy = _pair_fuzzy(prev, curr, used_p, used_c) if ALLOW_FUZZY_HOSPITAL_MATCH else []
    pairs.extend(fuzzy); used_p.update(x[0] for x in fuzzy); used_c.update(x[1] for x in fuzzy)
    rows=[]
    for pi, ci, method, dist, sim in pairs:
        r = {c: curr.at[ci,c] for c in META_COLS}; r.update({"name_prev":prev.at[pi,"name"], bp:prev.at[pi,"beds"], bc:curr.at[ci,"beds"], "match_method":method, "match_distance_km":dist, "name_similarity":sim})
        r["beds_added"] = r[bc] - r[bp] if pd.notna(r[bc]) and pd.notna(r[bp]) else np.nan
        if pd.isna(r["beds_added"]):
            r["change_type"] = "unknown_bed_change"
        elif r["beds_added"] > 0:
            r["change_type"] = "increase"
        elif r["beds_added"] < 0:
            r["change_type"] = "decrease"
        else:
            r["change_type"] = "unchanged"
        rows.append(r)
    for ci in curr.index:
        if ci in used_c: continue
        r={c:curr.at[ci,c] for c in META_COLS}; r.update({"name_prev":pd.NA,bp:0.0,bc:curr.at[ci,"beds"],"beds_added":curr.at[ci,"beds"],"change_type":"new","match_method":"unmatched_new","match_distance_km":np.nan,"name_similarity":np.nan}); rows.append(r)
    for pi in prev.index:
        if pi in used_p: continue
        r={c:prev.at[pi,c] for c in META_COLS}; r.update({"name_prev":prev.at[pi,"name"],bp:prev.at[pi,"beds"],bc:0.0,"beds_added":-prev.at[pi,"beds"] if pd.notna(prev.at[pi,"beds"]) else np.nan,"change_type":"closed","match_method":"unmatched_closed","match_distance_km":np.nan,"name_similarity":np.nan}); rows.append(r)
    out=pd.DataFrame(rows, columns=cols)
    bad_added = ~np.isfinite(pd.to_numeric(out["beds_added"], errors="coerce").to_numpy(dtype=float))
    if bad_added.any():
        sample = out.loc[bad_added, ["name", "name_prev", bp, bc, "change_type", "match_method"]].head(10).to_dict("records")
        raise ValueError(
            f"{prev_year}-{curr_year} produced {int(bad_added.sum())} non-finite beds_added values. "
            f"This violates the complete-bed invariant. Sample={sample}"
        )
    out["match_review_required"] = (
        out["match_method"].eq("exact_name")
        & pd.to_numeric(out["match_distance_km"], errors="coerce").gt(HOSPITAL_EXACT_MATCH_REVIEW_KM)
    )
    print(f"matching {prev_year}-{curr_year}: exact={sum(out['match_method'].eq('exact_name'))}, fuzzy={sum(out['match_method'].eq('fuzzy_spatial_name'))}, new={sum(out['change_type'].eq('new'))}, closed={sum(out['change_type'].eq('closed'))}, review={int(out['match_review_required'].sum())}")
    return out


def load_admin():
    try: import geopandas as gpd
    except ImportError as e: raise ImportError("5_1 需要 geopandas 做医院点-县级行政区匹配") from e
    ensure_exists(COUNTY_SHP, "县级行政区 shp"); gdf=gpd.read_file(COUNTY_SHP); required={"省级","地级","县级","县级码","geometry"}; missing=required-set(gdf.columns)
    if missing: raise KeyError(f"{COUNTY_SHP} 缺少字段：{sorted(missing)}")
    keep=[c for c in ["省级","地级","县级","县级码","县级类","geometry"] if c in gdf.columns]; gdf=gdf[keep].copy()
    if gdf.crs is None: raise ValueError(f"行政区 shp 无 CRS：{COUNTY_SHP}")
    gdf=gdf.to_crs("EPSG:4326"); gdf["县级码"]=gdf["县级码"].map(norm6); return gdf


def infer_county_type(name: pd.Series) -> pd.Series:
    s=name.astype("string")
    return pd.Series(np.select([s.str.endswith("区",na=False),s.str.endswith("市",na=False),s.str.endswith("旗",na=False)],["市辖区","县级市","县"],default="县"),index=s.index,dtype="string")


def spatial_enrich(changes: pd.DataFrame, admin) -> pd.DataFrame:
    import geopandas as gpd
    pts=changes[pd.to_numeric(changes["lng"],errors="coerce").notna() & pd.to_numeric(changes["lat"],errors="coerce").notna()].copy(); bad=changes.drop(index=pts.index).copy()
    g=gpd.GeoDataFrame(pts,geometry=gpd.points_from_xy(pts["lng"],pts["lat"]),crs="EPSG:4326"); j=gpd.sjoin(g,admin,how="left",predicate="intersects").drop(columns=["geometry","index_right"],errors="ignore")

    if j.index.duplicated().any(): j=j[~j.index.duplicated(keep="first")]
    if len(bad):
        for c in ["省级","地级","县级","县级码","县级类"]:
            if c not in bad.columns: bad[c]=pd.NA
        j=pd.concat([pd.DataFrame(j),bad],ignore_index=True,sort=False)
    else: j=pd.DataFrame(j)
    if "县级类" not in j.columns: j["县级类"]=infer_county_type(j["县级"])
    else:
        j["县级类"]=j["县级类"].astype("string").replace({"不统计":"市辖区","旗":"县"}); j["县级类"]=j["县级类"].fillna(infer_county_type(j["县级"]))
    j["县级码"]=j["县级码"].map(norm6); j["city_name_norm"]=j["地级"].astype("string").str.strip()
    m=j["city_name_norm"].eq("不统计")|j["city_name_norm"].isna(); j.loc[m,"city_name_norm"]=j.loc[m,"省级"].astype("string")
    m2=j["city_name_norm"].isin(["海南省","湖北省"]); j.loc[m2,"city_name_norm"]=j.loc[m2,"县级"].astype("string"); return j


def _population_for_year(df: pd.DataFrame, pop_year: int, suffix: str) -> pd.DataFrame:
    county=read_stats("accessibility","county"); city=read_stats("accessibility","city"); prov=read_stats("accessibility","provincial")
    county=county[county["Year"].eq(pop_year)][["县级码","pop_num"]].rename(columns={"pop_num":f"county_pop_{suffix}"})
    city=city[city["Year"].eq(pop_year)][["地级","pop_num"]].rename(columns={"pop_num":f"city_pop_{suffix}"})
    prov=prov[prov["Year"].eq(pop_year)][["省级","pop_num"]].rename(columns={"pop_num":f"province_pop_{suffix}"})
    out=df.merge(county,on="县级码",how="left").merge(city,left_on="city_name_norm",right_on="地级",how="left").merge(prov,on="省级",how="left")
    c=f"city_pop_{suffix}"; p=f"province_pop_{suffix}"; q=f"county_pop_{suffix}"
    m=out[c].isna() & out["省级"].isin(MUNICIPALITIES); out.loc[m,c]=out.loc[m,p]; m=out[c].isna(); out.loc[m,c]=out.loc[m,q]
    return out.drop(columns=["地级",p],errors="ignore")


def add_transition_population(df: pd.DataFrame, prev_year: int, curr_year: int) -> pd.DataFrame:
    out=_population_for_year(df,prev_year,"prev"); out=_population_for_year(out,curr_year,"curr")
    out["county_pop_transition_mean"]=out[["county_pop_prev","county_pop_curr"]].mean(axis=1,skipna=False)
    out["city_pop_transition_mean"]=out[["city_pop_prev","city_pop_curr"]].mean(axis=1,skipna=False)

    out["county_pop"]=out["county_pop_transition_mean"]; out["city_pop"]=out["city_pop_transition_mean"]
    out["population_denominator_years"]=f"{prev_year}-{curr_year}_mean"; return out


def main():
    HOSPITAL_CHANGES_ROOT.mkdir(parents=True,exist_ok=True); annual_dir=HOSPITAL_CHANGES_ROOT/"annual"; annual_dir.mkdir(parents=True,exist_ok=True); admin=load_admin()
    overall=spatial_enrich(compute_hospital_changes(load_hospitals(BASE_YEAR),load_hospitals(END_YEAR),BASE_YEAR,END_YEAR),admin)
    overall.to_csv(HOSPITAL_CHANGES_ROOT/f"hosps_changed_type_{BASE_YEAR}_{END_YEAR}.csv",index=False,encoding="utf-8-sig"); overall.to_csv(HOSPITAL_CHANGES_ROOT/"hosps_changed_type.csv",index=False,encoding="utf-8-sig")
    overall.loc[overall["match_review_required"] | overall["match_method"].eq("fuzzy_spatial_name")].to_csv(HOSPITAL_CHANGES_ROOT/f"match_qc_{BASE_YEAR}_{END_YEAR}.csv",index=False,encoding="utf-8-sig")
    print(f"5_1 overall change: {overall['change_type'].value_counts().to_dict()}")
    summary=[]
    for year in CHANGE_YEARS:
        prev_year=year-1; ch=compute_hospital_changes(load_hospitals(prev_year),load_hospitals(year),prev_year,year); ch=add_transition_population(spatial_enrich(ch,admin),prev_year,year)
        ch.to_csv(annual_dir/f"hosps_changed_type_{year}.csv",index=False,encoding="utf-8-sig"); ch.loc[ch["match_review_required"] | ch["match_method"].eq("fuzzy_spatial_name")].to_csv(annual_dir/f"match_qc_{year}.csv",index=False,encoding="utf-8-sig"); counts=ch["change_type"].value_counts().to_dict(); summary.append({"year":year,**{k:counts.get(k,0) for k in ["new","increase","decrease","unchanged","closed"]}}); print(f"5_1 {prev_year}-{year}: {counts}")
    pd.DataFrame(summary).to_csv(HOSPITAL_CHANGES_ROOT/"annual_change_counts.csv",index=False,encoding="utf-8-sig")


if __name__ == "__main__": main()
