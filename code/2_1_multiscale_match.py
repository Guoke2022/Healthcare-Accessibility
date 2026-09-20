# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from config import RUN_POLICY
from utils.cache import build_fingerprint, prepare_stage_directory

import numpy as np
import pandas as pd
import geopandas as gpd

from config import (
    YEARS,
    SERVICE_SCOPES,
    PROFILES,
    ADMIN_SHP,
    ADMIN_COLS,
    MULTISCALE_ROOT,
    EASTERN_REGION,
    NORTHERN,
    ENABLE_CITY_LEVEL,
    ENABLE_POOR_COUNTY,
    POOR_COUNTY_XLSX,
    ENABLE_URBAN_RURAL,
    YEAR_TO_GURS,
    APPLY_LEGACY_POSTPROCESS,
    LEGACY_IQR_THRESHOLD,
    LEGACY_IDW_K,
    LEGACY_IDW_POWER,
    WRITE_COMPAT_NATIONAL_CSV,
    PREPARED_ROOT,
    MULTISCALE_YEAR_WORKERS,
    ONLY_PROVINCES,
    get_run_year,
    is_year_worker,
    run_script_for_all_years,
)
from utils.multiscale import (
    accessibility_grid_dir,
    matched_parts_dir,
    discover_parts,
    read_new_grid_part,
    parquet_columns,
    validate_complete_province_parts,
)

REQUIRED_ACCESSIBILITY_COLS = {
    "grid_id",
    "population",
    "lon",
    "lat",
    "accessibility",
    "nearest_hospital_time_min",
}

GRID_ADMIN_LOOKUP_ROOT = MULTISCALE_ROOT / "_grid_admin_lookup"
LOOKUP_READY_ENV = "NC_2_1_GRID_ADMIN_LOOKUP_READY"


def _landscan_signature(year: int) -> tuple[str, Path, dict]:

    meta_path = PREPARED_ROOT / str(year) / "metadata" / "landscan_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"缺少 LandScan metadata：{meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    key_obj = {k: meta.get(k) for k in ["width", "height", "crs", "transform"]}
    if any(key_obj[k] is None for k in key_obj):
        raise KeyError(f"LandScan metadata 缺 width/height/crs/transform：{meta_path}")
    raw = json.dumps(key_obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    sig = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return sig, meta_path, key_obj


def _province_table() -> pd.DataFrame:
    p = PREPARED_ROOT / "common" / "provinces.parquet"
    if not p.exists():
        raise FileNotFoundError(f"缺少省级基准：{p}")
    df = pd.read_parquet(p, columns=["province_id", "province_name"]).sort_values("province_id")
    if ONLY_PROVINCES:
        wanted = {str(x) for x in ONLY_PROVINCES}
        df = df[df["province_name"].astype(str).isin(wanted)].copy()
        missing = sorted(wanted - set(df["province_name"].astype(str)))
        if missing:
            raise ValueError(f"ONLY_PROVINCES 中存在未知省份：{missing}")
    return df


def _population_part(year: int, pid: int, pname: str) -> Path:
    p = PREPARED_ROOT / str(year) / "population_parts" / f"province_{pid:03d}_{pname}.parquet"
    if p.exists():
        return p
    cands = sorted((PREPARED_ROOT / str(year) / "population_parts").glob(f"province_{pid:03d}_*.parquet"))
    if len(cands) == 1:
        return cands[0]
    raise FileNotFoundError(f"找不到唯一 population part：year={year}, province={pid}/{pname}")


def _lookup_path(signature: str, pid: int, pname: str) -> Path:
    return GRID_ADMIN_LOOKUP_ROOT / signature / "parts" / f"province_{pid:03d}_{pname}.parquet"


def _apply_static_admin_labels(joined: pd.DataFrame, poor_counties: set[str]) -> pd.DataFrame:
    joined = joined.copy()
    joined = joined[~joined["省级"].isin(["澳门特别行政区", "香港特别行政区"])].copy()
    joined["Eastern_Region"] = np.where(joined["省级"].isin(EASTERN_REGION), "Eastern", "NotEastern")
    joined["Coastal_Inland"] = np.where(joined["省级"].isin(EASTERN_REGION), "Coastal", "Inland")
    joined["North_South"] = np.where(joined["省级"].isin(NORTHERN), "Northern", "Southern")
    if ENABLE_POOR_COUNTY:
        joined["Poor_County"] = np.where(
            joined["县级码"].astype("string").isin(poor_counties), "PoorCounty", "NotPoor"
        )
    joined = add_city_level(joined)
    return joined


def build_grid_admin_lookups(admin_all: gpd.GeoDataFrame, poor_counties: set[str]) -> dict[int, Path]:

    groups: dict[str, list[int]] = {}
    meta_paths: dict[str, list[Path]] = {}
    for year in YEARS:
        sig, meta_path, _ = _landscan_signature(year)
        groups.setdefault(sig, []).append(year)
        meta_paths.setdefault(sig, []).append(meta_path)

    provinces = _province_table()
    admin_inputs = sorted(ADMIN_SHP.parent.glob(ADMIN_SHP.stem + ".*")) or [ADMIN_SHP]
    year_lookup: dict[int, Path] = {}

    print(f"\n[2_1 lookup] LandScan grid signatures={len(groups)}")
    for sig, years in groups.items():
        sig_root = GRID_ADMIN_LOOKUP_ROOT / sig
        pop_inputs = []
        for y in years:
            for r in provinces.itertuples(index=False):
                pop_inputs.append(_population_part(y, int(r.province_id), str(r.province_name)))
        fp = build_fingerprint(
            config={
                "stage": "2_1_grid_admin_lookup",
                "signature": sig,
                "years": years,
                "admin_cols": ADMIN_COLS,
                "enable_city_level": ENABLE_CITY_LEVEL,
                "enable_poor_county": ENABLE_POOR_COUNTY,
            },
            files=[*meta_paths[sig], *pop_inputs, *admin_inputs, Path(__file__).resolve(), Path(__file__).resolve().parent / "utils" / "city_levels.py"],
        )
        status = prepare_stage_directory(
            sig_root, fp, run_policy=RUN_POLICY,
            payload={"stage": "2_1_grid_admin_lookup", "signature": sig, "years": years},
        )
        print(f"[2_1 lookup] signature={sig} years={years[0]}-{years[-1]} status={status}")

        for r in provinces.itertuples(index=False):
            pid, pname = int(r.province_id), str(r.province_name)
            out = _lookup_path(sig, pid, pname)
            if status == "reused_same_input" and out.exists():
                continue

            frames = []
            for y in years:
                part = _population_part(y, pid, pname)
                z = pd.read_parquet(part, columns=["grid_id", "lon", "lat"])
                frames.append(z)
            coords = pd.concat(frames, ignore_index=True)
            coords["grid_id"] = pd.to_numeric(coords["grid_id"], errors="raise").astype("int64")
            coord_unique = coords.drop_duplicates(["grid_id", "lon", "lat"])
            if coord_unique["grid_id"].duplicated().any():
                bad = coord_unique.loc[coord_unique["grid_id"].duplicated(keep=False), "grid_id"].head(10).tolist()
                raise RuntimeError(
                    f"同一 LandScan signature 内相同 grid_id 出现不同坐标：signature={sig}, "
                    f"province={pname}, examples={bad}"
                )
            coords = coord_unique.sort_values("grid_id").drop_duplicates("grid_id", keep="first")

            gdf = gpd.GeoDataFrame(
                coords, geometry=gpd.points_from_xy(coords["lon"], coords["lat"]), crs="EPSG:4326"
            )
            admin = admin_all[admin_all["省级"].astype(str).eq(pname)].copy()
            if admin.empty:
                admin = admin_all
            joined = gpd.sjoin(gdf, admin, how="left", predicate="intersects")
            if joined["grid_id"].duplicated().any():
                joined = joined.sort_values(["grid_id", "县级码"], na_position="last").drop_duplicates("grid_id", keep="first")
            joined = joined.drop(columns=["index_right", "geometry", "lon", "lat"], errors="ignore")
            joined = _apply_static_admin_labels(joined, poor_counties)

            keep = [
                "grid_id", "县级", "县级码", "地级", "地级码", "省级", "省级码",
                "Eastern_Region", "Coastal_Inland", "North_South", "city_level", "city_name_norm",
            ]
            if ENABLE_POOR_COUNTY:
                keep.append("Poor_County")
            keep = [c for c in keep if c in joined.columns]
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(joined[keep]).to_parquet(out, index=False, compression="zstd")
            print(f"  lookup {pname}: {len(joined):,} unique grids")

        for y in years:
            year_lookup[y] = sig_root

    return year_lookup


def validate_accessibility_outputs() -> None:

    problems = []

    print("\n[2_1-1] 检查 1_2 输出完整性")
    for scope in SERVICE_SCOPES:
        for profile in PROFILES:
            for year in YEARS:
                folder = accessibility_grid_dir(year, scope, profile)
                parts = discover_parts(folder)

                if not parts:
                    problems.append(f"{year}/{scope}/{profile}: 无省级 Parquet -> {folder}")
                    continue

                try:
                    validate_complete_province_parts(
                        parts,
                        f"1_2 {year}/{scope}/{profile}",
                    )
                except Exception as e:
                    problems.append(str(e))
                    continue


                schema_problems = []
                for part in parts:
                    cols = set(parquet_columns(part))
                    missing = sorted(REQUIRED_ACCESSIBILITY_COLS - cols)
                    if missing:
                        schema_problems.append(f"{part.name} 缺字段 {missing}")
                if schema_problems:
                    problems.append(
                        f"{year}/{scope}/{profile}: " + "; ".join(schema_problems[:10])
                    )
                    continue

                print(
                    f"OK {year} | {scope} | {profile} | "
                    f"province parts={len(parts)} | province set complete"
                )

    if problems:
        text = "\n".join(f"- {x}" for x in problems)
        raise RuntimeError(
            "1_2 输出不完整，2_1 暂停。先把 2014~2024 缺失年份补齐：\n" + text
        )


def add_urban_rural(df: pd.DataFrame, raster) -> pd.DataFrame:

    if raster is None:
        return df

    import rasterio

    def one(lon, lat):
        try:
            row, col = raster.index(float(lon), float(lat))
            window = raster.read(
                1,
                window=rasterio.windows.Window(col - 5, row - 5, 10, 10),
                boundless=True,
                fill_value=raster.nodata,
            )
            if raster.nodata is not None:
                window = window[window != raster.nodata]
            window = window[np.isfinite(window)]
            if len(window) == 0:
                return "Unknown"
            unique, counts = np.unique(window, return_counts=True)
            majority = unique[np.argmax(counts)]
            return "Urban" if majority == 1 else "Rural" if majority == 2 else "Unknown"
        except Exception:
            return "Unknown"

    out = df.copy()
    out["Urban_Rural"] = [
        one(lon, lat) for lon, lat in zip(out["lon"], out["lat"])
    ]
    return out


def load_admin() -> gpd.GeoDataFrame:
    if not ADMIN_SHP.exists():
        raise FileNotFoundError(f"行政区划文件不存在：{ADMIN_SHP}")

    admin = gpd.read_file(ADMIN_SHP)
    missing = [c for c in ADMIN_COLS if c not in admin.columns]
    if missing:
        raise KeyError(f"行政区划缺字段 {missing}；实际字段={list(admin.columns)}")
    if admin.crs is None:
        raise ValueError("行政区划没有 CRS。")

    admin = admin[ADMIN_COLS + ["geometry"]].copy().to_crs("EPSG:4326")
    for c in ["县级码", "地级码", "省级码"]:
        admin[c] = admin[c].astype("string").str.replace(r"\.0$", "", regex=True)
    if "县级码" in admin:
        admin["县级码"] = admin["县级码"].str.zfill(6)
    return admin


def legacy_postprocess(df: pd.DataFrame) -> pd.DataFrame:

    if not APPLY_LEGACY_POSTPROCESS or len(df) == 0:
        return df

    from scipy.spatial import cKDTree

    out = df.copy()


    out["travel_time"] = pd.to_numeric(
        out["travel_time"], errors="coerce"
    ).astype("float64")
    out["acc"] = pd.to_numeric(
        out["acc"], errors="coerce"
    ).astype("float64")

    tt = out["travel_time"]
    acc = out["acc"]

    finite_tt = tt[np.isfinite(tt)]
    if len(finite_tt) > 0:
        q1 = finite_tt.quantile(0.25)
        q3 = finite_tt.quantile(0.75)
        iqr = q3 - q1
        upper = q3 + LEGACY_IQR_THRESHOLD * iqr
        out.loc[np.isfinite(tt) & (tt > upper), "travel_time"] = upper

    tt = pd.to_numeric(out["travel_time"], errors="coerce")
    valid = np.isfinite(tt) & (tt >= 1.0)
    invalid = ~valid

    if invalid.any() and valid.any():
        xy_valid = out.loc[valid, ["lon", "lat"]].to_numpy(dtype=np.float64)
        xy_invalid = out.loc[invalid, ["lon", "lat"]].to_numpy(dtype=np.float64)

        tree = cKDTree(xy_valid)
        k = min(int(LEGACY_IDW_K), len(xy_valid))
        distances, indices = tree.query(xy_invalid, k=k)


        if k == 1:
            distances = distances[:, None]
            indices = indices[:, None]

        weights = 1.0 / (np.power(distances, LEGACY_IDW_POWER) + 1e-9)
        weights = weights / weights.sum(axis=1, keepdims=True)

        for col in ["travel_time", "acc"]:
            vals = pd.to_numeric(out.loc[valid, col], errors="coerce").to_numpy(dtype=np.float64)

            neighbor_vals = vals[indices]
            finite_neighbor = np.isfinite(neighbor_vals)
            local_w = weights * finite_neighbor
            denom = local_w.sum(axis=1)
            numer = np.nansum(neighbor_vals * local_w, axis=1)
            filled = np.divide(
                numer,
                denom,
                out=np.full_like(numer, np.nan, dtype=np.float64),
                where=denom > 0,
            )
            out.loc[invalid, col] = np.asarray(filled, dtype=np.float64)

    return out


def add_city_level(df: pd.DataFrame) -> pd.DataFrame:
    if not ENABLE_CITY_LEVEL:
        return df

    try:
        from utils.city_levels import (
            Megacity_list,
            Supercity_list,
            Type_I_Large_Cities_list,
            Type_II_Large_Cities_list,
            Medium_sized_cites_list,
        )
    except ImportError:
        print(
            "Warning: 未找到 utils.city_levels，city_level 将留空。"
            "请把你原项目中的 utils/city_levels.py 放回 PYTHONPATH。"
        )
        df["city_level"] = pd.NA
        return df

    out = df.copy()
    for col in ["县级", "地级", "省级"]:
        out[col] = out[col].astype("string").str.strip()

    out["city_key"] = out["地级"].copy()
    mask = out["city_key"].eq("不统计") | out["city_key"].isna()
    out.loc[mask, "city_key"] = out.loc[mask, "省级"]

    mask2 = out["city_key"].isin(["海南省", "湖北省"])
    out.loc[mask2, "city_key"] = out.loc[mask2, "县级"]

    mask3 = out["city_key"].isna()
    out.loc[mask3, "city_key"] = out.loc[mask3, "县级"]

    city_level_map = {
        **{c: "Mega City" for c in Megacity_list},
        **{c: "Super City" for c in Supercity_list},
        **{c: "Type I Large City" for c in Type_I_Large_Cities_list},
        **{c: "Type II Large City" for c in Type_II_Large_Cities_list},
        **{c: "Medium-sized City" for c in Medium_sized_cites_list},
    }
    out["city_level"] = out["city_key"].map(city_level_map)
    small = out["city_level"].isna() & out["city_key"].notna()
    out.loc[small, "city_level"] = "Small City"
    out["city_name_norm"] = out["city_key"]
    return out


def load_poor_counties() -> set[str]:
    if not ENABLE_POOR_COUNTY:
        return set()
    if not POOR_COUNTY_XLSX.exists():
        raise FileNotFoundError(f"贫困县名单不存在：{POOR_COUNTY_XLSX}")
    x = pd.read_excel(POOR_COUNTY_XLSX)
    if "行政区划代码" not in x.columns:
        raise KeyError("贫困县 Excel 缺少“行政区划代码”")
    return set(
        x["行政区划代码"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(6)
    )


def match_one_part(
    src_path: Path,
    lookup_path: Path,
) -> pd.DataFrame:

    df = read_new_grid_part(src_path)
    df = legacy_postprocess(df)
    if not lookup_path.exists():
        raise FileNotFoundError(f"缺少 grid-admin lookup：{lookup_path}")
    lookup = pd.read_parquet(lookup_path)
    if lookup["grid_id"].duplicated().any():
        raise RuntimeError(f"grid-admin lookup 存在重复 grid_id：{lookup_path}")

    joined = df.merge(lookup, on="grid_id", how="left", validate="many_to_one", indicator=True)
    n_missing = int((joined["_merge"] != "both").sum())
    if n_missing:
        sample = joined.loc[joined["_merge"] != "both", "grid_id"].head(10).tolist()
        raise RuntimeError(
            f"{src_path.name}: {n_missing:,} grids 未命中 grid-admin lookup；sample={sample}. "
            "请删除/重建 result/2_1_multiscale_matched/_grid_admin_lookup 后重跑。"
        )
    joined = joined.drop(columns=["_merge"])

    keep = [
        "grid_id", "pop", "acc", "travel_time", "lon", "lat",
        "n_reachable_hospitals", "grid_snap_distance_km", "grid_snap_kind",
        "县级", "县级码", "地级", "地级码", "省级", "省级码",
        "Eastern_Region", "Coastal_Inland", "North_South",
        "city_level", "city_name_norm",
    ]
    if ENABLE_POOR_COUNTY:
        keep.append("Poor_County")
    keep = [c for c in keep if c in joined.columns]
    return pd.DataFrame(joined[keep].copy())


def process_year(year: int) -> None:
    sig, _, _ = _landscan_signature(year)


    if os.environ.get(LOOKUP_READY_ENV, "0") != "1":
        admin = load_admin()
        poor_counties = load_poor_counties()
        build_grid_admin_lookups(admin, poor_counties)

    for scope in SERVICE_SCOPES:
        for profile in PROFILES:
            src_dir = accessibility_grid_dir(year, scope, profile)
            src_parts = discover_parts(src_dir)

            stage_dir = MULTISCALE_ROOT / str(year) / scope / profile
            lookup_inputs = []
            for src_path in src_parts:
                try:
                    pid = int(src_path.name.split("_", 2)[1])
                except Exception as e:
                    raise ValueError(f"无法从省级文件名解析 province_id：{src_path.name}") from e
                pname = src_path.stem.split("_", 2)[2]
                lookup_inputs.append(_lookup_path(sig, pid, pname))

            extra_inputs = []
            if ENABLE_URBAN_RURAL and year in YEAR_TO_GURS and YEAR_TO_GURS[year].exists():
                extra_inputs.append(YEAR_TO_GURS[year])

            stage_fingerprint = build_fingerprint(
                config={
                    "stage": "2_1_multiscale_match", "year": year, "scope": scope, "profile": profile,
                    "landscan_signature": sig,
                    "apply_legacy_postprocess": APPLY_LEGACY_POSTPROCESS,
                    "legacy_iqr_threshold": LEGACY_IQR_THRESHOLD, "legacy_idw_k": LEGACY_IDW_K,
                    "legacy_idw_power": LEGACY_IDW_POWER, "enable_city_level": ENABLE_CITY_LEVEL,
                    "enable_poor_county": ENABLE_POOR_COUNTY, "enable_urban_rural": ENABLE_URBAN_RURAL,
                },
                files=[*src_parts, *lookup_inputs, *extra_inputs, Path(__file__).resolve(), Path(__file__).resolve().parent / "utils" / "multiscale.py"],
            )
            stage_status = prepare_stage_directory(
                stage_dir, stage_fingerprint, run_policy=RUN_POLICY,
                payload={"stage": "2_1_multiscale_match", "year": year},
            )
            print(f"2_1 stage dir: {stage_status} -> {stage_dir}")

            out_dir = matched_parts_dir(year, scope, profile)
            out_dir.mkdir(parents=True, exist_ok=True)
            compat_frames = []

            raster = None
            if ENABLE_URBAN_RURAL and year in YEAR_TO_GURS:
                gurs_path = YEAR_TO_GURS[year]
                if gurs_path.exists():
                    import rasterio
                    raster = rasterio.open(gurs_path)
                else:
                    print(f"Warning: GURS 不存在，{year} 跳过城乡标签：{gurs_path}")

            print(f"\n=== 2_1 match | {year} | scope={scope} | profile={profile} | parts={len(src_parts)} | lookup={sig} ===")
            try:
                for i, src_path in enumerate(src_parts, 1):
                    out_path = out_dir / src_path.name
                    if stage_status == "reused_same_input" and out_path.exists():
                        print(f"[{i}/{len(src_parts)}] {src_path.name}: 输入未变且匹配分片已存在，跳过")
                        if WRITE_COMPAT_NATIONAL_CSV:
                            compat_frames.append(pd.read_parquet(out_path))
                        continue

                    pid = int(src_path.name.split("_", 2)[1])
                    pname = src_path.stem.split("_", 2)[2]
                    matched = match_one_part(src_path, _lookup_path(sig, pid, pname))
                    if ENABLE_URBAN_RURAL:
                        matched = add_urban_rural(matched, raster) if raster is not None else matched.assign(Urban_Rural=pd.NA)
                    matched.to_parquet(out_path, index=False, compression="zstd")
                    print(f"[{i}/{len(src_parts)}] {src_path.name}: {len(matched):,} grids")
                    if WRITE_COMPAT_NATIONAL_CSV:
                        compat_frames.append(matched)
            finally:
                if raster is not None:
                    raster.close()

            validate_complete_province_parts(discover_parts(out_dir), f"2_1 output {year}/{scope}/{profile}")
            if WRITE_COMPAT_NATIONAL_CSV and compat_frames:
                compat_dir = MULTISCALE_ROOT / str(year) / scope / profile / "compat_csv"
                compat_dir.mkdir(parents=True, exist_ok=True)
                pd.concat(compat_frames, ignore_index=True).to_csv(
                    compat_dir / f"national_{year}.csv", index=False, encoding="utf-8-sig"
                )


def main() -> None:
    if is_year_worker():
        process_year(get_run_year())
        return

    validate_accessibility_outputs()
    admin = load_admin()
    poor_counties = load_poor_counties()
    build_grid_admin_lookups(admin, poor_counties)
    print(
        f"\n2_1：grid-admin lookup 已就绪；开始年度并行 workers="
        f"{min(MULTISCALE_YEAR_WORKERS, len(YEARS))}."
    )
    run_script_for_all_years(
        __file__, years=YEARS, workers=MULTISCALE_YEAR_WORKERS,
        extra_env={LOOKUP_READY_ENV: "1"},
    )


if __name__ == "__main__":
    main()
