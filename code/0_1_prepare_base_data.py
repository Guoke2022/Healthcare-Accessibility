# -*- coding: utf-8 -*-


from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from pyproj import Transformer
from rasterio.errors import WindowError
from rasterio.features import geometry_mask, geometry_window

from config import (
    DATA_ROOT,
    PREPARED_ROOT,
    RUN_POLICY,
    get_run_year,
    is_year_worker,
    run_script_for_all_years,
    get_osm_snapshot_year,
    get_osm_snapshot_date,
    get_osm_snapshot_path,
    POPULATION_DATASET,
    population_dataset_label,
    get_population_raster_path,
)

from utils.cache import (
    build_fingerprint,
    cache_valid,
    write_cache_meta,
    prepare_stage_directory,
)
from utils.input_signatures import (
    hospital_routing_signature,
    hospital_supply_signature,
    population_routing_signature,
    population_accessibility_signature,
    SIGNATURE_SCHEMA_VERSION,
)


# BRANCH_THRESHOLD_HOSPITAL_INPUT_OVERRIDE_V2
HOSPITAL_DATA_DIR = Path(os.environ.get("NC_HOSPITAL_DATA_DIR", DATA_ROOT / "beds")).expanduser().resolve()

# =============================================================================

# =============================================================================

YEAR = get_run_year()
OSM_SNAPSHOT_YEAR = get_osm_snapshot_year(YEAR)
OSM_SNAPSHOT_DATE = get_osm_snapshot_date(YEAR)

HOSPITAL_CSV = HOSPITAL_DATA_DIR / f"{YEAR}.csv"
POPULATION_TIF = get_population_raster_path(YEAR)
OSM_PBF = get_osm_snapshot_path(YEAR)
PROVINCE_GEOJSON = DATA_ROOT / "中国84.geojson"

OUTPUT_ROOT = PREPARED_ROOT


PROVINCE_NAME_COL = "name"


HOSPITAL_LON_COL = "lng"
HOSPITAL_LAT_COL = "lat"
HOSPITAL_BEDS_COL = "beds"
HOSPITAL_NAME_COL = "name"
HOSPITAL_DECLARED_PROVINCE_COL = "province"


POPULATION_MIN_EXCLUSIVE = 0


CALCULATE_OSM_SHA256 = False


OVERWRITE = (RUN_POLICY == "force_rebuild")
COMMON_ONLY_ENV_VAR = "NC_PREPARE_COMMON_ONLY"

# Cache schema versions. Bump only the affected version when the corresponding
# transformation semantics change; this avoids invalidating unrelated expensive stages.
PROVINCE_CACHE_SCHEMA_VERSION = 2
HOSPITAL_CACHE_SCHEMA_VERSION = 2
POPULATION_CACHE_SCHEMA_VERSION = 2


# =============================================================================

# =============================================================================

def log(msg: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label}不存在：{path}")


def safe_filename(name: str) -> str:

    s = re.sub(r'[\\/:*?"<>|]+', "_", str(name))
    return s.strip().replace(" ", "_")


def file_sha256(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def read_csv_robust(path: Path) -> pd.DataFrame:

    last_error = None
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            df = pd.read_csv(path, encoding=enc)
            log(f"医院 CSV 编码：{enc}")
            return df
        except UnicodeDecodeError as e:
            last_error = e
    raise RuntimeError(f"无法识别 CSV 编码：{path}") from last_error


def make_geometry_valid(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    try:

        gdf["geometry"] = gdf.geometry.make_valid()
    except Exception:

        gdf["geometry"] = gdf.geometry.buffer(0)
    return gdf


def json_dump(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


# =============================================================================

# =============================================================================

def prepare_provinces() -> gpd.GeoDataFrame:
    out_path = OUTPUT_ROOT / "common" / "provinces.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fingerprint = build_fingerprint(
        config={
            "stage": "0_1_provinces",
            "cache_schema_version": PROVINCE_CACHE_SCHEMA_VERSION,
            "province_name_col": PROVINCE_NAME_COL,
            "target_crs": "EPSG:4326",
        },
        files=[PROVINCE_GEOJSON],
    )
    if not OVERWRITE and cache_valid(out_path, fingerprint, event_payload={"stage":"0_1_provinces","year":"common"}):
        log(f"省界缓存指纹一致，直接读取：{out_path}")
        return gpd.read_parquet(out_path)
    if out_path.exists() and not OVERWRITE:
        log("省界缓存指纹失配，自动重建")

    log("读取省级行政区...")
    provinces = gpd.read_file(PROVINCE_GEOJSON)

    if PROVINCE_NAME_COL not in provinces.columns:
        raise KeyError(
            f"省界缺少字段 {PROVINCE_NAME_COL!r}；实际字段：{list(provinces.columns)}"
        )
    if provinces.crs is None:
        raise ValueError("省界 GeoJSON 没有 CRS，无法安全处理。")

    provinces = provinces[[PROVINCE_NAME_COL, "geometry"]].copy()
    provinces = make_geometry_valid(provinces)
    provinces = provinces[~provinces.geometry.is_empty & provinces.geometry.notna()].copy()
    provinces = provinces.to_crs("EPSG:4326")


    provinces = provinces.dissolve(by=PROVINCE_NAME_COL, as_index=False)


    provinces = provinces.sort_values(PROVINCE_NAME_COL).reset_index(drop=True)
    provinces.insert(0, "province_id", np.arange(1, len(provinces) + 1, dtype=np.int16))
    provinces = provinces.rename(columns={PROVINCE_NAME_COL: "province_name"})

    bounds = provinces.geometry.bounds
    provinces["min_lon"] = bounds["minx"].astype("float64")
    provinces["min_lat"] = bounds["miny"].astype("float64")
    provinces["max_lon"] = bounds["maxx"].astype("float64")
    provinces["max_lat"] = bounds["maxy"].astype("float64")

    provinces.to_parquet(out_path, index=False)
    write_cache_meta(
        out_path,
        fingerprint,
        payload={"n_provinces": int(len(provinces))},
    )
    log(f"省界完成：{len(provinces)} 个省级单元 -> {out_path}")
    return provinces


# =============================================================================

# =============================================================================

def prepare_hospitals(provinces: gpd.GeoDataFrame) -> pd.DataFrame:
    out_path = OUTPUT_ROOT / str(YEAR) / "hospitals.parquet"
    qc_path = OUTPUT_ROOT / str(YEAR) / "qc" / "hospitals_qc.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    qc_path.parent.mkdir(parents=True, exist_ok=True)

    fingerprint = build_fingerprint(
        config={
            "stage": "0_1_hospitals",
            "cache_schema_version": HOSPITAL_CACHE_SCHEMA_VERSION,
            "signature_schema_version": SIGNATURE_SCHEMA_VERSION,
            "year": YEAR,
            "lon_col": HOSPITAL_LON_COL,
            "lat_col": HOSPITAL_LAT_COL,
            "beds_col": HOSPITAL_BEDS_COL,
        },
        files=[HOSPITAL_CSV, OUTPUT_ROOT / "common" / "provinces.parquet"],
    )
    if not OVERWRITE and cache_valid(
        out_path, fingerprint, event_payload={"stage":"0_1_hospitals","year":YEAR}
    ):
        log(f"医院标准表缓存指纹一致，直接读取：{out_path}")
        return pd.read_parquet(out_path)
    if out_path.exists() and not OVERWRITE:
        log("医院标准表缓存指纹失配，自动重建")

    log("读取并标准化医院数据...")
    df = read_csv_robust(HOSPITAL_CSV)

    required = [HOSPITAL_LON_COL, HOSPITAL_LAT_COL, HOSPITAL_BEDS_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"医院 CSV 缺少必要字段：{missing}；实际字段：{list(df.columns)}")

    df = df.copy()
    df.insert(0, "source_row", np.arange(len(df), dtype=np.int32))
    df.insert(
        0,
        "hospital_id",
        [f"H{YEAR}_{i:06d}" for i in range(1, len(df) + 1)],
    )


    df["lon"] = pd.to_numeric(df[HOSPITAL_LON_COL], errors="coerce")
    df["lat"] = pd.to_numeric(df[HOSPITAL_LAT_COL], errors="coerce")
    df["beds_std"] = pd.to_numeric(df[HOSPITAL_BEDS_COL], errors="coerce")

    valid_coord = (
        df["lon"].between(-180, 180)
        & df["lat"].between(-90, 90)
        & df["lon"].notna()
        & df["lat"].notna()
    )


    point_gdf = gpd.GeoDataFrame(
        df.loc[valid_coord, ["hospital_id"]].copy(),
        geometry=gpd.points_from_xy(
            df.loc[valid_coord, "lon"],
            df.loc[valid_coord, "lat"],
        ),
        crs="EPSG:4326",
    )

    prov_join = provinces[["province_id", "province_name", "geometry"]].copy()

    if len(point_gdf) > 0:
        joined = gpd.sjoin(
            point_gdf,
            prov_join,
            how="left",
            predicate="within",
        )

        joined = (
            joined.reset_index(drop=True)
            .sort_values(["hospital_id", "province_id"], na_position="last")
            .drop_duplicates("hospital_id", keep="first")
        )
        spatial_map = joined.set_index("hospital_id")[["province_id", "province_name"]]
        df["province_id_spatial"] = df["hospital_id"].map(spatial_map["province_id"])
        df["province_name_spatial"] = df["hospital_id"].map(spatial_map["province_name"])
    else:
        df["province_id_spatial"] = np.nan
        df["province_name_spatial"] = None


    lon6 = df["lon"].round(6)
    lat6 = df["lat"].round(6)
    coord_key = pd.DataFrame({"lon6": lon6, "lat6": lat6})
    df["coord_duplicate_n"] = coord_key.groupby(["lon6", "lat6"])["lon6"].transform("size")
    df.loc[~valid_coord, "coord_duplicate_n"] = 0
    df["coord_duplicate_n"] = df["coord_duplicate_n"].fillna(0).astype(np.int16)

    df["qc_invalid_coordinate"] = ~valid_coord
    df["qc_outside_china"] = valid_coord & df["province_name_spatial"].isna()
    df["qc_beds_missing_or_nonpositive"] = df["beds_std"].isna() | (df["beds_std"] <= 0)
    df["qc_duplicate_coordinate"] = df["coord_duplicate_n"] > 1

    if HOSPITAL_DECLARED_PROVINCE_COL in df.columns:
        declared = df[HOSPITAL_DECLARED_PROVINCE_COL].astype("string").str.strip()
        spatial = df["province_name_spatial"].astype("string").str.strip()
        df["qc_declared_spatial_province_mismatch"] = (
            declared.notna()
            & spatial.notna()
            & (declared != spatial)
        )
    else:
        df["qc_declared_spatial_province_mismatch"] = False


    df["is_valid_for_routing"] = (
        ~df["qc_invalid_coordinate"]
        & ~df["qc_outside_china"]
    )


    routing_sig = hospital_routing_signature(df)
    supply_sig = hospital_supply_signature(df)


    df.to_parquet(out_path, index=False)
    write_cache_meta(
        out_path,
        fingerprint,
        payload={
            "n_hospitals": int(len(df)),
            "n_valid_for_routing": int(df["is_valid_for_routing"].sum()),
            "routing_signature": routing_sig,
            "supply_signature": supply_sig,
            "signature_schema_version": SIGNATURE_SCHEMA_VERSION,
        },
    )

    qc_cols = [
        "hospital_id",
        "source_row",
    ]
    for c in (
        HOSPITAL_NAME_COL,
        HOSPITAL_DECLARED_PROVINCE_COL,
        "region",
        "area",
        "addr",
        "grade",
    ):
        if c in df.columns and c not in qc_cols:
            qc_cols.append(c)

    qc_cols += [
        "lon",
        "lat",
        "beds_std",
        "province_name_spatial",
        "coord_duplicate_n",
        "qc_invalid_coordinate",
        "qc_outside_china",
        "qc_beds_missing_or_nonpositive",
        "qc_duplicate_coordinate",
        "qc_declared_spatial_province_mismatch",
        "is_valid_for_routing",
    ]

    qc_mask = (
        df["qc_invalid_coordinate"]
        | df["qc_outside_china"]
        | df["qc_beds_missing_or_nonpositive"]
        | df["qc_duplicate_coordinate"]
        | df["qc_declared_spatial_province_mismatch"]
    )
    df.loc[qc_mask, qc_cols].to_csv(qc_path, index=False, encoding="utf-8-sig")

    log(
        "医院完成："
        f"总数={len(df):,}；"
        f"可用于 routing={int(df['is_valid_for_routing'].sum()):,}；"
        f"QC异常={int(qc_mask.sum()):,}"
    )
    log(f"医院标准表 -> {out_path}")
    log(f"医院 QC -> {qc_path}")
    return df


# =============================================================================

# =============================================================================

def prepare_population_raster(provinces_wgs84: gpd.GeoDataFrame) -> pd.DataFrame:

    parts_dir = OUTPUT_ROOT / str(YEAR) / "population_parts"
    summary_path = OUTPUT_ROOT / str(YEAR) / "qc" / "population_summary.csv"


    metadata_path = OUTPUT_ROOT / str(YEAR) / "metadata" / "landscan_metadata.json"

    population_stage_fingerprint = build_fingerprint(
        config={
            "stage": "0_1_population_parts",
            "cache_schema_version": POPULATION_CACHE_SCHEMA_VERSION,
            "signature_schema_version": SIGNATURE_SCHEMA_VERSION,
            "year": YEAR,
            "population_min_exclusive": POPULATION_MIN_EXCLUSIVE,
            "population_dataset": POPULATION_DATASET,
            "cell_representation": "raster_cell_center",
        },
        files=[POPULATION_TIF, OUTPUT_ROOT / "common" / "provinces.parquet"],
    )
    population_stage_status = prepare_stage_directory(
        parts_dir,
        population_stage_fingerprint,
        run_policy=RUN_POLICY,
        payload={"stage": "0_1_population_parts", "year": YEAR},
    )
    log(f"人口分片目录状态：{population_stage_status}")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    summaries = []

    log(f"打开人口栅格：{population_dataset_label()} | {POPULATION_TIF}")
    with rasterio.open(POPULATION_TIF) as src:
        if src.crs is None:
            raise ValueError(f"人口栅格没有 CRS，无法安全进行省界 mask：{POPULATION_TIF}")

        raster_crs = src.crs
        transform = src.transform
        width = src.width
        height = src.height
        nodata = src.nodata
        dtype = src.dtypes[0]

        log(
            f"{population_dataset_label()}: size={width}×{height}, crs={raster_crs}, "
            f"dtype={dtype}, nodata={nodata}"
        )


        provinces_raster = provinces_wgs84[
            ["province_id", "province_name", "geometry"]
        ].to_crs(raster_crs)


        need_transform_to_wgs84 = raster_crs.to_string().upper() not in (
            "EPSG:4326",
            "OGC:CRS84",
        )
        transformer = None
        if need_transform_to_wgs84:
            transformer = Transformer.from_crs(
                raster_crs,
                "EPSG:4326",
                always_xy=True,
            )

        for i, row in provinces_raster.iterrows():
            province_id = int(row["province_id"])
            province_name = str(row["province_name"])
            geom = row.geometry

            part_path = (
                parts_dir
                / f"province_{province_id:03d}_{safe_filename(province_name)}.parquet"
            )

            part_fingerprint = build_fingerprint(
                config={
                    "stage": "0_1_population_part",
                    "cache_schema_version": POPULATION_CACHE_SCHEMA_VERSION,
                    "signature_schema_version": SIGNATURE_SCHEMA_VERSION,
                    "year": YEAR,
                    "province_id": province_id,
                    "province_name": province_name,
                    "population_min_exclusive": POPULATION_MIN_EXCLUSIVE,
                    "population_dataset": POPULATION_DATASET,
                    "cell_representation": "raster_cell_center",
                    "mask_all_touched": False,
                },
                files=[
                    POPULATION_TIF,
                    OUTPUT_ROOT / "common" / "provinces.parquet",
                ],
            )

            if not OVERWRITE and cache_valid(
                part_path, part_fingerprint,
                event_payload={"stage":"0_1_population_part","year":YEAR,"province_id":province_id,"province_name":province_name},
            ):
                old = pd.read_parquet(part_path, columns=["population"])
                summaries.append(
                    {
                        "province_id": province_id,
                        "province_name": province_name,
                        "n_populated_cells": int(len(old)),
                        "population_sum": float(old["population"].sum()),
                        "status": "existing_validated",
                    }
                )
                log(
                    f"[{province_id:02d}/{len(provinces_raster)}] "
                    f"{province_name} 缓存指纹一致，跳过"
                )
                continue
            if part_path.exists() and not OVERWRITE:
                log(f"{province_name}: population 缓存指纹失配，自动重建")

            t0 = time.time()

            try:
                window = geometry_window(
                    src,
                    [geom],
                    pad_x=0,
                    pad_y=0,
                )
            except WindowError:
                log(f"{province_name}: 与当前人口栅格无交集，跳过")
                summaries.append(
                    {
                        "province_id": province_id,
                        "province_name": province_name,
                        "n_populated_cells": 0,
                        "population_sum": 0.0,
                        "status": "no_overlap",
                    }
                )
                continue


            window = window.round_offsets().round_lengths()

            data = src.read(1, window=window, masked=False)
            win_transform = src.window_transform(window)


            inside = geometry_mask(
                [geom],
                out_shape=data.shape,
                transform=win_transform,
                invert=True,
                all_touched=False,
            )

            valid = inside & np.isfinite(data) & (data > POPULATION_MIN_EXCLUSIVE)
            if nodata is not None and np.isfinite(nodata):
                valid &= (data != nodata)

            local_rows, local_cols = np.nonzero(valid)

            if len(local_rows) == 0:
                empty_part = pd.DataFrame(
                    {
                        "grid_id": pd.Series(dtype="int64"),
                        "province_id": pd.Series(dtype="int16"),
                        "province_name": pd.Series(dtype="string"),
                        "src_row": pd.Series(dtype="int32"),
                        "src_col": pd.Series(dtype="int32"),
                        "population": pd.Series(dtype="float32"),
                        "lon": pd.Series(dtype="float64"),
                        "lat": pd.Series(dtype="float64"),
                    }
                )
                empty_part.to_parquet(part_path, index=False)
                write_cache_meta(
                    part_path,
                    part_fingerprint,
                    payload={
                        "province_id": province_id,
                        "province_name": province_name,
                        "n_populated_cells": 0,
                        "population_sum": 0.0,
                        "routing_signature": population_routing_signature(empty_part),
                        "accessibility_signature": population_accessibility_signature(empty_part),
                        "signature_schema_version": SIGNATURE_SCHEMA_VERSION,
                    },
                )

                summaries.append(
                    {
                        "province_id": province_id,
                        "province_name": province_name,
                        "n_populated_cells": 0,
                        "population_sum": 0.0,
                        "status": "ok",
                    }
                )
                continue

            global_rows = local_rows.astype(np.int64) + int(window.row_off)
            global_cols = local_cols.astype(np.int64) + int(window.col_off)


            grid_id = global_rows * np.int64(width) + global_cols


            c = global_cols.astype(np.float64) + 0.5
            r = global_rows.astype(np.float64) + 0.5

            x = (
                transform.c
                + c * transform.a
                + r * transform.b
            )
            y = (
                transform.f
                + c * transform.d
                + r * transform.e
            )

            if transformer is not None:
                lon, lat = transformer.transform(x, y)
                lon = np.asarray(lon, dtype=np.float64)
                lat = np.asarray(lat, dtype=np.float64)
            else:
                lon = x.astype(np.float64, copy=False)
                lat = y.astype(np.float64, copy=False)

            population = data[valid]


            if np.issubdtype(population.dtype, np.integer):
                population_out = population.astype(np.int32, copy=False)
            else:
                population_out = population.astype(np.float32, copy=False)

            part = pd.DataFrame(
                {
                    "grid_id": grid_id.astype(np.int64, copy=False),
                    "province_id": np.full(
                        len(grid_id),
                        province_id,
                        dtype=np.int16,
                    ),
                    "province_name": province_name,
                    "src_row": global_rows.astype(np.int32, copy=False),
                    "src_col": global_cols.astype(np.int32, copy=False),
                    "population": population_out,
                    "lon": lon,
                    "lat": lat,
                }
            )


            part.to_parquet(
                part_path,
                index=False,
                compression="zstd",
            )

            pop_sum = float(part["population"].sum())
            routing_sig = population_routing_signature(part)
            accessibility_sig = population_accessibility_signature(part)
            write_cache_meta(
                part_path,
                part_fingerprint,
                payload={
                    "province_id": province_id,
                    "province_name": province_name,
                    "n_populated_cells": int(len(part)),
                    "population_sum": pop_sum,
                    "routing_signature": routing_sig,
                    "accessibility_signature": accessibility_sig,
                    "signature_schema_version": SIGNATURE_SCHEMA_VERSION,
                },
            )
            summaries.append(
                {
                    "province_id": province_id,
                    "province_name": province_name,
                    "n_populated_cells": int(len(part)),
                    "population_sum": pop_sum,
                    "status": "ok",
                }
            )

            log(
                f"[{province_id:02d}/{len(provinces_raster)}] "
                f"{province_name}: cells={len(part):,}, "
                f"pop={pop_sum:,.0f}, "
                f"{time.time() - t0:.1f}s"
            )


            del data, inside, valid, local_rows, local_cols
            del global_rows, global_cols, grid_id, x, y, part

        metadata = {
            "year": YEAR,
            "source_file": str(POPULATION_TIF),
            "population_dataset": POPULATION_DATASET,
            "population_dataset_label": population_dataset_label(),
            "width": width,
            "height": height,
            "count": src.count,
            "dtype": dtype,
            "nodata": nodata,
            "crs": str(raster_crs),
            "transform": [
                transform.a,
                transform.b,
                transform.c,
                transform.d,
                transform.e,
                transform.f,
            ],
            "bounds": list(src.bounds),
            "population_filter": f"population > {POPULATION_MIN_EXCLUSIVE}",
            "grid_id_formula": "grid_id = src_row * raster_width + src_col",
            "cell_representation": "raster cell center",
        }
        json_dump(metadata, metadata_path)

    summary = pd.DataFrame(summaries).sort_values("province_id")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    log(
        f"{population_dataset_label()} 完成："
        f"有效人口格网={int(summary['n_populated_cells'].sum()):,}；"
        f"人口总和={summary['population_sum'].sum():,.0f}"
    )
    log(f"人口分片目录 -> {parts_dir}")
    log(f"人口汇总 -> {summary_path}")
    return summary


# =============================================================================

# =============================================================================

def register_osm_metadata() -> dict:
    metadata_path = OUTPUT_ROOT / str(YEAR) / "metadata" / "osm_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    stat = OSM_PBF.stat()
    meta = {
        "year": YEAR,
        "analysis_year": YEAR,
        "osm_snapshot_year": OSM_SNAPSHOT_YEAR,
        "osm_snapshot_date": OSM_SNAPSHOT_DATE,
        "osm_snapshot_alignment": "analysis_year_end_proxy_y_plus_1_jan1",
        "source_file": str(OSM_PBF),
        "filename": OSM_PBF.name,
        "size_bytes": stat.st_size,
        "size_gb": stat.st_size / (1024 ** 3),
        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "sha256": None,
        "note": (
            "0_1_prepare_base_data 不裁切 OSM。"
            "OSM 快照按统一口径使用分析年 Y 对应的 Y+1-01-01；"
            "后续 0_2_prepare_road_graph 按省界与固定 routing buffer 裁切；工程中不存在 OD-candidate 阶段。"
        ),
    }

    if CALCULATE_OSM_SHA256:
        log("计算 OSM PBF SHA256（会顺序读取整个大文件）...")
        meta["sha256"] = file_sha256(OSM_PBF)

    json_dump(meta, metadata_path)
    log(
        f"OSM 已登记：analysis_year={YEAR}, snapshot={OSM_SNAPSHOT_DATE}, {OSM_PBF.name}, "
        f"{meta['size_gb']:.2f} GB -> {metadata_path}"
    )
    return meta


# =============================================================================

# =============================================================================

def prepare_common_data() -> gpd.GeoDataFrame:

    ensure_exists(PROVINCE_GEOJSON, "省级行政区 GeoJSON")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    return prepare_provinces()


def main() -> None:
    total_t0 = time.time()

    log("=" * 80)
    log("0_1_prepare_base_data 开始")
    log(f"YEAR = {YEAR}")
    log("=" * 80)

    ensure_exists(PROVINCE_GEOJSON, "省级行政区 GeoJSON")
    ensure_exists(HOSPITAL_CSV, "医院 CSV")
    ensure_exists(POPULATION_TIF, f"{population_dataset_label()} population raster")
    ensure_exists(OSM_PBF, "中国 OSM PBF")


    provinces = prepare_common_data()


    hospitals = prepare_hospitals(provinces)
    population_summary = prepare_population_raster(provinces)
    osm_meta = register_osm_metadata()

    run_summary = {
        "year": YEAR,
        "completed_at": datetime.now().isoformat(),
        "elapsed_seconds": time.time() - total_t0,
        "n_provinces": int(len(provinces)),
        "n_hospitals": int(len(hospitals)),
        "n_hospitals_valid_for_routing": int(
            hospitals["is_valid_for_routing"].sum()
        ),
        "n_populated_cells": int(
            population_summary["n_populated_cells"].sum()
        ),
        "population_sum": float(
            population_summary["population_sum"].sum()
        ),
        "osm_snapshot_year": OSM_SNAPSHOT_YEAR,
        "osm_snapshot_date": OSM_SNAPSHOT_DATE,
        "osm_file": str(OSM_PBF),
        "output_root": str(OUTPUT_ROOT),
    }

    summary_path = OUTPUT_ROOT / str(YEAR) / "metadata" / "run_summary.json"
    json_dump(run_summary, summary_path)

    log("=" * 80)
    log("全部完成")
    log(
        f"provinces={run_summary['n_provinces']}, "
        f"hospitals={run_summary['n_hospitals']:,}, "
        f"populated_cells={run_summary['n_populated_cells']:,}"
    )
    log(f"总耗时：{run_summary['elapsed_seconds'] / 60:.1f} min")
    log(f"输出目录：{OUTPUT_ROOT}")
    log("=" * 80)


if __name__ == "__main__":
    common_only = os.environ.get(COMMON_ONLY_ENV_VAR, "0").strip().lower() in {"1", "true", "yes", "y"}
    if common_only:
        log("仅准备跨年份共享数据（common-only）")
        prepare_common_data()
    elif is_year_worker():
        main()
    else:

        prepare_common_data()
        run_script_for_all_years(__file__)
