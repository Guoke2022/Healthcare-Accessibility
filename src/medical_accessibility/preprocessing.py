from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

x_pi = 3.14159265358979324 * 3000.0 / 180.0
pi = 3.1415926535897932384626
a = 6378245.0
ee = 0.00669342162296594323
interval = 0.000001


def format_time(value):
    text = str(value).strip()
    if text == "-1":
        return -1
    if "年" in text:
        year_part = text.split("年")[0]
        if year_part.isdigit() and len(year_part) == 4:
            return int(year_part)
    if text.isdigit() and len(text) == 4:
        return int(text)
    for sep in ["-", ".", "/"]:
        if sep in text:
            parts = text.split(sep)
            if parts and parts[0].isdigit() and len(parts[0]) == 4:
                return int(parts[0])
    return -1


def merge_hospital_metadata(base_csv: Path, supplemental_csv: Path, output_csv: Path) -> pd.DataFrame:
    base_df = pd.read_csv(base_csv, encoding="utf-8")
    supplemental_df = pd.read_csv(supplemental_csv, encoding="utf-8")
    addr_candidates = [column for column in ["addr", "addr_x", "addr_y"] if column in base_df.columns]
    keep_cols = ["name", "construction_time", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025", *addr_candidates]
    keep_cols = [column for column in keep_cols if column in base_df.columns]
    merged = pd.merge(supplemental_df, base_df[keep_cols], on="name", how="left", suffixes=("", "_base"))
    if "addr" not in merged.columns:
        for candidate in ["addr_x", "addr_y", "addr_x_base", "addr_y_base"]:
            if candidate in merged.columns:
                merged["addr"] = merged[candidate]
                break
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, encoding="utf-8-sig", index=False)
    return merged


def clean_construction_years(input_csv: Path, output_csv: Path, review_csv: Path | None = None, complete_csv: Path | None = None):
    df = pd.read_csv(input_csv, encoding="utf-8")
    df["construction_time"] = df["construction_time"].apply(format_time)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, encoding="utf-8-sig", index=False)
    target_cols = ["construction_time", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]
    mask = df[target_cols].eq(-1).any(axis=1)
    if review_csv is not None:
        df[mask].to_csv(review_csv, encoding="utf-8-sig", index=False)
    if complete_csv is not None:
        df[~mask].to_csv(complete_csv, encoding="utf-8-sig", index=False)
    return df


def geocode_baidu(address: str, ak: str):
    import requests

    response = requests.get("https://api.map.baidu.com/geocoding/v3/", params={"ak": ak, "address": address, "output": "json"}, timeout=10)
    data = response.json()
    if data["status"] == 0:
        location = data["result"]["location"]
        return location["lng"], location["lat"], data["result"]["confidence"], data["result"]["comprehension"]
    return None, None, None, None


def geocode_addresses(input_csv: Path, output_csv: Path, ak: str, address_col: str = "addr") -> pd.DataFrame:
    df = pd.read_csv(input_csv, encoding="utf-8")
    results = [geocode_baidu(address, ak) for address in df[address_col]]
    df["lng_bd08"] = [item[0] for item in results]
    df["lat_bd08"] = [item[1] for item in results]
    df["confidence"] = [item[2] for item in results]
    df["comprehension"] = [item[3] for item in results]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return df


def out_of_china(lng, lat):
    return not (73.66 < lng < 135.05 and 3.86 < lat < 53.55)


def _transformlat(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + 0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 * math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * pi) + 40.0 * math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * pi) + 320 * math.sin(lat * pi / 30.0)) * 2.0 / 3.0
    return ret


def _transformlng(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + 0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 * math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * pi) + 40.0 * math.sin(lng / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * pi) + 300.0 * math.sin(lng / 30.0 * pi)) * 2.0 / 3.0
    return ret


def wgs84_to_gcj02(lng, lat):
    if out_of_china(lng, lat):
        return lng, lat
    dlng = _transformlng(lng - 105.0, lat - 35.0)
    dlat = _transformlat(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    return lng + dlng, lat + dlat


def bd09_to_gcj02(bd_lon, bd_lat):
    x = bd_lon - 0.0065
    y = bd_lat - 0.006
    z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * x_pi)
    theta = math.atan2(y, x) - 0.000003 * math.cos(x * x_pi)
    return z * math.cos(theta), z * math.sin(theta)


def gcj02_to_wgs84(lng, lat):
    if out_of_china(lng, lat):
        return lng, lat
    dlng = _transformlng(lng - 105.0, lat - 35.0)
    dlat = _transformlat(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    wgslng = lng + dlng
    wgslat = lat + dlat
    corrected_lng, corrected_lat = wgs84_to_gcj02(wgslng, wgslat)
    clng = corrected_lng - lng
    clat = corrected_lat - lat
    dis = math.sqrt(clng * clng + clat * clat)
    while dis > interval:
        clng = clng / 2
        clat = clat / 2
        wgslng = wgslng - clng
        wgslat = wgslat - clat
        corrected_lng, corrected_lat = wgs84_to_gcj02(wgslng, wgslat)
        cclng = corrected_lng - lng
        cclat = corrected_lat - lat
        dis = math.sqrt(cclng * cclng + cclat * cclat)
        clng = clng if math.fabs(clng) > math.fabs(cclng) else cclng
        clat = clat if math.fabs(clat) > math.fabs(cclat) else cclat
    return wgslng, wgslat


def bd09_to_wgs84(bd_lon, bd_lat):
    lon, lat = bd09_to_gcj02(bd_lon, bd_lat)
    return gcj02_to_wgs84(lon, lat)


def convert_bd09_columns_to_wgs84(input_csv: Path, output_csv: Path, lng_col: str = "lng_bd08", lat_col: str = "lat_bd08"):
    df = pd.read_csv(input_csv, encoding="utf-8")
    converted = [bd09_to_wgs84(row[lng_col], row[lat_col]) for _, row in df.iterrows()]
    df["lng"] = [item[0] for item in converted]
    df["lat"] = [item[1] for item in converted]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return df


def assign_hospitals_to_years(input_csv: Path, output_dir: Path, start_year: int = 2014, end_year: int = 2024):
    df = pd.read_csv(input_csv, encoding="utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    for year in range(start_year, end_year + 1):
        condition1 = df["construction_time"] <= year
        condition2 = df["3A_year"].isna() | (df["3A_year"] <= year)
        condition3 = df[str(year)] > 0
        df_year = df[condition1 & condition2 & condition3].copy()
        keep_cols = ["name", "grade", "type", "province", "region", "area", "addr", "construction_time", "3A_year", str(year), "lng", "lat", "lng_bd08", "lat_bd08", "confidence", "comprehension"]
        df_year = df_year[keep_cols].rename(columns={str(year): "beds"})
        df_year.to_csv(output_dir / f"{year}.csv", encoding="utf-8-sig", index=False)


def reproject_raster_to_epsg4326(src_path: Path, dst_path: Path):
    import rasterio
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    target_crs = "EPSG:4326"
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(src.crs, target_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({"crs": target_crs, "transform": transform, "width": width, "height": height, "compress": "lzw", "BIGTIFF": "YES", "tiled": True, "blockxsize": 256, "blockysize": 256})
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest,
                )
