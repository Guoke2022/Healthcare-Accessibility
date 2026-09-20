# -*- coding: utf-8 -*-
"""Canonical data readers and derived grouped statistics for manuscript figures.

Plotting scripts consume tables through this module; expensive Urban/Rural derivation is
precomputed and fingerprinted separately from plotting.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    RESULT_ROOT,
    ADMIN_SHP,
    PROVINCE_SHP,
    CITY_SHP,
    COUNTY_SHP,
    SERVICE_SCOPES,
    PROFILES,
    YEARS,
    YEAR_TO_GURS,
    CI_MATCHED_ROOT,
    CI_ANALYSIS_ROOT,
)
from utils.cache import build_fingerprint, cache_valid, write_cache_meta
from utils.inequality_schema import INEQUALITY_SCHEMA_VERSION
from utils.multiscale import (
    calculate_accessibility_stats,
    calculate_travel_time_stats,
    discover_parts,
    matched_parts_dir,
    stats_output_dir,
)


def _scope_profile() -> tuple[str, str]:
    scope = os.environ.get("NC_FIGURE_SERVICE_SCOPE") or (SERVICE_SCOPES[0] if len(SERVICE_SCOPES) == 1 else None)
    profile = os.environ.get("NC_FIGURE_SPEED_PROFILE") or (PROFILES[0] if len(PROFILES) == 1 else None)
    if not scope or not profile:
        raise RuntimeError(
            "绘图需要唯一 scope/profile；请设置 NC_FIGURE_SERVICE_SCOPE 和 NC_FIGURE_SPEED_PROFILE。"
        )
    return scope, profile


def figure_root() -> Path:
    p = RESULT_ROOT / "Figure"
    p.mkdir(parents=True, exist_ok=True)
    return p


def figure_dir(*parts: str) -> Path:
    p = figure_root().joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_stats(group: str, kind: str) -> pd.DataFrame:
    """Read a current 2_2 statistics CSV.

    kind: ``travel_time`` or ``accessibility``. Historical accessibility files used
    ``*_acc_stats.csv``; ``stats_output_dir`` already points at the corresponding folder.
    """
    scope, profile = _scope_profile()
    suffix = "travel_time_stats.csv" if kind == "travel_time" else "acc_stats.csv"
    path = stats_output_dir(scope, profile, kind) / f"{group}_{suffix}"
    if not path.exists():
        raise FileNotFoundError(f"缺少新版 2_2 统计文件：{path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def _parquet_columns(path: Path) -> set[str]:
    try:
        import pyarrow.parquet as pq
        return set(pq.ParquetFile(path).schema.names)
    except Exception:
        return set(pd.read_parquet(path).columns)


def _read_matched_year(year: int, columns: list[str]) -> pd.DataFrame:
    scope, profile = _scope_profile()
    parts = discover_parts(matched_parts_dir(year, scope, profile))
    if not parts:
        raise FileNotFoundError(f"缺少新版 2_1 matched 数据：{matched_parts_dir(year, scope, profile)}")
    frames = []
    for p in parts:
        available = _parquet_columns(p)
        use = [c for c in columns if c in available]
        missing = [c for c in columns if c not in available]
        # Urban_Rural is allowed to be absent and may be reconstructed below.
        hard_missing = [c for c in missing if c != "Urban_Rural"]
        if hard_missing:
            raise KeyError(f"{p} 缺少绘图字段：{hard_missing}")
        frames.append(pd.read_parquet(p, columns=use))
    return pd.concat(frames, ignore_index=True)


def _group_cache_info(group: str, kind: str):
    """Return cache path/fingerprint for a derived grouped-statistics table."""
    if group not in {"Coastal_Inland", "Urban_Rural"}:
        raise ValueError(group)
    if kind not in {"accessibility", "travel_time"}:
        raise ValueError(kind)

    years_to_build = [y for y in YEARS if y in YEAR_TO_GURS] if group == "Urban_Rural" else list(YEARS)
    scope, profile = _scope_profile()
    cache_dir = RESULT_ROOT / "2_2_multiscale_analysis" / scope / profile / "derived_groups"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{group}_{'acc' if kind == 'accessibility' else 'travel_time'}_stats.csv"

    fingerprint_files = [Path(__file__).resolve(), Path(__file__).resolve().parent / "inequality_metrics.py", Path(__file__).resolve().parent / "multiscale.py", Path(__file__).resolve().parent / "inequality_schema.py"]
    for y in years_to_build:
        fingerprint_files.extend(discover_parts(matched_parts_dir(int(y), scope, profile)))
        if group == "Urban_Rural":
            gurs = YEAR_TO_GURS.get(int(y))
            if gurs is not None and Path(gurs).exists():
                fingerprint_files.append(Path(gurs))
    fp = build_fingerprint(
        config={
            "derived_group": group,
            "kind": kind,
            "years": list(map(int, years_to_build)),
            "scope": scope,
            "profile": profile,
            # v3: Urban/Rural uses a standalone fast precomputation instead of
            # one raster read per grid during plotting.
            "version": 4,
            "inequality_schema_version": INEQUALITY_SCHEMA_VERSION,
        },
        files=fingerprint_files,
    )
    return cache, fp, list(map(int, years_to_build)), scope, profile


def _urban_rural_year_cache(year: int, kind: str, scope: str, profile: str) -> tuple[Path, str]:
    """Per-year resume cache for the expensive GURS classification."""
    parts = discover_parts(matched_parts_dir(int(year), scope, profile))
    if not parts:
        raise FileNotFoundError(f"缺少新版 2_1 matched 数据：{matched_parts_dir(year, scope, profile)}")
    gurs = YEAR_TO_GURS.get(int(year))
    if gurs is None or not Path(gurs).exists():
        raise FileNotFoundError(f"{year} 找不到GURS：{gurs}")

    out_dir = RESULT_ROOT / "2_2_multiscale_analysis" / scope / profile / "derived_groups" / "Urban_Rural_yearly"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "acc" if kind == "accessibility" else "travel_time"
    cache = out_dir / f"Urban_Rural_{int(year)}_{suffix}_stats.csv"
    fp = build_fingerprint(
        config={
            "derived_group": "Urban_Rural",
            "kind": kind,
            "year": int(year),
            "scope": scope,
            "profile": profile,
            "algorithm": "legacy_10x10_majority_integral_window",
            "version": 2,
            "inequality_schema_version": INEQUALITY_SCHEMA_VERSION,
        },
        files=[*parts, Path(gurs), Path(__file__).resolve(), Path(__file__).resolve().parent / "inequality_metrics.py", Path(__file__).resolve().parent / "multiscale.py", Path(__file__).resolve().parent / "inequality_schema.py"],
    )
    return cache, fp


def _classify_gurs_10x10_fast(df: pd.DataFrame, raster) -> pd.Series:
    """Vectorized equivalent of the historical 10x10 GURS majority rule.

    Historical code performed one ``raster.read`` call per accessibility grid.  That is
    exact but extremely slow nationally.  Here grids are processed city-by-city: one
    raster window is read for the city and integral images count each categorical GURS
    value inside every grid's 10x10 neighbourhood.  The window bounds, nodata handling,
    sorted-value tie breaking and Urban=1/Rural=2 mapping match the old implementation.
    """
    import rasterio
    from rasterio.transform import rowcol
    from rasterio.warp import transform as rio_transform
    from rasterio.windows import Window

    labels = pd.Series("Unknown", index=df.index, dtype="object")
    if len(df) == 0:
        return labels
    if raster.crs is None:
        raise ValueError("GURS 栅格缺少 CRS。")

    # city_name_norm is deliberately used only as an IO tiling key; it does not
    # participate in the classification definition itself.
    if "city_name_norm" in df.columns:
        group_key = df["city_name_norm"].astype("string").fillna("__MISSING_CITY__")
    else:
        group_key = pd.Series("__ALL__", index=df.index, dtype="string")

    nodata = raster.nodata
    raster_crs = str(raster.crs).upper()

    for _, zidx in group_key.groupby(group_key, sort=False).groups.items():
        z = df.loc[zidx]
        lon = pd.to_numeric(z["lon"], errors="coerce").to_numpy(dtype=np.float64)
        lat = pd.to_numeric(z["lat"], errors="coerce").to_numpy(dtype=np.float64)
        coord_ok = np.isfinite(lon) & np.isfinite(lat)
        if not coord_ok.any():
            continue

        valid_index = z.index[coord_ok]
        xs, ys = lon[coord_ok], lat[coord_ok]
        if raster_crs not in {"EPSG:4326", "OGC:CRS84"}:
            xx, yy = rio_transform("EPSG:4326", raster.crs, xs.tolist(), ys.tolist())
            xs = np.asarray(xx, dtype=np.float64)
            ys = np.asarray(yy, dtype=np.float64)

        # raster.index uses floor; rowcol(..., op=np.floor) is its vectorized equivalent.
        rr, cc = rowcol(raster.transform, xs, ys, op=np.floor)
        rows = np.asarray(rr, dtype=np.int64)
        cols = np.asarray(cc, dtype=np.int64)

        # Old window is Window(col-5, row-5, 10, 10), i.e. [center-5, center+5).
        r0 = max(0, int(rows.min()) - 5)
        c0 = max(0, int(cols.min()) - 5)
        r1 = min(raster.height, int(rows.max()) + 5)
        c1 = min(raster.width, int(cols.max()) + 5)
        if r1 <= r0 or c1 <= c0:
            continue

        arr = raster.read(1, window=Window(c0, r0, c1 - c0, r1 - r0), boundless=False)
        finite = np.isfinite(arr)
        if nodata is not None and np.isfinite(nodata):
            finite &= arr != nodata
        values = np.unique(arr[finite])
        if len(values) == 0:
            continue

        # Categorical GURS should contain only a handful of classes. Guard against
        # accidentally pointing at a continuous raster, which would be both wrong and huge.
        if len(values) > 32:
            raise ValueError(
                f"GURS 窗口出现 {len(values)} 个类别值，疑似不是分类栅格；"
                f"示例={values[:10].tolist()}"
            )

        lr = rows - r0
        lc = cols - c0
        top = np.clip(lr - 5, 0, arr.shape[0]).astype(np.int64)
        bottom = np.clip(lr + 5, 0, arr.shape[0]).astype(np.int64)
        left = np.clip(lc - 5, 0, arr.shape[1]).astype(np.int64)
        right = np.clip(lc + 5, 0, arr.shape[1]).astype(np.int64)

        counts = np.zeros((len(rows), len(values)), dtype=np.int16)
        for j, value in enumerate(values):
            mask = finite & (arr == value)
            # Padded integral image: rectangle count is O(1) for every grid.
            integ = np.pad(mask.astype(np.int32), ((1, 0), (1, 0)), mode="constant")
            integ = integ.cumsum(axis=0).cumsum(axis=1)
            counts[:, j] = (
                integ[bottom, right]
                - integ[top, right]
                - integ[bottom, left]
                + integ[top, left]
            ).astype(np.int16)

        total = counts.sum(axis=1)
        has_value = total > 0
        # values is sorted and np.argmax returns the first maximum, exactly matching
        # np.unique(..., return_counts=True) + unique[np.argmax(counts)] on ties.
        winners = values[np.argmax(counts, axis=1)]
        out = np.full(len(rows), "Unknown", dtype=object)
        out[has_value & (winners == 1)] = "Urban"
        out[has_value & (winners == 2)] = "Rural"
        labels.loc[valid_index] = out

    return labels


def _compute_urban_rural_one_year(year: int, scope: str, profile: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify one year once, then calculate both accessibility and travel-time stats."""
    import rasterio

    parts = discover_parts(matched_parts_dir(int(year), scope, profile))
    if not parts:
        raise FileNotFoundError(f"缺少新版 2_1 matched 数据：{matched_parts_dir(year, scope, profile)}")
    gurs_path = YEAR_TO_GURS.get(int(year))
    if gurs_path is None or not Path(gurs_path).exists():
        raise FileNotFoundError(f"{year} 找不到GURS：{gurs_path}")

    frames: list[pd.DataFrame] = []
    with rasterio.open(gurs_path) as raster:
        for i, part in enumerate(parts, 1):
            available = _parquet_columns(part)
            required = ["pop", "acc", "travel_time", "lon", "lat"]
            missing = [c for c in required if c not in available]
            if missing:
                raise KeyError(f"{part} 缺少城乡预统计字段：{missing}")
            use = required + [c for c in ["city_name_norm", "Urban_Rural"] if c in available]
            z = pd.read_parquet(part, columns=list(dict.fromkeys(use)))

            existing = z.get("Urban_Rural")
            if existing is not None and existing.isin(["Urban", "Rural", "Unknown"]).any():
                z["Urban_Rural"] = existing.fillna("Unknown")
            else:
                z["Urban_Rural"] = _classify_gurs_10x10_fast(z, raster)

            frames.append(z[["pop", "acc", "travel_time", "Urban_Rural"]].copy())
            n_u = int((z["Urban_Rural"] == "Urban").sum())
            n_r = int((z["Urban_Rural"] == "Rural").sum())
            n_x = int((z["Urban_Rural"] == "Unknown").sum())
            print(
                f"  [{i:02d}/{len(parts):02d}] {part.name}: "
                f"grids={len(z):,}, Urban={n_u:,}, Rural={n_r:,}, Unknown={n_x:,}",
                flush=True,
            )

    all_df = pd.concat(frames, ignore_index=True)
    acc_rows = []
    time_rows = []
    for value, sub in all_df.groupby("Urban_Rural", dropna=False, sort=True):
        acc_rows.append({"Year": int(year), "Urban_Rural": value, **calculate_accessibility_stats(sub)})
        time_rows.append({"Year": int(year), "Urban_Rural": value, **calculate_travel_time_stats(sub)})
    return pd.DataFrame(acc_rows), pd.DataFrame(time_rows)


def precompute_urban_rural_stats(force: bool = False) -> dict[str, Path]:
    """Precompute Figure-3 Urban/Rural statistics before any plotting.

    Work is checkpointed year-by-year.  A rerun reuses every year whose matched parts,
    GURS raster and algorithm fingerprint are unchanged.
    """
    final_acc, final_acc_fp, years, scope, profile = _group_cache_info("Urban_Rural", "accessibility")
    final_time, final_time_fp, years_time, scope2, profile2 = _group_cache_info("Urban_Rural", "travel_time")
    if years_time != years or scope2 != scope or profile2 != profile:
        raise RuntimeError("Urban/Rural accessibility 与 travel-time 缓存配置不一致。")

    if not force and cache_valid(final_acc, final_acc_fp) and cache_valid(final_time, final_time_fp):
        print(f"Urban/Rural 统计缓存已有效，直接复用：{final_acc}", flush=True)
        return {"accessibility": final_acc, "travel_time": final_time}

    acc_years: list[pd.DataFrame] = []
    time_years: list[pd.DataFrame] = []
    for year in years:
        acc_cache, acc_fp = _urban_rural_year_cache(year, "accessibility", scope, profile)
        time_cache, time_fp = _urban_rural_year_cache(year, "travel_time", scope, profile)
        acc_ok = (not force) and cache_valid(acc_cache, acc_fp)
        time_ok = (not force) and cache_valid(time_cache, time_fp)

        if acc_ok and time_ok:
            print(f"[{year}] 年度城乡统计已缓存，跳过。", flush=True)
            acc_df = pd.read_csv(acc_cache, encoding="utf-8-sig")
            time_df = pd.read_csv(time_cache, encoding="utf-8-sig")
        else:
            print(f"[{year}] 开始按10x10 GURS 口径预计算城乡统计...", flush=True)
            acc_df, time_df = _compute_urban_rural_one_year(year, scope, profile)
            acc_df.to_csv(acc_cache, index=False, encoding="utf-8-sig")
            time_df.to_csv(time_cache, index=False, encoding="utf-8-sig")
            write_cache_meta(acc_cache, acc_fp, payload={"derived_group": "Urban_Rural", "kind": "accessibility", "year": year})
            write_cache_meta(time_cache, time_fp, payload={"derived_group": "Urban_Rural", "kind": "travel_time", "year": year})
            print(f"[{year}] 完成并写入年度缓存。", flush=True)
        acc_years.append(acc_df)
        time_years.append(time_df)

    acc_out = pd.concat(acc_years, ignore_index=True).sort_values(["Year", "Urban_Rural"]).reset_index(drop=True)
    time_out = pd.concat(time_years, ignore_index=True).sort_values(["Year", "Urban_Rural"]).reset_index(drop=True)
    acc_out.to_csv(final_acc, index=False, encoding="utf-8-sig")
    time_out.to_csv(final_time, index=False, encoding="utf-8-sig")
    write_cache_meta(final_acc, final_acc_fp, payload={"derived_group": "Urban_Rural", "kind": "accessibility", "precomputed": True})
    write_cache_meta(final_time, final_time_fp, payload={"derived_group": "Urban_Rural", "kind": "travel_time", "precomputed": True})
    print(f"Urban/Rural accessibility 统计完成：{final_acc}", flush=True)
    print(f"Urban/Rural travel-time 统计完成：{final_time}", flush=True)
    return {"accessibility": final_acc, "travel_time": final_time}


def read_group_stats(group: str, kind: str = "accessibility") -> pd.DataFrame:
    """Return grouped statistics expected by the current manuscript Figure scripts.

    Urban/Rural is intentionally *not* calculated inside a plotting process.  Run
    ``2_2_precompute_urban_rural_stats.py`` once first; Figure 3 then becomes a cheap
    cache read instead of appearing to hang while millions of GURS windows are classified.
    """
    direct = {
        "national": "national",
        "provincial": "provincial",
        "city": "city",
        "county": "county",
        "North_South": "North_South",
        "city_level": "city_level",
    }
    if group in direct:
        return read_stats(direct[group], kind)

    if group not in {"Coastal_Inland", "Urban_Rural"}:
        raise ValueError(f"未知绘图分组：{group}")

    cache, fp, years_to_build, scope, profile = _group_cache_info(group, kind)

    # Public reproduction releases the already-derived grouped tables but not the
    # >1 GB 2_1 matched grid data or GURS rasters used to derive them.  In this
    # explicitly selected mode, an existing released table is a canonical input
    # and therefore should be read directly rather than invalidated because its
    # private upstream fingerprints are unavailable.
    public_reproduction = os.environ.get("NC_PUBLIC_REPRODUCTION", "0").strip().lower() in {"1", "true", "yes", "y"}
    if public_reproduction and cache.exists():
        return pd.read_csv(cache, encoding="utf-8-sig")

    if cache_valid(cache, fp):
        return pd.read_csv(cache, encoding="utf-8-sig")

    if group == "Urban_Rural":
        raise FileNotFoundError(
            "Figure 3 所需 Urban/Rural 统计尚未预计算，或 2_1/GURS 输入已经变化。\n"
            "请先运行：python 2_2_precompute_urban_rural_stats.py\n"
            f"预计算完成后 Figure 3 将直接读取：{cache}"
        )

    # Coastal/Inland does not need raster classification, so its light derived
    # aggregation can still be generated automatically on first use.
    calc = calculate_accessibility_stats if kind == "accessibility" else calculate_travel_time_stats
    rows: list[dict] = []
    for year in years_to_build:
        df = _read_matched_year(int(year), ["pop", "acc", "travel_time", group])
        for value, sub in df.groupby(group, dropna=False):
            rows.append({"Year": int(year), group: value, **calc(sub)})
    out = pd.DataFrame(rows)
    out.to_csv(cache, index=False, encoding="utf-8-sig")
    write_cache_meta(cache, fp, payload={"derived_group": group, "kind": kind})
    return out

def read_inequality_stats(group: str) -> pd.DataFrame:
    """Return formal inequality statistics plus SCV for Figure 3."""
    df = read_group_stats(group, "accessibility").copy()
    mean = pd.to_numeric(df.get("pop_mean"), errors="coerce")
    std = pd.to_numeric(df.get("pop_std"), errors="coerce")
    df["pop_scv"] = np.where(np.isfinite(mean) & (mean != 0), (std / mean) ** 2, np.nan)
    return df


def _normalize_admin_fields(g):
    for c in ["县级码", "地级码", "省级码"]:
        if c in g.columns:
            g[c] = g[c].astype("string").str.replace(r"\.0$", "", regex=True)
    if "县级码" in g.columns:
        g["县级码"] = g["县级码"].str.zfill(6)
    return g


def _read_admin_shp(path, label: str):
    import geopandas as gpd
    if not path.exists():
        raise FileNotFoundError(f"{label}行政区划不存在：{path}")
    g = gpd.read_file(path)
    if g.crs is None:
        raise ValueError(f"{label}行政区划没有 CRS：{path}")
    return _normalize_admin_fields(g.to_crs("EPSG:4326"))


def load_admin_level(level: str):
    """Return fixed province/city/county boundaries for mapped figures.

    Prefer released direct boundary layers when available. Fall back to the
    historical county-based dissolve only if the dedicated layer is absent.
    """
    if level == "county":
        return _read_admin_shp(COUNTY_SHP, "县级").copy()

    county = _read_admin_shp(ADMIN_SHP, "县级")

    if level == "province":
        if PROVINCE_SHP.exists():
            g = _read_admin_shp(PROVINCE_SHP, "省级")


            # merge without mutating the released shapefile schema.
            if "省级" not in g.columns and "省" in g.columns:
                g["省级"] = g["省"].astype("string")
            if "省" not in g.columns and "省级" in g.columns:
                g["省"] = g["省级"].astype("string")
            return g
        out = county.dropna(subset=["省级"]).dissolve(by="省级", as_index=False)
        out["省"] = out["省级"]
        return out

    if level == "city":
        if CITY_SHP.exists():
            return _read_admin_shp(CITY_SHP, "地级")
        key = county["地级"].astype("string")
        muni = county["省级"].isin(["北京市", "上海市", "天津市", "重庆市"])
        key = key.mask(key.isna() & muni, county["省级"].astype("string"))
        tmp = county.copy()
        tmp["地级"] = key
        return tmp.dropna(subset=["地级"]).dissolve(by="地级", as_index=False)
    raise ValueError(level)


def read_ci_year(year: int, columns: list[str] | None = None) -> pd.DataFrame:
    scope, profile = _scope_profile()
    part_dir = CI_MATCHED_ROOT / str(int(year)) / scope / profile / "parts"
    parts = discover_parts(part_dir)
    if not parts:
        raise FileNotFoundError(f"缺少新版 3_1 CI matched 数据：{part_dir}")
    frames = []
    for p in parts:
        available = _parquet_columns(p)
        use = [c for c in (columns or list(available)) if c in available]
        frames.append(pd.read_parquet(p, columns=use))
    return pd.concat(frames, ignore_index=True)


def read_ci_kde_curves(variable: str | None = None) -> pd.DataFrame:
    """Read compact precomputed KDE curves used by the public Figure 4 workflow.

    The private/HPC workflow may derive these curves once from ``3_1_ci_matched``
    via ``tools/build_ci_plot_inputs.py``.  The public repository releases only
    the resulting small table, so Figure 4 does not require the ~1.16 GB matched
    CI grid dataset.
    """
    scope, profile = _scope_profile()
    path = CI_ANALYSIS_ROOT / scope / profile / "plot_inputs" / "kde_curves.csv"
    if not path.exists():
        raise FileNotFoundError(f"缺少 Figure 4 预计算 KDE 曲线：{path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"variable", "year", "x", "density"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"KDE 曲线文件缺少字段 {sorted(missing)}：{path}")
    if variable is not None:
        df = df[df["variable"].astype(str).eq(str(variable))].copy()
    return df


def read_ci_curve_points() -> pd.DataFrame:
    """Read compact precomputed concentration-curve points used by Figure 4.2.

    The file is optional for public reproduction. When absent, Figure 4.2 can
    still redraw CI trend plots from the released 3_2 summary tables but will
    skip concentration-curve panels unless the private 3_1 matched dataset is
    available.
    """
    scope, profile = _scope_profile()
    path = CI_ANALYSIS_ROOT / scope / profile / "plot_inputs" / "ci_curve_points.csv"
    if not path.exists():
        raise FileNotFoundError(f"缺少 Figure 4.2 预计算 concentration curves：{path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"curve_family", "group_name", "year", "x", "y", "ci"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CI curve file missing columns {sorted(missing)}: {path}")
    return df

def hospital_count_csv() -> Path:
    """Locate the historical external-validation table used by Figure 1.5.

    The official-statistics series is not generated by the computational pipeline, so it
    must remain an external input rather than being fabricated from hospital records.
    """
    env = os.environ.get("NC_FIGURE1_HOSPITAL_COUNT_CSV")
    candidates = [Path(env).expanduser() if env else None,
                  figure_dir("Figure 1") / "hospital_count_statistics.csv"]
    for p in candidates:
        if p is not None and p.exists():
            return p
    raise FileNotFoundError(
        "Figure 1.5 需要旧图使用的外部核验表“三甲医院数量统计.csv”。"
        "请把文件放到 result/Figure/figure 1/，或设置 NC_FIGURE1_HOSPITAL_COUNT_CSV。"
        "该表中的 Official Statistics 不能由新版计算结果推测。"
    )


def shapley_plot_table() -> pd.DataFrame:
    """Map exact Shapley output to the plotting-table schema."""
    src = RESULT_ROOT / "shapley_summary.csv"
    if not src.exists():
        alt = RESULT_ROOT / "4_2_shapley_decomposition" / "shapley_summary.csv"
        src = alt if alt.exists() else src
    if not src.exists():
        raise FileNotFoundError(f"缺少新版 Shapley 结果：{src}")
    x = pd.read_csv(src, encoding="utf-8-sig")
    outcome_map = {
        "accessibility": ("accessibility", "pop_median"),
        "gini": ("inequality_acc", "pop_gini"),
        "theil": ("inequality_acc", "pop_theil"),
        "atkinson_05": ("inequality_acc", "pop_atkinson_05"),
    }
    factor_map = {"hospital": "Bed", "population": "Population", "road": "Road"}
    rows = []
    for outcome, (task, indicator) in outcome_map.items():
        sub = x[x["outcome"].astype(str).eq(outcome)].copy()
        if sub.empty:
            continue
        base = float(pd.to_numeric(sub["baseline_value"], errors="coerce").iloc[0])
        target = float(pd.to_numeric(sub["target_value"], errors="coerce").iloc[0])
        base_year = int(pd.to_numeric(sub.get("base_year", pd.Series([2014])), errors="coerce").dropna().iloc[0]) if "base_year" in sub else 2014
        target_year = int(pd.to_numeric(sub.get("target_year", pd.Series([2024])), errors="coerce").dropna().iloc[0]) if "target_year" in sub else 2024
        rows.append({"task": task, "indicator": indicator, "factor": str(base_year), "Contribution_abs": base, "Contribution_pct": np.nan})
        for f in ["hospital", "population", "road"]:
            r = sub[sub["factor"].astype(str).eq(f)]
            if r.empty:
                raise ValueError(f"Shapley {outcome} 缺少 factor={f}")
            rows.append({
                "task": task,
                "indicator": indicator,
                "factor": factor_map[f],
                "Contribution_abs": float(pd.to_numeric(r["contribution_abs"], errors="coerce").iloc[0]),
                "Contribution_pct": float(pd.to_numeric(r["share_pct"], errors="coerce").iloc[0]),
            })
        rows.append({"task": task, "indicator": indicator, "factor": str(target_year), "Contribution_abs": target, "Contribution_pct": np.nan})
    return pd.DataFrame(rows)


def time_threshold_frame(group: str, key: str) -> pd.DataFrame:
    """Return BASE/END-year median travel-time values for valid study units only."""
    df = read_stats(group, "travel_time")
    base = int(min(YEARS)); end = int(max(YEARS))
    a = df[df["Year"].eq(base)][[key, "pop_median"]].rename(columns={"pop_median": "14_time"})
    b = df[df["Year"].eq(end)][[key, "pop_median"]].rename(columns={"pop_median": "24_time"})
    return a.merge(b, on=key, how="inner")

