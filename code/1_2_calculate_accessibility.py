# -*- coding: utf-8 -*-
"""Module utilities for 1 2 calculate accessibility."""


from __future__ import annotations

import json
import math
import re
import subprocess
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

from config import (
    ALLOW_CROSS_PROVINCE_HOSPITALS,
    ANALYSIS_MODE,
    PREPARED_ROOT,
    TRAVEL_TIME_ROOT as TRAVEL_TIME_STAGE_ROOT,
    ACCESSIBILITY_ROOT,
    RUN_POLICY,
    MAX_ANALYSIS_TIME_MIN,
    PROJECT_ROOT,
    SPEED_PROFILES,
    get_run_year,
    is_year_worker,
    run_script_for_all_years,
    ONLY_PROVINCES,
    REQUIRE_COMPLETE_RUN_WHEN_UNFILTERED,
    R_GT100_POLICY,
    R_TOPOLOGY_QC_ENABLED,
    R_TOPOLOGY_SUSPECT_THRESHOLD,
    R_TOPOLOGY_SUSPECT_MAX_REACHABLE_GRIDS,
    service_scope_tag,
)
from utils.cache import (
    build_fingerprint,
    cache_valid,
    json_load as cache_json_load,
    write_cache_meta,
    prepare_stage_directory,
)
from utils.input_signatures import population_accessibility_signature_for_path


# =============================================================================

# =============================================================================

YEAR = get_run_year()

# Shapley / counterfactual component overrides.
# Main analysis leaves them unset and therefore uses YEAR for all components.
ROAD_YEAR = int(os.environ.get("NC_COMPONENT_ROAD_YEAR", YEAR))
POPULATION_YEAR = int(os.environ.get("NC_COMPONENT_POPULATION_YEAR", YEAR))
HOSPITAL_YEAR = int(os.environ.get("NC_COMPONENT_HOSPITAL_YEAR", YEAR))
SCENARIO_LABEL = os.environ.get("NC_COMPONENT_SCENARIO", "").strip()

# 0_1
HOSPITALS_PATH = (
    PREPARED_ROOT / str(HOSPITAL_YEAR) / "hospitals.parquet"
)
POPULATION_PARTS_DIR = (
    PREPARED_ROOT / str(POPULATION_YEAR) / "population_parts"
)
PROVINCES_PATH = PREPARED_ROOT / "common" / "provinces.parquet"


SERVICE_SCOPE = service_scope_tag()
_travel_override = os.environ.get("NC_TRAVEL_TIME_INPUT_ROOT_OVERRIDE", "").strip()
TRAVEL_TIME_INPUT_ROOT = (
    Path(_travel_override).expanduser().resolve()
    if _travel_override
    else TRAVEL_TIME_STAGE_ROOT / str(YEAR) / SERVICE_SCOPE
)

# 1_2
_access_override = os.environ.get("NC_ACCESSIBILITY_OUTPUT_ROOT_OVERRIDE", "").strip()
OUTPUT_ROOT = (
    Path(_access_override).expanduser().resolve()
    if _access_override
    else ACCESSIBILITY_ROOT / str(YEAR) / SERVICE_SCOPE
)


# -------------------------------------------------------------------------

# -------------------------------------------------------------------------

SEARCH_THRESHOLD_MIN = MAX_ANALYSIS_TIME_MIN


R_SCALE = 10000.0

# -------------------------------------------------------------------------
# Targeted OSM topology fallback
# -------------------------------------------------------------------------
#


#
# key = (problematic road year, campus_id)
# value = reliable reference road year
#

OSM_TOPOLOGY_FALLBACK_SCHEMA_VERSION = 1
OSM_TOPOLOGY_FALLBACK_RULES = {
    (2014, "R001773"): 2015,
    (2014, "R001772"): 2015,
    (2014, "R000528"): 2015,
    (2014, "R000792"): 2016,
    (2014, "R000496"): 2016,
    (2015, "R000455"): 2016,
    (2015, "R000448"): 2016,
    (2015, "R000792"): 2016,
    (2015, "R000475"): 2016,
    (2015, "R000502"): 2017,
    (2015, "R000496"): 2016,
    (2016, "R000502"): 2017,
}


OSM_TOPOLOGY_FALLBACK_COUNTERFACTUAL_DYNAMIC = True


_FIXED_ROAD_SENSITIVITY_ACTIVE = (
    ANALYSIS_MODE == "fixed_road_sensitivity"
    and bool(os.environ.get("NC_FIXED_OSM_ANALYSIS_YEAR", "").strip())
)
if _FIXED_ROAD_SENSITIVITY_ACTIVE:
    OSM_TOPOLOGY_FALLBACK_RULES = {}
    OSM_TOPOLOGY_FALLBACK_COUNTERFACTUAL_DYNAMIC = False


# -------------------------------------------------------------------------

# -------------------------------------------------------------------------


TRAVEL_BATCH_ROWS = 2_000_000

PARQUET_COMPRESSION = "zstd"


# >>> TIME_ANOMALY_IDW_PATCH_20260902 >>>


# - nearest_hospital_time_min < 1 min -> repair


TIME_ANOMALY_IDW_ENABLED = True
TIME_ANOMALY_IDW_K = 50
TIME_ANOMALY_IDW_POWER = 2.0
TIME_ANOMALY_IDW_INCLUDE_LT1 = True
TIME_ANOMALY_IDW_LT1_THRESHOLD_MIN = 1.0

TIME_LT1_QC_THRESHOLD_MIN = TIME_ANOMALY_IDW_LT1_THRESHOLD_MIN
TIME_ANOMALY_IDW_CHUNK_SIZE = 100_000
# <<< TIME_ANOMALY_IDW_PATCH_20260902 <<<

# -------------------------------------------------------------------------

# -------------------------------------------------------------------------


OVERWRITE = (RUN_POLICY == "force_rebuild")

# -------------------------------------------------------------------------
# Snap QC
# -------------------------------------------------------------------------

RUN_SNAP_QC = True

SNAP_QC_THRESHOLDS_KM = [0.5, 1.0, 2.0, 5.0]


# =============================================================================

# =============================================================================

PROVINCE_RE = re.compile(
    r"^province_(\d{3})_(.+)\.parquet$"
)


def log(msg: str) -> None:
    now = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    print(f"[{now}] {msg}", flush=True)


def ensure_exists(
    path: Path,
    label: str,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{label}不存在：{path}"
        )


def json_dump(
    obj: dict,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )


def gaussian_decay(
    t_min: np.ndarray,
    threshold_min: float,
) -> np.ndarray:
    """Helper for gaussian_decay."""


    t = np.asarray(
        t_min,
        dtype=np.float64,
    )

    out = np.zeros(
        len(t),
        dtype=np.float64,
    )

    valid = (
        np.isfinite(t)
        & (t >= 0.0)
        & (t <= threshold_min)
    )

    if not np.any(valid):
        return out

    tv = t[valid]

    numerator = (
        np.exp(
            -0.5
            * (tv / threshold_min) ** 2
        )
        - math.exp(-0.5)
    )

    denominator = (
        1.0
        - math.exp(-0.5)
    )

    out[valid] = (
        numerator / denominator
    )


    np.clip(
        out,
        0.0,
        1.0,
        out=out,
    )

    return out


def parse_province_file(
    path: Path,
):
    m = PROVINCE_RE.match(path.name)
    if not m:
        raise ValueError(
            f"无法解析省份文件名：{path.name}"
        )

    return int(m.group(1)), m.group(2)


def scan_matrix_files(
    profile: str,
) -> pd.DataFrame:
    matrix_dir = (
        TRAVEL_TIME_INPUT_ROOT
        / profile
        / "matrix"
    )

    ensure_exists(
        matrix_dir,
        f"{profile} matrix 目录",
    )

    rows = []

    for path in sorted(
        matrix_dir.glob(
            "province_*.parquet"
        )
    ):
        province_id, province_name = (
            parse_province_file(path)
        )

        if (
            ONLY_PROVINCES is not None
            and province_name
            not in set(ONLY_PROVINCES)
        ):
            continue

        nearest_path = (
            TRAVEL_TIME_INPUT_ROOT
            / profile
            / "nearest"
            / path.name
        )
        ensure_exists(
            nearest_path,
            f"{province_name}/{profile} nearest-time",
        )

        rows.append(
            {
                "province_id": province_id,
                "province_name": province_name,
                "matrix_path": path,
                "nearest_path": nearest_path,
            }
        )

    if not rows:
        raise RuntimeError(
            f"{profile} 没找到任何 matrix 文件："
            f"{matrix_dir}"
        )

    return (
        pd.DataFrame(rows)
        .sort_values("province_id")
        .reset_index(drop=True)
    )


def population_path_for(
    province_id: int,
    province_name: str,
) -> Path:
    return (
        POPULATION_PARTS_DIR
        / (
            f"province_{province_id:03d}_"
            f"{province_name}.parquet"
        )
    )


def load_population_sorted(
    path: Path,
) -> pd.DataFrame:
    pop = pd.read_parquet(
        path,
        columns=[
            "grid_id",
            "population",
            "lon",
            "lat",
        ],
    )

    if pop["grid_id"].duplicated().any():
        raise RuntimeError(
            f"人口表存在重复 grid_id：{path}"
        )


    pop = (
        pop.sort_values("grid_id")
        .reset_index(drop=True)
    )

    pop["grid_id"] = (
        pop["grid_id"].astype(np.int64)
    )
    pop["population"] = pd.to_numeric(
        pop["population"],
        errors="coerce",
    ).fillna(0)

    return pop


def locate_grid_positions(
    sorted_grid_ids: np.ndarray,
    query_grid_ids: np.ndarray,
    label: str,
) -> np.ndarray:
    """Helper for locate_grid_positions."""


    q = np.asarray(
        query_grid_ids,
        dtype=np.int64,
    )

    pos = np.searchsorted(
        sorted_grid_ids,
        q,
    )

    bad = (
        (pos >= len(sorted_grid_ids))
    )


    safe_pos = np.minimum(
        pos,
        max(
            len(sorted_grid_ids) - 1,
            0,
        ),
    )

    if len(sorted_grid_ids) == 0:
        raise RuntimeError(
            f"{label}: population grid 为空"
        )

    bad |= (
        sorted_grid_ids[safe_pos]
        != q
    )

    if np.any(bad):
        examples = q[bad][:10].tolist()
        raise RuntimeError(
            f"{label}: travel matrix 中存在 "
            f"population 表找不到的 grid_id，"
            f"示例={examples}"
        )

    return pos.astype(
        np.int64,
        copy=False,
    )


# =============================================================================

# =============================================================================

def repair_time_anomalies_by_idw(
    *,
    lon: np.ndarray,
    lat: np.ndarray,
    population: np.ndarray,
    nearest: np.ndarray,
    accessibility: np.ndarray,
    province_name: str,
    profile: str,
):
    # Repair anomalous nearest-time cells and the same cells' accessibility.
    # repair reason code:
    #   0 = not repaired
    #   1 = non-finite nearest_hospital_time_min
    #   2 = finite nearest_hospital_time_min < threshold
    nearest = np.asarray(nearest, dtype=np.float64)
    accessibility = np.asarray(accessibility, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    population = np.asarray(population, dtype=np.float64)

    if not (
        len(nearest) == len(accessibility) == len(lon) == len(lat) == len(population)
    ):
        raise RuntimeError(f"{province_name}/{profile}: time-IDW arrays length mismatch")

    nearest_new = nearest.copy()
    accessibility_new = accessibility.copy()
    n = len(nearest)
    repair_applied = np.zeros(n, dtype=bool)
    repair_reason = np.zeros(n, dtype=np.uint8)

    if not TIME_ANOMALY_IDW_ENABLED or n == 0:
        return nearest_new, accessibility_new, repair_applied, repair_reason, {
            "idw_enabled": bool(TIME_ANOMALY_IDW_ENABLED),
            "nonfinite_nearest_grid_num": 0,
            "lt1_nearest_grid_num": 0,
            "invalid_grid_num": 0,
            "target_with_coordinates_grid_num": 0,
            "donor_grid_num": 0,
            "repaired_grid_num": 0,
            "invalid_missing_xy_grid_num": 0,
            "invalid_population": 0.0,
            "repaired_population": 0.0,
            "invalid_population_share": 0.0,
            "repaired_population_share": 0.0,
            "zero_access_population_share_before": np.nan,
            "zero_access_population_share_after": np.nan,
        }

    finite_xy = np.isfinite(lon) & np.isfinite(lat)
    nonfinite_nearest = ~np.isfinite(nearest)

    # <1 min values are treated as routing/time anomalies and repaired by IDW.
    lt1 = np.zeros(n, dtype=bool)
    if TIME_ANOMALY_IDW_INCLUDE_LT1:
        lt1 = (
            np.isfinite(nearest)
            & (nearest < TIME_ANOMALY_IDW_LT1_THRESHOLD_MIN)
        )

    invalid = nonfinite_nearest | lt1
    donor = (
        (~invalid)
        & finite_xy
        & np.isfinite(nearest)
        & np.isfinite(accessibility)
    )
    target = invalid & finite_xy

    repair_reason[nonfinite_nearest] = 1
    repair_reason[lt1] = 2

    positive_pop = np.isfinite(population) & (population > 0)
    total_pop = float(np.sum(population[positive_pop], dtype=np.float64))
    invalid_pop = float(np.sum(population[invalid & positive_pop], dtype=np.float64))

    valid_acc_before = (
        positive_pop
        & np.isfinite(accessibility)
        & (accessibility >= 0)
    )
    zero_pop_before = float(
        np.sum(
            population[valid_acc_before & (accessibility == 0)],
            dtype=np.float64,
        )
    )

    n_target = int(target.sum())
    n_donor = int(donor.sum())

    if n_target > 0:
        if n_donor == 0:
            raise RuntimeError(
                f"{province_name}/{profile}: time-IDW has target but no donor"
            )

        donor_xy = np.column_stack([lon[donor], lat[donor]])
        target_xy = np.column_stack([lon[target], lat[target]])
        donor_time = nearest[donor]
        donor_acc = accessibility[donor]

        kk = min(int(TIME_ANOMALY_IDW_K), n_donor)
        tree = cKDTree(donor_xy)

        repaired_time = np.empty(n_target, dtype=np.float64)
        repaired_acc = np.empty(n_target, dtype=np.float64)

        for start in range(0, n_target, int(TIME_ANOMALY_IDW_CHUNK_SIZE)):
            end = min(start + int(TIME_ANOMALY_IDW_CHUNK_SIZE), n_target)
            dist, idx = tree.query(target_xy[start:end], k=kk)

            if kk == 1:
                dist = np.asarray(dist)[:, None]
                idx = np.asarray(idx)[:, None]

            weights = 1.0 / (
                np.power(np.asarray(dist, dtype=np.float64), TIME_ANOMALY_IDW_POWER)
                + 1e-9
            )
            weights = weights / weights.sum(axis=1, keepdims=True)

            repaired_time[start:end] = np.sum(donor_time[idx] * weights, axis=1)
            repaired_acc[start:end] = np.sum(donor_acc[idx] * weights, axis=1)

        nearest_new[target] = repaired_time
        accessibility_new[target] = repaired_acc

        good_repair = (
            target
            & np.isfinite(nearest_new)
            & np.isfinite(accessibility_new)
        )
        repair_applied[good_repair] = True

    repaired_pop = float(
        np.sum(population[repair_applied & positive_pop], dtype=np.float64)
    )

    valid_acc_after = (
        positive_pop
        & np.isfinite(accessibility_new)
        & (accessibility_new >= 0)
    )
    zero_pop_after = float(
        np.sum(
            population[valid_acc_after & (accessibility_new == 0)],
            dtype=np.float64,
        )
    )

    qc = {
        "idw_enabled": True,
        "nonfinite_nearest_grid_num": int(nonfinite_nearest.sum()),
        "lt1_nearest_grid_num": int(lt1.sum()),
        "lt1_repair_enabled": bool(TIME_ANOMALY_IDW_INCLUDE_LT1),
        "invalid_grid_num": int(invalid.sum()),
        "target_with_coordinates_grid_num": n_target,
        "donor_grid_num": n_donor,
        "repaired_grid_num": int(repair_applied.sum()),
        "invalid_missing_xy_grid_num": int((invalid & ~finite_xy).sum()),
        "invalid_population": invalid_pop,
        "repaired_population": repaired_pop,
        "invalid_population_share": (
            invalid_pop / total_pop if total_pop > 0 else np.nan
        ),
        "repaired_population_share": (
            repaired_pop / total_pop if total_pop > 0 else np.nan
        ),
        "zero_access_population_share_before": (
            zero_pop_before / total_pop if total_pop > 0 else np.nan
        ),
        "zero_access_population_share_after": (
            zero_pop_after / total_pop if total_pop > 0 else np.nan
        ),
    }
    return nearest_new, accessibility_new, repair_applied, repair_reason, qc


# =============================================================================

# =============================================================================

def load_hospitals():
    log("读取医院表...")

    hospitals = pd.read_parquet(
        HOSPITALS_PATH
    ).copy()

    required = [
        "source_row",
        "beds_std",
    ]

    missing = [
        x
        for x in required
        if x not in hospitals.columns
    ]

    if missing:
        raise KeyError(
            f"hospitals.parquet 缺少字段："
            f"{missing}"
        )

    hospitals["source_row"] = (
        hospitals["source_row"]
        .astype(np.int32)
    )

    if hospitals[
        "source_row"
    ].duplicated().any():
        raise RuntimeError(
            "hospitals.parquet 的 source_row "
            "必须唯一。"
        )

    hospitals = (
        hospitals
        .sort_values("source_row")
        .reset_index(drop=True)
    )

    max_row = int(
        hospitals["source_row"].max()
    )

    # dense lookup arrays
    n_dense = max_row + 1

    beds = np.full(
        n_dense,
        np.nan,
        dtype=np.float64,
    )

    beds_raw = pd.to_numeric(
        hospitals["beds_std"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    invalid_beds = (~np.isfinite(beds_raw)) | (beds_raw <= 0)
    beds_use = beds_raw.copy()
    beds_use[invalid_beds] = np.nan

    rows = hospitals[
        "source_row"
    ].to_numpy(dtype=np.int64)

    beds[rows] = beds_use

    log(
        f"医院={len(hospitals):,}；"
        f"beds_missing_or_nonpositive={int(invalid_beds.sum())}; "
        "policy=exclude_no_imputation"
    )

    return (
        hospitals,
        beds,
        n_dense,
    )


# =============================================================================

# =============================================================================

def build_s1_input_fingerprint(
    profile: str,
    province_files: pd.DataFrame,
) -> str:
    files = [HOSPITALS_PATH]
    province_keys = []
    population_signatures = []

    for _, row in province_files.iterrows():
        province_id = int(row["province_id"])
        province_name = str(row["province_name"])
        matrix_path = Path(row["matrix_path"])
        pop_path = population_path_for(province_id, province_name)
        # Population magnitudes matter to S1, but a harmless parquet rewrite should not.
        pop_sig = population_accessibility_signature_for_path(pop_path)
        files.append(matrix_path)
        province_keys.append([province_id, province_name])
        population_signatures.append([province_id, province_name, pop_sig])

    return build_fingerprint(
        config={
            "stage": "1_2_s1",
            "year": YEAR,
            "road_year": ROAD_YEAR,
            "population_year": POPULATION_YEAR,
            "hospital_year": HOSPITAL_YEAR,
            "scenario": SCENARIO_LABEL or None,
            "service_scope": SERVICE_SCOPE,
            "allow_cross_province_hospitals": ALLOW_CROSS_PROVINCE_HOSPITALS,
            "profile": profile,
            "search_threshold_min": SEARCH_THRESHOLD_MIN,
            "r_scale": R_SCALE,
            "beds_missing_policy": "exclude_no_imputation",
            "r_gt100_policy": R_GT100_POLICY,
            "r_topology_qc_enabled": R_TOPOLOGY_QC_ENABLED,
            "r_topology_suspect_threshold": R_TOPOLOGY_SUSPECT_THRESHOLD,
            "r_topology_suspect_max_reachable_grids": R_TOPOLOGY_SUSPECT_MAX_REACHABLE_GRIDS,
            "provinces": province_keys,
            "population_accessibility_signatures": population_signatures,
            "osm_topology_fallback_schema_version": OSM_TOPOLOGY_FALLBACK_SCHEMA_VERSION,
            "osm_topology_fallback_reference_years": _fallback_reference_years_for_context(),
        },
        files=files + _fallback_reference_dependency_files(profile, province_files),
    )


def _load_hospital_snap_diagnostics() -> pd.DataFrame:
    """Helper for _load_hospital_snap_diagnostics."""
    snap_dir = TRAVEL_TIME_INPUT_ROOT / "snap_qc"
    router_dir = TRAVEL_TIME_INPUT_ROOT / "router_qc"
    if not snap_dir.exists():
        return pd.DataFrame()

    frames = []
    pattern = re.compile(r"province_(\d+)_(.+)_hospital_snap\.parquet$")
    for path in sorted(snap_dir.glob("province_*_hospital_snap.parquet")):
        m = pattern.match(path.name)
        if not m:
            continue
        routing_pid = int(m.group(1))
        routing_pname = m.group(2)
        snap = pd.read_parquet(path)
        if "hospital_row" not in snap.columns:
            continue
        snap = snap.copy()
        snap["routing_province_id"] = routing_pid
        snap["routing_province_name"] = routing_pname

        comp_path = router_dir / f"province_{routing_pid:03d}_{routing_pname}_component_stats.csv"
        if comp_path.exists():
            comp = pd.read_csv(comp_path)
            comp["component_id"] = pd.to_numeric(comp["component_id"], errors="coerce").astype("Int64")
            base_cols = [c for c in [
                "component_id", "road_length_km", "segment_count", "node_count",
                "raw_grid_count", "raw_hospital_count", "is_tiny", "is_sparse_grid", "is_suspicious",
            ] if c in comp.columns]
            comp = comp[base_cols].copy()

            final = comp.rename(columns={
                "component_id": "component_id",
                "road_length_km": "component_road_length_km",
                "segment_count": "component_segment_count",
                "node_count": "component_node_count",
                "raw_grid_count": "component_raw_grid_count",
                "raw_hospital_count": "component_raw_hospital_count",
                "is_tiny": "component_is_tiny",
                "is_sparse_grid": "component_is_sparse_grid",
                "is_suspicious": "component_is_suspicious",
            })
            snap["component_id"] = pd.to_numeric(snap["component_id"], errors="coerce").astype("Int64")
            snap = snap.merge(final, on="component_id", how="left", validate="many_to_one")

            if "raw_component_id" in snap.columns:
                raw = comp.rename(columns={
                    "component_id": "raw_component_id",
                    "road_length_km": "raw_component_road_length_km",
                    "segment_count": "raw_component_segment_count",
                    "node_count": "raw_component_node_count",
                    "raw_grid_count": "raw_component_grid_count",
                    "raw_hospital_count": "raw_component_hospital_count",
                    "is_tiny": "raw_component_is_tiny",
                    "is_sparse_grid": "raw_component_is_sparse_grid",
                    "is_suspicious": "raw_component_is_suspicious",
                })
                snap["raw_component_id"] = pd.to_numeric(snap["raw_component_id"], errors="coerce").astype("Int64")
                snap = snap.merge(raw, on="raw_component_id", how="left", validate="many_to_one")
        frames.append(snap)

    if not frames:
        return pd.DataFrame()
    all_snap = pd.concat(frames, ignore_index=True, sort=False)
    return all_snap


# =============================================================================
# 3A. Targeted OSM topology fallback
# =============================================================================

_FALLBACK_PLAN_CACHE: dict[str, pd.DataFrame] = {}
_REFERENCE_HOSPITAL_CACHE: dict[int, pd.DataFrame] = {}


def _bool_series(s: pd.Series) -> pd.Series:
    """Helper for _bool_series."""
    if pd.api.types.is_bool_dtype(s.dtype):
        return s.fillna(False).astype(bool)
    num = pd.to_numeric(s, errors="coerce")
    text = s.astype("string").str.strip().str.lower()
    return (num.eq(1) | text.isin({"true", "t", "yes", "y", "1"})).fillna(False)


def _normalise_snap_for_fallback(
    snap: pd.DataFrame,
    hospitals: pd.DataFrame,
) -> pd.DataFrame:
    """Helper for _normalise_snap_for_fallback."""
    if snap.empty:
        return snap.copy()
    z = snap.copy()
    z["hospital_row"] = pd.to_numeric(z["hospital_row"], errors="coerce").astype("Int64")
    z = z[z["hospital_row"].notna()].copy()
    if z["hospital_row"].duplicated().any():
        if "province_id_spatial" in hospitals.columns:
            spatial_map = hospitals.set_index("source_row")["province_id_spatial"].to_dict()
            z["_spatial_pid"] = z["hospital_row"].map(spatial_map)
            z["_province_match"] = pd.to_numeric(z["_spatial_pid"], errors="coerce").eq(
                pd.to_numeric(z["routing_province_id"], errors="coerce")
            )
        else:
            z["_province_match"] = False
        z["_snap_sort"] = pd.to_numeric(z.get("snap_distance_km", np.nan), errors="coerce")
        z = (
            z.sort_values(
                ["hospital_row", "_province_match", "_snap_sort"],
                ascending=[True, False, True],
            )
            .drop_duplicates("hospital_row", keep="first")
            .drop(columns=[c for c in ["_spatial_pid", "_province_match", "_snap_sort"] if c in z.columns])
        )
    else:
        z = z.drop_duplicates("hospital_row", keep="first")
    return z


def _fallback_reference_years_for_context() -> list[int]:
    years = {
        int(ref_year)
        for (bad_road_year, _campus_id), ref_year in OSM_TOPOLOGY_FALLBACK_RULES.items()
        if int(bad_road_year) == ROAD_YEAR
    }
    if (
        OSM_TOPOLOGY_FALLBACK_COUNTERFACTUAL_DYNAMIC
        and bool(SCENARIO_LABEL)
        and HOSPITAL_YEAR > ROAD_YEAR
    ):
        years.add(int(HOSPITAL_YEAR))
    return sorted(years)


def _fallback_reference_dependency_files(
    profile: str,
    province_files: pd.DataFrame,
) -> list[Path]:
    """Helper for _fallback_reference_dependency_files."""
    files: list[Path] = []
    for ref_year in _fallback_reference_years_for_context():
        hosp = PREPARED_ROOT / str(ref_year) / "hospitals.parquet"
        if hosp.exists():
            files.append(hosp)
        ref_matrix_dir = (
            TRAVEL_TIME_STAGE_ROOT
            / str(ref_year)
            / SERVICE_SCOPE
            / profile
            / "matrix"
        )
        for _, row in province_files.iterrows():
            matrix_name = Path(row["matrix_path"]).name
            rp = ref_matrix_dir / matrix_name
            if rp.exists():
                files.append(rp)

    return sorted(set(files), key=lambda x: str(x).lower())


def _load_reference_hospitals(ref_year: int) -> pd.DataFrame:
    ref_year = int(ref_year)
    if ref_year in _REFERENCE_HOSPITAL_CACHE:
        return _REFERENCE_HOSPITAL_CACHE[ref_year]
    path = PREPARED_ROOT / str(ref_year) / "hospitals.parquet"
    ensure_exists(path, f"OSM fallback reference hospitals {ref_year}")
    z = pd.read_parquet(path).copy()
    required = {"source_row", "campus_id"}
    missing = required - set(z.columns)
    if missing:
        raise RuntimeError(
            f"{path} 缺少 OSM fallback 所需字段：{sorted(missing)}"
        )
    z["source_row"] = pd.to_numeric(z["source_row"], errors="raise").astype(np.int32)
    z["campus_id"] = z["campus_id"].astype(str)
    if z["campus_id"].duplicated().any():
        dup = z.loc[z["campus_id"].duplicated(False), ["campus_id", "source_row"]]
        raise RuntimeError(f"{ref_year} reference hospitals 的 campus_id 不唯一：\n{dup.head(20)}")
    _REFERENCE_HOSPITAL_CACHE[ref_year] = z
    return z


def _build_osm_topology_fallback_plan(
    profile: str,
    hospitals: pd.DataFrame,
) -> pd.DataFrame:
    """Helper for _build_osm_topology_fallback_plan."""


    cache_key = f"{profile}|{ROAD_YEAR}|{POPULATION_YEAR}|{HOSPITAL_YEAR}|{SCENARIO_LABEL}|{SERVICE_SCOPE}"
    if cache_key in _FALLBACK_PLAN_CACHE:
        return _FALLBACK_PLAN_CACHE[cache_key].copy()

    if SERVICE_SCOPE != "same_province":

        plan = pd.DataFrame()
        _FALLBACK_PLAN_CACHE[cache_key] = plan
        return plan.copy()

    if "campus_id" not in hospitals.columns:
        raise RuntimeError("启用 OSM topology fallback 时 hospitals.parquet 必须包含 campus_id")

    h = hospitals.copy()
    h["campus_id"] = h["campus_id"].astype(str)
    if h["campus_id"].duplicated().any():
        dup = h.loc[h["campus_id"].duplicated(False), ["campus_id", "source_row"]]
        raise RuntimeError(f"当前 hospitals.parquet 的 campus_id 不唯一：\n{dup.head(20)}")

    rows: list[dict] = []
    planned: set[int] = set()


    for (bad_road_year, campus_id), ref_year in OSM_TOPOLOGY_FALLBACK_RULES.items():
        if int(bad_road_year) != ROAD_YEAR:
            continue
        hit = h[h["campus_id"].eq(str(campus_id))]
        if hit.empty:
            continue
        row = hit.iloc[0]
        source_row = int(row["source_row"])
        rows.append({
            "source_row": source_row,
            "campus_id": str(campus_id),
            "name": str(row.get("name", "")),
            "problem_road_year": ROAD_YEAR,
            "reference_road_year": int(ref_year),
            "reason": "preconfirmed_osm_topology_anomaly",
            "dynamic_counterfactual": False,
        })
        planned.add(source_row)


    if (
        OSM_TOPOLOGY_FALLBACK_COUNTERFACTUAL_DYNAMIC
        and bool(SCENARIO_LABEL)
        and HOSPITAL_YEAR > ROAD_YEAR
    ):
        snap = _normalise_snap_for_fallback(_load_hospital_snap_diagnostics(), h)
        if len(snap) and "component_is_suspicious" in snap.columns:
            suspect = snap[_bool_series(snap["component_is_suspicious"])].copy()
            h_by_row = h.set_index("source_row", drop=False)
            for _, sr in suspect.iterrows():
                source_row = int(sr["hospital_row"])
                if source_row in planned or source_row not in h_by_row.index:
                    continue
                row = h_by_row.loc[source_row]
                rows.append({
                    "source_row": source_row,
                    "campus_id": str(row["campus_id"]),
                    "name": str(row.get("name", "")),
                    "problem_road_year": ROAD_YEAR,
                    "reference_road_year": int(HOSPITAL_YEAR),
                    "reason": "counterfactual_old_road_suspicious_component",
                    "dynamic_counterfactual": True,
                })
                planned.add(source_row)

    plan = pd.DataFrame(rows)
    if len(plan):
        if "province_id_spatial" not in h.columns:
            raise RuntimeError(
                "same_province OSM topology fallback 需要 hospitals.parquet 的 province_id_spatial"
            )
        province_map = h.set_index("source_row")["province_id_spatial"].to_dict()
        plan["province_id"] = pd.to_numeric(
            plan["source_row"].map(province_map), errors="raise"
        ).astype(int)
        plan = plan.sort_values(["province_id", "source_row"]).reset_index(drop=True)

    audit_dir = OUTPUT_ROOT / profile
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / "osm_topology_fallback_plan.csv"
    if len(plan):
        audit = plan.copy()
        audit.insert(0, "scenario", SCENARIO_LABEL or "main")
        audit.insert(1, "road_year", ROAD_YEAR)
        audit.insert(2, "population_year", POPULATION_YEAR)
        audit.insert(3, "hospital_year", HOSPITAL_YEAR)
        audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
        log(
            f"{profile}: OSM topology fallback 计划={len(plan)} 个 hospital-road pairs "
            f"-> {audit_path.name}"
        )
    elif audit_path.exists():
        audit_path.unlink()

    _FALLBACK_PLAN_CACHE[cache_key] = plan.copy()
    return plan


def _grid_membership_mask(sorted_grid_ids: np.ndarray, candidate_ids: np.ndarray) -> np.ndarray:
    if len(candidate_ids) == 0:
        return np.zeros(0, dtype=bool)
    pos = np.searchsorted(sorted_grid_ids, candidate_ids)
    valid = pos < len(sorted_grid_ids)
    out = np.zeros(len(candidate_ids), dtype=bool)
    if np.any(valid):
        idx = np.flatnonzero(valid)
        out[idx] = sorted_grid_ids[pos[idx]] == candidate_ids[idx]
    return out


def _reference_rows_for_fallback(
    *,
    profile: str,
    province_id: int,
    province_name: str,
    current_grid_ids: np.ndarray,
    targets: pd.DataFrame,
    current_matrix_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    """Helper for _reference_rows_for_fallback."""
    out_h: list[np.ndarray] = []
    out_g: list[np.ndarray] = []
    out_t: list[np.ndarray] = []
    row_counts: dict[int, int] = {int(x): 0 for x in targets["source_row"]}

    for ref_year, grp in targets.groupby("reference_road_year", sort=True):
        ref_year = int(ref_year)
        ref_hosp = _load_reference_hospitals(ref_year)
        ref_by_campus = ref_hosp.set_index("campus_id", drop=False)

        ref_to_current: dict[int, int] = {}
        for _, target in grp.iterrows():
            cid = str(target["campus_id"])
            if cid not in ref_by_campus.index:
                raise RuntimeError(
                    f"OSM fallback 无法在 {ref_year} hospitals.parquet 找到 campus_id={cid} "
                    f"({target.get('name','')})。"
                )
            ref_row = ref_by_campus.loc[cid]
            if isinstance(ref_row, pd.DataFrame):
                raise RuntimeError(f"reference campus_id={cid} 非唯一")
            if "province_id_spatial" in ref_row.index:
                ref_pid = pd.to_numeric(pd.Series([ref_row["province_id_spatial"]]), errors="coerce").iloc[0]
                if pd.notna(ref_pid) and int(ref_pid) != int(province_id):
                    raise RuntimeError(
                        f"OSM fallback {cid}: reference province={int(ref_pid)} != current province={province_id}"
                    )
            ref_to_current[int(ref_row["source_row"])] = int(target["source_row"])

        ref_matrix = (
            TRAVEL_TIME_STAGE_ROOT
            / str(ref_year)
            / SERVICE_SCOPE
            / profile
            / "matrix"
            / current_matrix_name
        )
        ensure_exists(ref_matrix, f"OSM fallback reference matrix {ref_year}/{province_name}")
        ref_rows = np.array(sorted(ref_to_current), dtype=np.int32)

        pf = pq.ParquetFile(ref_matrix)
        for batch in pf.iter_batches(
            columns=["hospital_row", "grid_id", "travel_time_min"],
            batch_size=TRAVEL_BATCH_ROWS,
        ):
            h = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
            if len(h) == 0:
                continue
            mask = np.isin(h, ref_rows)
            if not np.any(mask):
                continue
            h = h[mask]
            g = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)[mask]
            t = batch.column(2).to_numpy(zero_copy_only=False).astype(np.float64, copy=False)[mask]

            grid_ok = _grid_membership_mask(current_grid_ids, g)
            if not np.any(grid_ok):
                continue
            h = h[grid_ok]
            g = g[grid_ok]
            t = t[grid_ok]

            mapped = np.fromiter((ref_to_current[int(x)] for x in h), dtype=np.int32, count=len(h))
            for current_row, n in zip(*np.unique(mapped, return_counts=True)):
                row_counts[int(current_row)] += int(n)
            out_h.append(mapped)
            out_g.append(g)
            out_t.append(t)

    missing = [row for row, n in row_counts.items() if n <= 0]
    if missing:
        bad = targets[targets["source_row"].isin(missing)][["source_row", "campus_id", "name", "reference_road_year"]]
        raise RuntimeError(
            "OSM topology fallback 未能从 reference matrix 取得任何 travel rows：\n"
            + bad.to_string(index=False)
        )

    if not out_h:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            row_counts,
        )
    return np.concatenate(out_h), np.concatenate(out_g), np.concatenate(out_t), row_counts


def iter_effective_travel_batches(
    *,
    profile: str,
    province_id: int,
    province_name: str,
    matrix_path: Path,
    grid_ids: np.ndarray,
    hospitals: pd.DataFrame,
):
    """Helper for iter_effective_travel_batches."""


    plan = _build_osm_topology_fallback_plan(profile, hospitals)
    targets = (
        plan[plan["province_id"].eq(int(province_id))].copy()
        if len(plan)
        else pd.DataFrame()
    )
    target_rows = (
        targets["source_row"].to_numpy(dtype=np.int32)
        if len(targets)
        else np.empty(0, dtype=np.int32)
    )


    pf = pq.ParquetFile(matrix_path)
    for batch in pf.iter_batches(
        columns=["hospital_row", "grid_id", "travel_time_min"],
        batch_size=TRAVEL_BATCH_ROWS,
    ):
        h = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
        g = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        t = batch.column(2).to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        if len(target_rows):
            keep = ~np.isin(h, target_rows)
            h, g, t = h[keep], g[keep], t[keep]
        if len(h):
            yield h, g, t

    if not len(targets):
        return


    h, g, t, counts = _reference_rows_for_fallback(
        profile=profile,
        province_id=province_id,
        province_name=province_name,
        current_grid_ids=grid_ids,
        targets=targets,
        current_matrix_name=matrix_path.name,
    )
    log(
        f"{province_name}/{profile}: OSM topology fallback 替换 "
        + ", ".join(
            f"{targets.loc[targets['source_row'].eq(row), 'name'].iloc[0]}:{n:,} rows"
            for row, n in sorted(counts.items())
        )
    )
    if len(h):
        yield h, g, t


def _annotate_fallback_on_hospital_R(out: pd.DataFrame, profile: str) -> pd.DataFrame:
    plan = _build_osm_topology_fallback_plan(profile, out)
    z = out.copy()
    z["osm_topology_fallback_applied"] = False
    z["osm_topology_fallback_reference_road_year"] = pd.Series(pd.NA, index=z.index, dtype="Int64")
    z["osm_topology_fallback_reason"] = ""
    if not len(plan):
        return z
    ref_map = plan.set_index("source_row")["reference_road_year"].to_dict()
    reason_map = plan.set_index("source_row")["reason"].to_dict()
    rows = z["source_row"].astype(int)
    mask = rows.isin(set(ref_map))
    z.loc[mask, "osm_topology_fallback_applied"] = True
    z.loc[mask, "osm_topology_fallback_reference_road_year"] = rows[mask].map(ref_map).astype("Int64")
    z.loc[mask, "osm_topology_fallback_reason"] = rows[mask].map(reason_map).astype(str)
    return z


def build_hospital_anomaly_qc(out: pd.DataFrame, profile: str) -> pd.DataFrame:
    """Helper for build_hospital_anomaly_qc."""


    out_dir = OUTPUT_ROOT / profile
    out_dir.mkdir(parents=True, exist_ok=True)


    diagnostic = out.copy()
    snap = _load_hospital_snap_diagnostics()
    if len(snap):
        snap["hospital_row"] = pd.to_numeric(
            snap["hospital_row"], errors="coerce"
        ).astype("Int64")


        if snap["hospital_row"].duplicated().any():
            if "province_id_spatial" in diagnostic.columns:
                spatial_map = diagnostic.set_index("source_row")[
                    "province_id_spatial"
                ].to_dict()
                snap["_spatial_pid"] = snap["hospital_row"].map(spatial_map)
                snap["_province_match"] = pd.to_numeric(
                    snap["_spatial_pid"], errors="coerce"
                ).eq(snap["routing_province_id"])
            else:
                snap["_province_match"] = False

            snap["_snap_sort"] = pd.to_numeric(
                snap.get("snap_distance_km", np.nan), errors="coerce"
            )
            snap = (
                snap.sort_values(
                    ["hospital_row", "_province_match", "_snap_sort"],
                    ascending=[True, False, True],
                )
                .drop_duplicates("hospital_row", keep="first")
                .drop(
                    columns=[
                        c
                        for c in ["_spatial_pid", "_province_match", "_snap_sort"]
                        if c in snap.columns
                    ]
                )
            )
        else:
            snap = snap.drop_duplicates("hospital_row", keep="first")

        diagnostic = diagnostic.merge(
            snap,
            left_on="source_row",
            right_on="hospital_row",
            how="left",
            validate="one_to_one",
            suffixes=("", "_snap"),
            sort=False,
        )

    raw_r = pd.to_numeric(diagnostic["R_raw_per_10000"], errors="coerce")
    reachable = pd.to_numeric(
        diagnostic["reachable_grid_records"], errors="coerce"
    )
    demand = pd.to_numeric(
        diagnostic["weighted_service_population"], errors="coerce"
    )
    beds = pd.to_numeric(diagnostic["beds_std"], errors="coerce")


    if "component_is_suspicious" in diagnostic.columns:
        s = diagnostic["component_is_suspicious"]
        if pd.api.types.is_bool_dtype(s.dtype):
            component_suspicious = s.fillna(False).astype(bool)
        else:
            s_num = pd.to_numeric(s, errors="coerce")
            s_text = s.astype("string").str.strip().str.lower()
            component_suspicious = (
                s_num.eq(1)
                | s_text.isin({"true", "t", "yes", "y", "1"})
            ).fillna(False)
    else:
        component_suspicious = pd.Series(
            False, index=diagnostic.index, dtype=bool
        )

    extreme_r_low_reach = (
        raw_r.gt(R_TOPOLOGY_SUSPECT_THRESHOLD)
        & reachable.le(R_TOPOLOGY_SUSPECT_MAX_REACHABLE_GRIDS)
    )
    topology_suspect = (
        R_TOPOLOGY_QC_ENABLED
        & (component_suspicious | extreme_r_low_reach)
    )
    zero_demand = beds.gt(0) & (
        ~np.isfinite(demand.to_numpy(dtype=np.float64))
        | demand.le(0).to_numpy()
    )
    very_low_reach = beds.gt(0) & reachable.le(
        R_TOPOLOGY_SUSPECT_MAX_REACHABLE_GRIDS
    )

    diagnostic["qc_component_suspicious"] = component_suspicious.to_numpy(dtype=bool)
    diagnostic["qc_extreme_R_low_reachability"] = extreme_r_low_reach.to_numpy(dtype=bool)
    diagnostic["qc_topology_suspect"] = topology_suspect.to_numpy(dtype=bool)
    diagnostic["qc_zero_service_population"] = zero_demand
    diagnostic["qc_very_low_reachable_grids"] = very_low_reach
    diagnostic["qc_anomaly_any"] = (
        diagnostic["qc_R_gt_100"].fillna(False).astype(bool)
        | diagnostic["qc_topology_suspect"].fillna(False).astype(bool)
        | diagnostic["qc_zero_service_population"].fillna(False).astype(bool)
        | diagnostic["qc_very_low_reachable_grids"].fillna(False).astype(bool)
    )

    reason = np.full(len(diagnostic), "", dtype=object)
    for flag, label in [
        ("qc_R_gt_100", "R_gt_100"),
        ("qc_component_suspicious", "suspicious_OSM_component"),
        ("qc_extreme_R_low_reachability", "extreme_R_and_low_reachability"),
        ("qc_zero_service_population", "zero_service_population"),
        ("qc_very_low_reachable_grids", "very_low_reachable_grids"),
    ]:
        mask = diagnostic[flag].fillna(False).to_numpy(dtype=bool)
        reason[mask] = np.where(
            reason[mask] == "", label, reason[mask] + ";" + label
        )
    diagnostic["qc_anomaly_reason"] = reason


    qc_cols = [
        "qc_component_suspicious",
        "qc_extreme_R_low_reachability",
        "qc_topology_suspect",
        "qc_zero_service_population",
        "qc_very_low_reachable_grids",
        "qc_anomaly_any",
        "qc_anomaly_reason",
    ]
    qc_map = diagnostic.set_index("source_row")[qc_cols]
    for c in qc_cols:
        out[c] = out["source_row"].map(qc_map[c])

    anomaly = diagnostic.loc[diagnostic["qc_anomaly_any"]].copy()
    anomaly.insert(0, "year", YEAR)
    anomaly.insert(1, "profile", profile)
    anomaly.insert(2, "service_scope", SERVICE_SCOPE)

    preferred = [
        "year", "profile", "service_scope",
        "hospital_id", "source_row", "name", "grade", "province_id_spatial", "province_name_spatial",
        "lon", "lat", "beds_std", "weighted_service_population", "reachable_grid_records",
        "weighted_mean_travel_time_min", "R_raw_per_10000", "R_per_10000",
        "qc_R_gt_100", "qc_component_suspicious", "qc_extreme_R_low_reachability",
        "qc_topology_suspect", "qc_zero_service_population",
        "qc_very_low_reachable_grids", "qc_anomaly_any", "qc_anomaly_reason",
        "routing_province_id", "routing_province_name",
        "raw_snap_kind", "raw_snap_distance_km", "raw_component_id",
        "raw_component_road_length_km", "raw_component_segment_count", "raw_component_node_count",
        "raw_component_grid_count", "raw_component_hospital_count", "raw_component_is_tiny",
        "raw_component_is_sparse_grid", "raw_component_is_suspicious",
        "component_rescued", "snap_kind", "snap_distance_km", "component_id",
        "component_road_length_km", "component_segment_count", "component_node_count",
        "component_raw_grid_count", "component_raw_hospital_count", "component_is_tiny",
        "component_is_sparse_grid", "component_is_suspicious",
    ]
    ordered = [c for c in preferred if c in anomaly.columns]
    ordered += [c for c in anomaly.columns if c not in ordered]
    anomaly = anomaly[ordered]

    anomaly_path = out_dir / "hospital_R_anomaly_qc.csv"
    anomaly.to_csv(anomaly_path, index=False, encoding="utf-8-sig")

    topology_path = out_dir / "hospital_R_topology_suspect_qc.csv"
    topology = (
        anomaly.loc[
            anomaly["qc_topology_suspect"].fillna(False).astype(bool)
        ].copy()
        if len(anomaly)
        else anomaly.copy()
    )
    topology.to_csv(topology_path, index=False, encoding="utf-8-sig")


    legacy_fatal = out_dir / "hospital_R_topology_fatal_qc.csv"
    if legacy_fatal.exists():
        legacy_fatal.unlink()

    if len(topology):
        n_component = int(
            topology["qc_component_suspicious"].fillna(False).sum()
        )
        n_extreme = int(
            topology["qc_extreme_R_low_reachability"].fillna(False).sum()
        )
        log(
            f"WARNING {profile}: 检测到 {len(topology)} 家疑似 topology anomaly："
            f"suspicious final component={n_component}，"
            f"extreme R + low reachability={n_extreme}。"
            "仅记录，不停止、不修改 R。"
        )
    if len(anomaly):
        log(f"{profile}: 已记录异常医院 {len(anomaly)} 条 -> {anomaly_path.name}")
    return anomaly

def calculate_s1(
    profile: str,
    province_files: pd.DataFrame,
    hospitals: pd.DataFrame,
    beds_dense: np.ndarray,
    n_hospital_dense: int,
) -> pd.DataFrame:
    """Helper for calculate_s1."""


    if R_GT100_POLICY not in {"keep", "exclude_legacy"}:
        raise ValueError(
            "R_GT100_POLICY 只能是 'keep' 或 'exclude_legacy'，"
            f"当前={R_GT100_POLICY!r}"
        )

    out_dir = OUTPUT_ROOT / profile
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hospital_R.parquet"

    input_fingerprint = build_s1_input_fingerprint(
        profile,
        province_files,
    )

    if (
        not OVERWRITE
        and cache_valid(
            out_path, input_fingerprint,
            event_payload={"stage":"1_2_s1","year":YEAR,"profile":profile},
        )
    ):
        log(
            f"{profile}: hospital_R 缓存指纹一致，直接读取"
        )
        cached = pd.read_parquet(out_path)

        if "qc_R_gt_100" in cached.columns:
            build_hospital_anomaly_qc(cached, profile)
        return cached

    if out_path.exists() and not OVERWRITE:
        log(
            f"{profile}: hospital_R 存在但输入指纹已变化，自动重算。"
            "这会阻止测试子集 R 被正式全国运行误用。"
        )

    log("")
    log("=" * 88)
    log(
        f"S1 计算：{profile}, scope={SERVICE_SCOPE}, "
        f"provinces={len(province_files)}"
    )
    log("=" * 88)

    weighted_population = np.zeros(
        n_hospital_dense,
        dtype=np.float64,
    )
    reachable_grid_records = np.zeros(
        n_hospital_dense,
        dtype=np.int64,
    )
    weighted_time_numerator = np.zeros(
        n_hospital_dense,
        dtype=np.float64,
    )

    for seq, row in province_files.iterrows():
        province_id = int(row["province_id"])
        province_name = str(row["province_name"])
        matrix_path = Path(row["matrix_path"])

        pop_path = population_path_for(
            province_id,
            province_name,
        )
        ensure_exists(
            pop_path,
            f"{province_name} population",
        )

        pop = load_population_sorted(pop_path)
        grid_ids = pop["grid_id"].to_numpy(
            dtype=np.int64
        )
        population = pop["population"].to_numpy(
            dtype=np.float64
        )

        province_rows = 0

        for h, g, t in iter_effective_travel_batches(
            profile=profile,
            province_id=province_id,
            province_name=province_name,
            matrix_path=matrix_path,
            grid_ids=grid_ids,
            hospitals=hospitals,
        ):
            if len(h) == 0:
                continue

            if (
                h.min(initial=0) < 0
                or h.max(initial=0) >= n_hospital_dense
            ):
                raise RuntimeError(
                    f"{province_name}/{profile}: "
                    "hospital_row 超出医院表范围"
                )

            pos = locate_grid_positions(
                grid_ids,
                g,
                f"{province_name}/{profile}",
            )

            w = gaussian_decay(
                t,
                SEARCH_THRESHOLD_MIN,
            )
            contribution = population[pos] * w

            weighted_population += np.bincount(
                h,
                weights=contribution,
                minlength=n_hospital_dense,
            )
            reachable_grid_records += np.bincount(
                h,
                minlength=n_hospital_dense,
            ).astype(np.int64)
            weighted_time_numerator += np.bincount(
                h,
                weights=(population[pos] * w * t),
                minlength=n_hospital_dense,
            )
            province_rows += len(h)

        log(
            f"[{seq + 1}/{len(province_files)}] "
            f"{province_name}: travel rows={province_rows:,}"
        )

    hospital_rows = hospitals[
        "source_row"
    ].to_numpy(dtype=np.int64)
    beds = beds_dense[hospital_rows]
    demand = weighted_population[hospital_rows]

    R_raw = np.full(
        len(hospitals),
        np.nan,
        dtype=np.float64,
    )
    valid = (
        np.isfinite(beds)
        & (beds > 0)
        & np.isfinite(demand)
        & (demand > 0)
    )
    R_raw[valid] = (
        beds[valid] / demand[valid] * R_SCALE
    )


    R = R_raw.copy()
    if R_GT100_POLICY == "exclude_legacy":
        R[np.isfinite(R) & (R > 100)] = np.nan

    mean_weighted_time = np.full(
        len(hospitals),
        np.nan,
        dtype=np.float64,
    )
    valid_time = weighted_population[hospital_rows] > 0
    mean_weighted_time[valid_time] = (
        weighted_time_numerator[
            hospital_rows[valid_time]
        ]
        / weighted_population[
            hospital_rows[valid_time]
        ]
    )

    out = hospitals.copy()
    out["weighted_service_population"] = demand
    out["reachable_grid_records"] = (
        reachable_grid_records[hospital_rows]
    )
    out["weighted_mean_travel_time_min"] = (
        mean_weighted_time
    )
    out["R_raw_per_10000"] = R_raw
    out["R_per_10000"] = R
    out["qc_R_invalid"] = (
        ~np.isfinite(R_raw) | (R_raw <= 0)
    )
    out["qc_R_gt_100"] = (
        np.isfinite(R_raw) & (R_raw > 100)
    )
    out["qc_R_excluded_by_legacy_policy"] = (
        out["qc_R_gt_100"]
        if R_GT100_POLICY == "exclude_legacy"
        else False
    )


    out = _annotate_fallback_on_hospital_R(out, profile)


    anomaly = build_hospital_anomaly_qc(out, profile)


    qc_gt100 = anomaly.loc[anomaly.get("qc_R_gt_100", False).astype(bool)].copy() if len(anomaly) else anomaly.copy()
    qc_gt100.to_csv(
        out_dir / "hospital_R_gt100_qc.csv",
        index=False,
        encoding="utf-8-sig",
    )

    out.to_parquet(
        out_path,
        index=False,
        compression=PARQUET_COMPRESSION,
    )
    write_cache_meta(
        out_path,
        input_fingerprint,
        payload={
            "profile": profile,
            "service_scope": SERVICE_SCOPE,
            "r_gt100_policy": R_GT100_POLICY,
            "r_topology_qc_enabled": R_TOPOLOGY_QC_ENABLED,
            "r_topology_suspect_threshold": R_TOPOLOGY_SUSPECT_THRESHOLD,
            "r_topology_suspect_max_reachable_grids": R_TOPOLOGY_SUSPECT_MAX_REACHABLE_GRIDS,
            "processed_provinces": province_files[
                ["province_id", "province_name"]
            ].to_dict("records"),
        },
    )

    log(
        f"{profile} S1 完成："
        f"有效 R={int(np.isfinite(R).sum()):,}/{len(R):,}；"
        f"R>100={int(out['qc_R_gt_100'].sum()):,}；"
        f"policy={R_GT100_POLICY}"
    )

    return out


# =============================================================================

# =============================================================================

def weighted_percentile(
    data: np.ndarray,
    weights: np.ndarray,
    q: float,
) -> float:
    """Helper for weighted_percentile."""
    data = np.asarray(data, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = (
        np.isfinite(data)
        & np.isfinite(weights)
        & (weights > 0)
    )
    data = data[valid]
    weights = weights[valid]
    if len(data) == 0:
        return np.nan
    sorter = np.argsort(data, kind="mergesort")
    data = data[sorter]
    weights = weights[sorter]
    cumsum = np.cumsum(weights, dtype=np.float64)
    if cumsum[-1] <= 0:
        return np.nan
    return float(
        np.interp(
            q / 100.0 * cumsum[-1],
            cumsum,
            data,
        )
    )


def weighted_std(
    data: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Helper for weighted_std."""
    data = np.asarray(data, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = (
        np.isfinite(data)
        & np.isfinite(weights)
        & (weights > 0)
    )
    data = data[valid]
    weights = weights[valid]
    if len(data) == 0 or weights.sum() <= 0:
        return np.nan
    mean = np.average(data, weights=weights)
    return float(np.sqrt(np.average((data - mean) ** 2, weights=weights)))


def summarize_grid_result(
    out: pd.DataFrame,
    *,
    province_id: int,
    province_name: str,
    profile: str,
    matrix_rows: float = np.nan,
    used_rows: float = np.nan,
    status: str = "ok",
) -> dict:
    population = pd.to_numeric(
        out["population"], errors="coerce"
    ).fillna(0).to_numpy(dtype=np.float64)
    accessibility = pd.to_numeric(
        out["accessibility"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    nearest = pd.to_numeric(
        out["nearest_hospital_time_min"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    n_reachable = pd.to_numeric(
        out["n_reachable_hospitals"], errors="coerce"
    ).fillna(0).to_numpy(dtype=np.int64)

    n_grids = len(out)
    pop_sum = float(np.nansum(population))
    has_access = n_reachable > 0
    has_positive_accessibility_final = (
        np.isfinite(accessibility) & (accessibility > 0)
    )
    has_nearest = (
        np.isfinite(nearest) & (nearest >= 0)
    )
    time_idw_applied = (
        out["time_idw_applied"].fillna(False).to_numpy(dtype=bool)
        if "time_idw_applied" in out.columns
        else np.zeros(len(out), dtype=bool)
    )
    time_lt1_qc_flag = (
        out["time_lt1_qc_flag"].fillna(False).to_numpy(dtype=bool)
        if "time_lt1_qc_flag" in out.columns
        else (
            np.isfinite(nearest)
            & (nearest < TIME_LT1_QC_THRESHOLD_MIN)
        )
    )
    nearest_pop = float(
        population[has_nearest].sum()
    )
    nearest_weighted_numerator = float(
        np.sum(
            nearest[has_nearest]
            * population[has_nearest],
            dtype=np.float64,
        )
    )

    def pop_for(mask):
        return float(population[mask].sum())

    time_0_30 = has_nearest & (nearest <= 30)
    time_30_60 = has_nearest & (nearest > 30) & (nearest <= 60)
    time_60_90 = has_nearest & (nearest > 60) & (nearest <= 90)
    time_lt_60 = has_nearest & (nearest <= 60)
    time_lt_90 = has_nearest & (nearest <= 90)

    return {
        "province_id": province_id,
        "province_name": province_name,
        "profile": profile,
        "n_grids": n_grids,
        "population_sum": pop_sum,
        "matrix_rows": matrix_rows,
        "matrix_rows_used_in_s2": used_rows,
        "n_grids_with_access": int(has_access.sum()),
        "grid_share_with_access": (
            float(has_access.mean())
            if n_grids > 0 else np.nan
        ),
        "population_share_with_access": (
            pop_for(has_access) / pop_sum
            if pop_sum > 0 else np.nan
        ),
        "n_grids_with_positive_accessibility_final": int(
            has_positive_accessibility_final.sum()
        ),
        "population_share_with_positive_accessibility_final": (
            pop_for(has_positive_accessibility_final) / pop_sum
            if pop_sum > 0 else np.nan
        ),
        "time_idw_repaired_grid_num": int(time_idw_applied.sum()),
        "time_idw_repaired_population": pop_for(time_idw_applied),
        "time_idw_repaired_population_share": (
            pop_for(time_idw_applied) / pop_sum
            if pop_sum > 0 else np.nan
        ),
        "time_lt1_qc_grid_num": int(time_lt1_qc_flag.sum()),
        "time_lt1_qc_population": pop_for(time_lt1_qc_flag),
        "time_lt1_qc_population_share": (
            pop_for(time_lt1_qc_flag) / pop_sum
            if pop_sum > 0 else np.nan
        ),
        "population_share_with_nearest_time": (
            nearest_pop / pop_sum
            if pop_sum > 0 else np.nan
        ),
        "mean_accessibility": (
            float(np.nanmean(accessibility))
            if n_grids > 0 else np.nan
        ),
        "population_weighted_accessibility": (
            float(
                np.nansum(accessibility * population)
                / pop_sum
            )
            if pop_sum > 0 else np.nan
        ),
        "population_weighted_accessibility_p25": (
            weighted_percentile(accessibility, population, 25)
            if pop_sum > 0 else np.nan
        ),
        "population_weighted_accessibility_median": (
            weighted_percentile(accessibility, population, 50)
            if pop_sum > 0 else np.nan
        ),
        "population_weighted_accessibility_p75": (
            weighted_percentile(accessibility, population, 75)
            if pop_sum > 0 else np.nan
        ),
        "population_weighted_accessibility_std": (
            weighted_std(accessibility, population)
            if pop_sum > 0 else np.nan
        ),
        "median_accessibility": (
            float(np.nanmedian(accessibility))
            if n_grids > 0 else np.nan
        ),
        "mean_nearest_time_min": (
            float(np.mean(nearest[has_nearest]))
            if np.any(has_nearest) else np.nan
        ),
        "population_weighted_nearest_time_min": (
            nearest_weighted_numerator / nearest_pop
            if nearest_pop > 0 else np.nan
        ),
        "population_weighted_nearest_time_p25_min": (
            weighted_percentile(nearest, population, 25)
            if nearest_pop > 0 else np.nan
        ),
        "population_weighted_nearest_time_median_min": (
            weighted_percentile(nearest, population, 50)
            if nearest_pop > 0 else np.nan
        ),
        "population_weighted_nearest_time_p75_min": (
            weighted_percentile(nearest, population, 75)
            if nearest_pop > 0 else np.nan
        ),
        "population_weighted_nearest_time_std": (
            weighted_std(nearest, population)
            if nearest_pop > 0 else np.nan
        ),
        "nearest_time_population_sum": nearest_pop,
        "nearest_time_weighted_numerator": nearest_weighted_numerator,

        "pop_num_0_30": pop_for(time_0_30),
        "pop_pct_0_30": pop_for(time_0_30) / pop_sum if pop_sum > 0 else np.nan,
        "pop_num_30_60": pop_for(time_30_60),
        "pop_pct_30_60": pop_for(time_30_60) / pop_sum if pop_sum > 0 else np.nan,
        "pop_num_60_90": pop_for(time_60_90),
        "pop_pct_60_90": pop_for(time_60_90) / pop_sum if pop_sum > 0 else np.nan,
        "pop_num_lt_60": pop_for(time_lt_60),
        "pop_pct_lt_60": pop_for(time_lt_60) / pop_sum if pop_sum > 0 else np.nan,
        "pop_num_lt_90": pop_for(time_lt_90),
        "pop_pct_lt_90": pop_for(time_lt_90) / pop_sum if pop_sum > 0 else np.nan,
        "status": status,
    }


def calculate_s2_one_province(
    profile: str,
    province_id: int,
    province_name: str,
    matrix_path: Path,
    nearest_path: Path,
    hospital_R: pd.DataFrame,
):
    out_dir = OUTPUT_ROOT / profile / "grids"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        out_dir
        / f"province_{province_id:03d}_{province_name}.parquet"
    )

    pop_path = population_path_for(
        province_id,
        province_name,
    )
    hospital_r_path = OUTPUT_ROOT / profile / "hospital_R.parquet"
    grid_snap_path = TRAVEL_TIME_INPUT_ROOT / "snap_qc" / f"province_{province_id:03d}_{province_name}_grid_snap.parquet"
    ensure_exists(grid_snap_path, f"{province_name} grid snap QC")

    pop_access_sig = population_accessibility_signature_for_path(pop_path)
    input_fingerprint = build_fingerprint(
        config={
            "stage": "1_2_s2",
            "year": YEAR,
            "road_year": ROAD_YEAR,
            "population_year": POPULATION_YEAR,
            "hospital_year": HOSPITAL_YEAR,
            "scenario": SCENARIO_LABEL or None,
            "service_scope": SERVICE_SCOPE,
            "profile": profile,
            "province_id": province_id,
            "province_name": province_name,
            "search_threshold_min": SEARCH_THRESHOLD_MIN,
            "snap_gate_applied": False,
            "grid_output_schema_version": 6,
            "r_gt100_policy": R_GT100_POLICY,
            "nearest_time_source": "1_1_multi_source_full_network",
            "time_anomaly_idw_enabled": TIME_ANOMALY_IDW_ENABLED,
            "time_anomaly_idw_k": TIME_ANOMALY_IDW_K,
            "time_anomaly_idw_power": TIME_ANOMALY_IDW_POWER,
            "time_anomaly_idw_trigger": "nonfinite_or_lt1_nearest_time",
            "time_anomaly_idw_include_lt1": TIME_ANOMALY_IDW_INCLUDE_LT1,
            "time_anomaly_idw_lt1_threshold_min": TIME_ANOMALY_IDW_LT1_THRESHOLD_MIN,

            "population_accessibility_signature": pop_access_sig,
            "osm_topology_fallback_schema_version": OSM_TOPOLOGY_FALLBACK_SCHEMA_VERSION,
            "osm_topology_fallback_reference_years": _fallback_reference_years_for_context(),
        },
        files=[
            matrix_path,
            nearest_path,
            hospital_r_path,
            grid_snap_path,
            *_fallback_reference_dependency_files(
                profile,
                pd.DataFrame([{
                    "province_id": province_id,
                    "province_name": province_name,
                    "matrix_path": matrix_path,
                }]),
            ),
        ],
    )

    if (
        not OVERWRITE
        and cache_valid(
            out_path, input_fingerprint,
            event_payload={"stage":"1_2_s2","year":YEAR,"profile":profile,"province_id":province_id,"province_name":province_name},
        )
    ):
        out = pd.read_parquet(
            out_path,
            columns=[
                "grid_id",
                "population",
                "accessibility",
                "nearest_hospital_time_min",
                "n_reachable_hospitals",
                "grid_snap_distance_km",
                "grid_snap_kind",
                "time_idw_applied",
                "time_idw_reason_code",
                "time_lt1_qc_flag",
            ],
        )
        return summarize_grid_result(
            out,
            province_id=province_id,
            province_name=province_name,
            profile=profile,
            status="existing_validated",
        )

    if out_path.exists() and not OVERWRITE:
        log(
            f"{province_name}/{profile}: grid 输出缓存失配，自动重算"
        )

    pop = load_population_sorted(pop_path)
    grid_ids = pop["grid_id"].to_numpy(dtype=np.int64)
    population = pop["population"].to_numpy(dtype=np.float64)
    n_grids = len(pop)


    nearest_df = pd.read_parquet(
        nearest_path,
        columns=[
            "grid_id",
            "nearest_hospital_time_min",
        ],
    ).sort_values("grid_id").reset_index(drop=True)

    if (
        len(nearest_df) != n_grids
        or not np.array_equal(
            nearest_df["grid_id"].to_numpy(dtype=np.int64),
            grid_ids,
        )
    ):
        raise RuntimeError(
            f"{province_name}/{profile}: nearest-time 与 population grid_id 不一致"
        )

    nearest = pd.to_numeric(
        nearest_df["nearest_hospital_time_min"],
        errors="coerce",
    ).to_numpy(dtype=np.float32)

    snap_df = pd.read_parquet(grid_snap_path, columns=["grid_id", "snap_kind", "snap_distance_km"]).sort_values("grid_id").reset_index(drop=True)
    if len(snap_df) != n_grids or not np.array_equal(snap_df["grid_id"].to_numpy(dtype=np.int64), grid_ids):
        raise RuntimeError(f"{province_name}/{profile}: grid-snap 与 population grid_id 不一致")
    grid_snap_km = pd.to_numeric(snap_df["snap_distance_km"], errors="coerce").to_numpy(dtype=np.float32)
    grid_snap_kind = pd.to_numeric(snap_df["snap_kind"], errors="coerce").fillna(255).to_numpy(dtype=np.uint8)

    accessibility = np.zeros(
        n_grids,
        dtype=np.float64,
    )
    n_reachable = np.zeros(
        n_grids,
        dtype=np.int32,
    )

    max_hospital_row = int(
        hospital_R["source_row"].max()
    )
    R_dense = np.full(
        max_hospital_row + 1,
        np.nan,
        dtype=np.float64,
    )
    hrows = hospital_R[
        "source_row"
    ].to_numpy(dtype=np.int64)
    R_dense[hrows] = hospital_R[
        "R_per_10000"
    ].to_numpy(dtype=np.float64)

    matrix_rows = 0
    used_rows = 0

    for h, g, t in iter_effective_travel_batches(
        profile=profile,
        province_id=province_id,
        province_name=province_name,
        matrix_path=matrix_path,
        grid_ids=grid_ids,
        hospitals=hospital_R,
    ):
        matrix_rows += len(h)
        if len(h) == 0:
            continue

        pos = locate_grid_positions(
            grid_ids,
            g,
            f"{province_name}/{profile}",
        )

        h_in_range = (
            (h >= 0) & (h < len(R_dense))
        )
        R_batch = np.full(
            len(h), np.nan, dtype=np.float64
        )
        if np.any(h_in_range):
            R_batch[h_in_range] = R_dense[
                h[h_in_range]
            ]

        valid = (
            np.isfinite(R_batch)
            & (R_batch > 0)
            & np.isfinite(t)
            & (t >= 0)
            & (t <= SEARCH_THRESHOLD_MIN)
        )
        if not np.any(valid):
            continue

        pos_v = pos[valid]
        t_v = t[valid]
        R_v = R_batch[valid]
        w = gaussian_decay(
            t_v,
            SEARCH_THRESHOLD_MIN,
        )
        contribution = R_v * w

        accessibility += np.bincount(
            pos_v,
            weights=contribution,
            minlength=n_grids,
        )
        n_reachable += np.bincount(
            pos_v,
            minlength=n_grids,
        ).astype(np.int32)
        used_rows += int(valid.sum())

    # Final province-local IDW repair. This does not alter S1 hospital R
    # or the routed sparse OD matrix; it only repairs the delivered grid surface.
    lon = pop["lon"].to_numpy(dtype=np.float64)
    lat = pop["lat"].to_numpy(dtype=np.float64)

    nearest_raw = np.asarray(nearest, dtype=np.float64).copy()
    accessibility_raw = np.asarray(accessibility, dtype=np.float64).copy()

    (
        nearest,
        accessibility,
        time_idw_applied,
        time_idw_reason_code,
        time_idw_qc,
    ) = repair_time_anomalies_by_idw(
        lon=lon,
        lat=lat,
        population=population,
        nearest=nearest_raw,
        accessibility=accessibility_raw,
        province_name=province_name,
        profile=profile,
    )

    # Preserve a raw-value QC flag even though these cells are now repaired.
    time_lt1_qc_flag = (
        np.isfinite(nearest_raw)
        & (nearest_raw < TIME_ANOMALY_IDW_LT1_THRESHOLD_MIN)
    )

    log(
        f"{province_name}/{profile}: time-IDW "
        f"repaired={time_idw_qc['repaired_grid_num']:,} grids; "
        f"pop={time_idw_qc['repaired_population_share']:.2%}; "
        f"nonfinite_time={time_idw_qc['nonfinite_nearest_grid_num']:,}; "
        f"lt1_time={time_idw_qc['lt1_nearest_grid_num']:,}; "
        f"zero-access pop "
        f"{time_idw_qc['zero_access_population_share_before']:.2%}"
        f" -> "
        f"{time_idw_qc['zero_access_population_share_after']:.2%}"
    )

    out = pd.DataFrame(
        {
            "grid_id": grid_ids,
            "population": population,
            "lon": lon,
            "lat": lat,
            "accessibility": accessibility.astype(np.float32),
            "nearest_hospital_time_min": nearest.astype(np.float32),
            "accessibility_raw": accessibility_raw.astype(np.float32),
            "nearest_hospital_time_min_raw": nearest_raw.astype(np.float32),
            "time_idw_applied": time_idw_applied,
            "time_idw_reason_code": time_idw_reason_code,
            "time_lt1_qc_flag": time_lt1_qc_flag,
            # Raw routed count; IDW does not invent OD routes.
            "n_reachable_hospitals": n_reachable,
            "grid_snap_distance_km": grid_snap_km,
            "grid_snap_kind": grid_snap_kind,
        }
    )

    out.to_parquet(
        out_path,
        index=False,
        compression=PARQUET_COMPRESSION,
    )
    write_cache_meta(
        out_path,
        input_fingerprint,
        payload={
            "province_id": province_id,
            "province_name": province_name,
            "profile": profile,
            "service_scope": SERVICE_SCOPE,
            "time_anomaly_idw": {
                "enabled": TIME_ANOMALY_IDW_ENABLED,
                "k": TIME_ANOMALY_IDW_K,
                "power": TIME_ANOMALY_IDW_POWER,
                "repair_trigger": "nonfinite_or_lt1_nearest_time",
                "include_lt1": TIME_ANOMALY_IDW_INCLUDE_LT1,
                "lt1_threshold_min": TIME_ANOMALY_IDW_LT1_THRESHOLD_MIN,
                "lt1_raw_qc_grid_num": time_idw_qc["lt1_nearest_grid_num"],
                "repaired_grid_num": time_idw_qc["repaired_grid_num"],
                "repaired_population_share": time_idw_qc["repaired_population_share"],
                "reason_codes": {
                    "0": "not repaired",
                    "1": "non-finite nearest_hospital_time_min",
                    "2": "finite nearest_hospital_time_min below lt1 threshold",
                },
            },
        },
    )

    return summarize_grid_result(
        out,
        province_id=province_id,
        province_name=province_name,
        profile=profile,
        matrix_rows=matrix_rows,
        used_rows=used_rows,
        status="ok",
    )


def calculate_s2(
    profile: str,
    province_files: pd.DataFrame,
    hospital_R: pd.DataFrame,
) -> pd.DataFrame:
    log("")
    log("=" * 88)
    log(f"S2 按省计算：{profile}")
    log("=" * 88)

    summaries = []

    for seq, row in province_files.iterrows():
        province_id = int(
            row["province_id"]
        )
        province_name = str(
            row["province_name"]
        )

        t0 = time.time()

        s = calculate_s2_one_province(
            profile=profile,
            province_id=province_id,
            province_name=province_name,
            matrix_path=Path(
                row["matrix_path"]
            ),
            nearest_path=Path(
                row["nearest_path"]
            ),
            hospital_R=hospital_R,
        )

        s["elapsed_seconds"] = (
            time.time() - t0
        )

        summaries.append(s)

        log(
            f"[{seq + 1}/{len(province_files)}] "
            f"{province_name}: "
            f"pop-weighted A="
            f"{s['population_weighted_accessibility']:.6f}, "
            f"population with access="
            f"{s['population_share_with_access']:.2%}, "
            f"{s['elapsed_seconds']:.1f}s"
        )

        pd.DataFrame(
            summaries
        ).to_csv(
            OUTPUT_ROOT
            / profile
            / "province_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    return pd.DataFrame(
        summaries
    )


# =============================================================================
# 5. Snap QC
# =============================================================================

def calculate_snap_qc(
    reference_provinces: pd.DataFrame,
) -> pd.DataFrame:
    """Helper for calculate_snap_qc."""


    if not RUN_SNAP_QC:
        return pd.DataFrame()

    snap_dir = (
        TRAVEL_TIME_INPUT_ROOT
        / "snap_qc"
    )

    if not snap_dir.exists():
        log(
            "snap_qc 目录不存在，"
            "跳过 snap QC"
        )
        return pd.DataFrame()

    log("")
    log("=" * 88)
    log("Snap-distance QC")
    log("=" * 88)

    rows = []

    for _, r in reference_provinces.iterrows():
        province_id = int(
            r["province_id"]
        )
        province_name = str(
            r["province_name"]
        )

        snap_path = (
            snap_dir
            / (
                f"province_{province_id:03d}_"
                f"{province_name}_grid_snap.parquet"
            )
        )

        if not snap_path.exists():
            log(
                f"{province_name}: "
                "缺 grid_snap，跳过"
            )
            continue

        pop_path = population_path_for(
            province_id,
            province_name,
        )

        pop = load_population_sorted(
            pop_path
        )

        snap = pd.read_parquet(
            snap_path,
            columns=[
                "grid_id",
                "snap_distance_km",
            ],
        ).sort_values(
            "grid_id"
        )

        if len(snap) != len(pop):
            raise RuntimeError(
                f"{province_name}: "
                f"snap rows={len(snap):,} != "
                f"population rows={len(pop):,}"
            )

        if not np.array_equal(
            snap["grid_id"].to_numpy(
                dtype=np.int64
            ),
            pop["grid_id"].to_numpy(
                dtype=np.int64
            ),
        ):
            raise RuntimeError(
                f"{province_name}: "
                "snap grid_id 与 population "
                "无法一一对应"
            )

        d = snap[
            "snap_distance_km"
        ].to_numpy(
            dtype=np.float64
        )

        p = pop[
            "population"
        ].to_numpy(
            dtype=np.float64
        )

        p_sum = float(
            p.sum()
        )

        row = {
            "province_id": province_id,
            "province_name": province_name,
            "n_grids": len(pop),
            "population_sum": p_sum,
            "snap_mean_km": float(
                np.mean(d)
            ),
            "snap_median_km": float(
                np.median(d)
            ),
            "snap_p95_km": float(
                np.quantile(
                    d,
                    0.95,
                )
            ),
            "snap_max_km": float(
                np.max(d)
            ),
        }

        for threshold in (
            SNAP_QC_THRESHOLDS_KM
        ):
            mask = d > threshold

            row[
                f"grid_share_snap_gt_{threshold:g}km"
            ] = float(
                mask.mean()
            )

            row[
                f"population_share_snap_gt_{threshold:g}km"
            ] = (
                float(
                    p[mask].sum()
                    / p_sum
                )
                if p_sum > 0
                else np.nan
            )

        rows.append(row)

    out = pd.DataFrame(rows)

    if len(out):
        out.to_csv(
            OUTPUT_ROOT
            / "snap_qc_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    return out


# =============================================================================

# =============================================================================

def build_profile_comparison(
    summaries: dict[str, pd.DataFrame],
):
    if len(SPEED_PROFILES) < 2:
        return pd.DataFrame()

    base_profile = SPEED_PROFILES[0]
    alt_profile = SPEED_PROFILES[1]

    a = summaries[
        base_profile
    ].copy()
    b = summaries[
        alt_profile
    ].copy()

    key = [
        "province_id",
        "province_name",
    ]

    metrics = [
        "population_weighted_accessibility",
        "mean_accessibility",
        "population_share_with_access",
        "population_weighted_nearest_time_min",
        "mean_nearest_time_min",
    ]

    a = a[
        key + metrics
    ].rename(
        columns={
            m: f"{base_profile}_{m}"
            for m in metrics
        }
    )

    b = b[
        key + metrics
    ].rename(
        columns={
            m: f"{alt_profile}_{m}"
            for m in metrics
        }
    )

    out = a.merge(
        b,
        on=key,
        how="outer",
        validate="one_to_one",
    )

    for m in metrics:
        old_col = (
            f"{base_profile}_{m}"
        )
        new_col = (
            f"{alt_profile}_{m}"
        )

        out[
            f"diff_{m}"
        ] = (
            out[new_col]
            - out[old_col]
        )

        denominator = out[
            old_col
        ].replace(
            0,
            np.nan,
        )

        out[
            f"pct_change_{m}"
        ] = (
            out[
                f"diff_{m}"
            ]
            / denominator
            * 100.0
        )

    out.to_csv(
        OUTPUT_ROOT
        / "profile_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return out


# =============================================================================

# =============================================================================

def national_summary(
    profile: str,
    province_summary: pd.DataFrame,
    hospital_R: pd.DataFrame,
) -> dict:
    """Helper for national_summary."""


    p = province_summary[
        "population_sum"
    ].to_numpy(dtype=np.float64)
    total_pop = float(np.nansum(p))

    def weighted_by_total_population(metric: str):
        x = province_summary[metric].to_numpy(dtype=np.float64)
        good = np.isfinite(x) & np.isfinite(p) & (p > 0)
        if not np.any(good):
            return np.nan
        return float(np.sum(x[good] * p[good]) / np.sum(p[good]))

    nearest_pop = float(
        np.nansum(
            province_summary["nearest_time_population_sum"].to_numpy(
                dtype=np.float64
            )
        )
    )
    nearest_numerator = float(
        np.nansum(
            province_summary["nearest_time_weighted_numerator"].to_numpy(
                dtype=np.float64
            )
        )
    )

    R = hospital_R["R_per_10000"].to_numpy(dtype=np.float64)

    row = {
        "profile": profile,
        "service_scope": SERVICE_SCOPE,
        "n_provinces": int(len(province_summary)),
        "population_sum": total_pop,
        "population_weighted_accessibility": weighted_by_total_population(
            "population_weighted_accessibility"
        ),
        "population_share_with_access": weighted_by_total_population(
            "population_share_with_access"
        ),
        "population_share_with_nearest_time": (
            nearest_pop / total_pop if total_pop > 0 else np.nan
        ),
        "population_weighted_nearest_time_min": (
            nearest_numerator / nearest_pop if nearest_pop > 0 else np.nan
        ),
        "nearest_time_population_sum": nearest_pop,
        "n_hospitals": int(len(hospital_R)),
        "n_hospitals_with_valid_R": int(np.isfinite(R).sum()),
        "n_hospitals_R_gt_100_raw": int(
            pd.to_numeric(
                hospital_R.get("R_raw_per_10000", hospital_R["R_per_10000"]),
                errors="coerce",
            ).gt(100).sum()
        ),
        "R_median": (
            float(np.nanmedian(R)) if np.any(np.isfinite(R)) else np.nan
        ),
        "R_mean": (
            float(np.nanmean(R)) if np.any(np.isfinite(R)) else np.nan
        ),
    }


    for suffix in ("0_30", "30_60", "60_90", "lt_60", "lt_90"):
        num_col = f"pop_num_{suffix}"
        if num_col in province_summary.columns:
            num = float(
                np.nansum(province_summary[num_col].to_numpy(dtype=np.float64))
            )
            row[num_col] = num
            row[f"pop_pct_{suffix}"] = (
                num / total_pop if total_pop > 0 else np.nan
            )

    return row


# =============================================================================

# =============================================================================

def main():
    total_t0 = time.time()

    log("=" * 96)
    log("1_2_calculate_accessibility 开始")
    log("=" * 96)
    log(
        f"YEAR={YEAR}, road={ROAD_YEAR}, population={POPULATION_YEAR}, hospital={HOSPITAL_YEAR}, "
        f"scenario={SCENARIO_LABEL or 'main'}, ANALYSIS_MODE={ANALYSIS_MODE}, "
        f"SEARCH_THRESHOLD_MIN={SEARCH_THRESHOLD_MIN}"
    )
    log(
        f"SPEED_PROFILES="
        f"{SPEED_PROFILES}"
    )
    log(
        f"SERVICE_SCOPE={SERVICE_SCOPE}, "
        f"ALLOW_CROSS_PROVINCE_HOSPITALS={ALLOW_CROSS_PROVINCE_HOSPITALS}, "
        f"R_GT100_POLICY={R_GT100_POLICY}"
    )
    log(
        f"R topology QC-only={R_TOPOLOGY_QC_ENABLED} | "
        f"suspect when R_raw>{R_TOPOLOGY_SUSPECT_THRESHOLD:g} & "
        f"reachable_grids<={R_TOPOLOGY_SUSPECT_MAX_REACHABLE_GRIDS}; "
        "never stop / never clip"
    )
    log(
        "OSM topology fallback: targeted travel-time-vector replacement; "
        f"reference_years={_fallback_reference_years_for_context() or 'none'}; "
        "no R interpolation / no R cap / no accessibility winsorization"
    )

    log(
        "time anomaly IDW="
        f"{TIME_ANOMALY_IDW_ENABLED} | "
        f"k={TIME_ANOMALY_IDW_K}, "
        f"power={TIME_ANOMALY_IDW_POWER:g}, "
        "trigger=nonfinite_or_lt1_nearest_time; "
        f"lt1={TIME_ANOMALY_IDW_INCLUDE_LT1} "
        f"(<{TIME_ANOMALY_IDW_LT1_THRESHOLD_MIN:g} min)"
    )
    ensure_exists(
        HOSPITALS_PATH,
        "hospitals.parquet",
    )
    ensure_exists(
        POPULATION_PARTS_DIR,
        "population_parts",
    )
    ensure_exists(
        TRAVEL_TIME_INPUT_ROOT,
        "travel_time root",
    )
    travel_stage_manifest = TRAVEL_TIME_INPUT_ROOT / "_stage_input.json"
    ensure_exists(travel_stage_manifest, "1_1 stage manifest")

    population_inputs = sorted(POPULATION_PARTS_DIR.glob("province_*.parquet"))
    if not population_inputs:
        raise FileNotFoundError(f"人口分片目录为空：{POPULATION_PARTS_DIR}")

    population_signature_rows = []
    for p in population_inputs:
        population_signature_rows.append([p.name, population_accessibility_signature_for_path(p)])

    stage_fingerprint = build_fingerprint(
        config={
            "stage": "1_2_calculate_accessibility",
            "year": YEAR,
            "road_year": ROAD_YEAR,
            "population_year": POPULATION_YEAR,
            "hospital_year": HOSPITAL_YEAR,
            "scenario": SCENARIO_LABEL or None,
            "service_scope": SERVICE_SCOPE,
            "speed_profiles": SPEED_PROFILES,
            "search_threshold_min": SEARCH_THRESHOLD_MIN,
            "R_scale": R_SCALE,
            "R_gt100_policy": R_GT100_POLICY,
            "r_topology_qc_enabled": R_TOPOLOGY_QC_ENABLED,
            "r_topology_suspect_threshold": R_TOPOLOGY_SUSPECT_THRESHOLD,
            "r_topology_suspect_max_reachable_grids": R_TOPOLOGY_SUSPECT_MAX_REACHABLE_GRIDS,
            "beds_missing_policy": "exclude_no_imputation",
            "snap_gate_applied": False,
            "time_anomaly_idw_enabled": TIME_ANOMALY_IDW_ENABLED,
            "time_anomaly_idw_k": TIME_ANOMALY_IDW_K,
            "time_anomaly_idw_power": TIME_ANOMALY_IDW_POWER,
            "time_anomaly_idw_trigger": "nonfinite_or_lt1_nearest_time",
            "time_anomaly_idw_include_lt1": TIME_ANOMALY_IDW_INCLUDE_LT1,
            "time_anomaly_idw_lt1_threshold_min": TIME_ANOMALY_IDW_LT1_THRESHOLD_MIN,
            "only_provinces": ONLY_PROVINCES,
            "population_accessibility_signatures": population_signature_rows,
            "osm_topology_fallback_schema_version": OSM_TOPOLOGY_FALLBACK_SCHEMA_VERSION,
            "osm_topology_fallback_reference_years": _fallback_reference_years_for_context(),
            "osm_topology_fallback_counterfactual_dynamic": OSM_TOPOLOGY_FALLBACK_COUNTERFACTUAL_DYNAMIC,
        },
        files=[
            HOSPITALS_PATH,
            PROVINCES_PATH,
            travel_stage_manifest,
            Path(__file__).resolve(),
        ],
    )
    stage_status = prepare_stage_directory(
        OUTPUT_ROOT,
        stage_fingerprint,
        run_policy=RUN_POLICY,
        payload={
            "stage": "1_2_calculate_accessibility",
            "year": YEAR,
            "road_year": ROAD_YEAR,
            "population_year": POPULATION_YEAR,
            "hospital_year": HOSPITAL_YEAR,
            "scenario": SCENARIO_LABEL or None,
        },
    )
    log(f"阶段目录状态：{stage_status}")

    (
        hospitals,
        beds_dense,
        n_hospital_dense,
    ) = load_hospitals()

    # -------------------------------------------------------------


    # -------------------------------------------------------------

    matrix_files = {
        profile: scan_matrix_files(
            profile
        )
        for profile in SPEED_PROFILES
    }


    if ONLY_PROVINCES is None and REQUIRE_COMPLETE_RUN_WHEN_UNFILTERED:
        expected_df = pd.read_parquet(
            PROVINCES_PATH,
            columns=["province_id", "province_name"],
        )
        expected = set(
            zip(
                expected_df["province_id"].astype(int),
                expected_df["province_name"].astype(str),
            )
        )
        for profile, files in matrix_files.items():
            actual = set(
                zip(
                    files["province_id"].astype(int),
                    files["province_name"].astype(str),
                )
            )
            if actual != expected:
                raise RuntimeError(
                    f"1_2/{profile} 只发现部分省份，拒绝作为正式全国运行。"
                    "测试请显式设置 ONLY_PROVINCES；正式运行请先补齐 0_1、0_2 路网和 1_1。"
                    f"\nmissing={sorted(expected - actual)}"
                    f"\nextra={sorted(actual - expected)}"
                )


    if len(SPEED_PROFILES) >= 2:
        reference = set(
            zip(
                matrix_files[
                    SPEED_PROFILES[0]
                ]["province_id"],
                matrix_files[
                    SPEED_PROFILES[0]
                ]["province_name"],
            )
        )

        for profile in SPEED_PROFILES[1:]:
            current = set(
                zip(
                    matrix_files[
                        profile
                    ]["province_id"],
                    matrix_files[
                        profile
                    ]["province_name"],
                )
            )

            if current != reference:
                only_ref = sorted(
                    reference - current
                )
                only_cur = sorted(
                    current - reference
                )

                raise RuntimeError(
                    "不同 speed profile 的省份不一致。\n"
                    f"{SPEED_PROFILES[0]} only="
                    f"{only_ref}\n"
                    f"{profile} only={only_cur}"
                )

    summaries = {}
    hospital_Rs = {}
    national_rows = []

    for profile in SPEED_PROFILES:
        province_files = (
            matrix_files[profile]
        )

        log("")
        log(
            f"{profile}: "
            f"发现 {len(province_files)} 个省级 matrix"
        )

        # S1
        hospital_R = calculate_s1(
            profile=profile,
            province_files=province_files,
            hospitals=hospitals,
            beds_dense=beds_dense,
            n_hospital_dense=n_hospital_dense,
        )

        hospital_Rs[profile] = (
            hospital_R
        )

        # S2
        province_summary = calculate_s2(
            profile=profile,
            province_files=province_files,
            hospital_R=hospital_R,
        )

        summaries[profile] = (
            province_summary
        )

        national_rows.append(
            national_summary(
                profile,
                province_summary,
                hospital_R,
            )
        )

    # -------------------------------------------------------------

    # -------------------------------------------------------------

    comparison = (
        build_profile_comparison(
            summaries
        )
    )

    # -------------------------------------------------------------

    # -------------------------------------------------------------

    reference_provinces = (
        matrix_files[
            SPEED_PROFILES[0]
        ]
    )

    snap_qc = calculate_snap_qc(
        reference_provinces
    )

    # -------------------------------------------------------------

    # -------------------------------------------------------------

    national_df = pd.DataFrame(
        national_rows
    )

    national_df.to_csv(
        OUTPUT_ROOT
        / "national_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------

    metadata = {
        "year": YEAR,
        "road_year": ROAD_YEAR,
        "population_year": POPULATION_YEAR,
        "hospital_year": HOSPITAL_YEAR,
        "scenario": SCENARIO_LABEL or None,
        "analysis_mode": ANALYSIS_MODE,
        "created_at": (
            datetime.now().isoformat()
        ),
        "search_threshold_min": (
            SEARCH_THRESHOLD_MIN
        ),
        "R_scale": R_SCALE,
        "beds_missing_policy": "exclude_no_imputation",
        "decay_function": (
            "(exp(-0.5*(t/T)^2)-exp(-0.5))/"
            "(1-exp(-0.5)); t>T => 0"
        ),
        "service_scope": SERVICE_SCOPE,
        "allow_cross_province_hospitals": ALLOW_CROSS_PROVINCE_HOSPITALS,
        "R_gt100_policy": R_GT100_POLICY,
        "R_topology_qc_mode": "flag_only_never_stop_never_clip",
        "R_topology_suspect_threshold": R_TOPOLOGY_SUSPECT_THRESHOLD,
        "R_topology_suspect_max_reachable_grids": R_TOPOLOGY_SUSPECT_MAX_REACHABLE_GRIDS,
        "S1_scope": (
            "same-province demand for each hospital"
            if not ALLOW_CROSS_PROVINCE_HOSPITALS
            else "national demand accumulated across all province matrices"
        ),
        "S2_scope": (
            "province grid output; hospital set follows service_scope"
        ),
        "nearest_hospital_time_semantics": (
            "full-network nearest routed hospital time; the current search threshold "
            "is applied to Ga2SFCA contributions and the sparse OD matrix cutoff; "
            "non-finite and <1-minute nearest-time cells in the delivered grid surface "
            "are repaired by province-local IDW while raw values are retained"
        ),
        "time_anomaly_idw": {
            "enabled": TIME_ANOMALY_IDW_ENABLED,
            "processing_unit": "province",
            "repair_trigger": "nonfinite_or_lt1_nearest_time",
            "k": TIME_ANOMALY_IDW_K,
            "power": TIME_ANOMALY_IDW_POWER,
            "accessibility_zero_alone_triggers_repair": False,
            "variables_repaired": [
                "nearest_hospital_time_min",
                "accessibility",
            ],
            "include_lt1": TIME_ANOMALY_IDW_INCLUDE_LT1,
            "lt1_threshold_min": TIME_ANOMALY_IDW_LT1_THRESHOLD_MIN,
            "lt1_qc_flag_semantics": "raw nearest time below lt1 threshold before repair",
            "raw_columns_retained": [
                "nearest_hospital_time_min_raw",
                "accessibility_raw",
            ],
            "repair_flag_column": "time_idw_applied",
            "repair_reason_code_column": "time_idw_reason_code",
            "lt1_qc_flag_column": "time_lt1_qc_flag",
        },
        "speed_profiles": (
            SPEED_PROFILES
        ),
        "snap_filter_applied": False,
        "snap_method": "street_segment_projection_except_motorway_node_only",
        "motorway_link_edge_snap_allowed": True,
        "snap_filter_note": "snap distance is reported only for OSM coverage QC and never excludes a grid or hospital",
        "snap_qc_thresholds_km": (
            SNAP_QC_THRESHOLDS_KM
            if RUN_SNAP_QC
            else None
        ),
        "n_provinces": {
            p: int(
                len(matrix_files[p])
            )
            for p in SPEED_PROFILES
        },
        "elapsed_seconds": (
            time.time() - total_t0
        ),
    }

    json_dump(
        metadata,
        OUTPUT_ROOT
        / "accessibility_metadata.json",
    )

    log("")
    log("=" * 96)
    log("1_2 全部完成")
    log(
        f"总耗时："
        f"{metadata['elapsed_seconds'] / 60:.1f} min"
    )
    log(
        f"输出目录：{OUTPUT_ROOT}"
    )

    if len(comparison):
        log(
            "已输出 legacy vs "
            "chn_osm_default 比较："
            "profile_comparison.csv"
        )

    if len(snap_qc):
        log(
            "已输出 OSM snap-gap QC："
            "snap_qc_summary.csv"
        )

    log("=" * 96)


if __name__ == "__main__":
    if is_year_worker():
        main()
    else:
        run_script_for_all_years(__file__)
