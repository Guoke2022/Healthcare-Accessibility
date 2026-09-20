# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import (
    DATA_ROOT, PROJECT_ROOT, PREPARED_ROOT, ROAD_GRAPH_ROOT, RUN_POLICY,
    ANALYSIS_MODE, MAX_ANALYSIS_TIME_MIN, ROUTING_BUFFER_KM, get_run_year, is_year_worker, run_script_for_all_years,
    get_osm_snapshot_year, get_osm_snapshot_date, get_osm_snapshot_path,
    ONLY_PROVINCES, REQUIRE_COMPLETE_RUN_WHEN_UNFILTERED,
)
from utils.cache import build_fingerprint, cache_valid, write_cache_meta, prepare_stage_directory
from utils.executables import resolve_cargo, resolve_osmium

YEAR = get_run_year()
OSM_SNAPSHOT_YEAR = get_osm_snapshot_year(YEAR)
OSM_SNAPSHOT_DATE = get_osm_snapshot_date(YEAR)
OSM_SOURCE_PBF = get_osm_snapshot_path(YEAR)
PROVINCES_PATH = PREPARED_ROOT / "common" / "provinces.parquet"
ROUTING_ROOT = ROAD_GRAPH_ROOT / str(YEAR)
ROAD_GRAPH_SUMMARY_PATH = ROUTING_ROOT / "road_graph_summary.csv"
ROAD_GRAPH_METADATA_PATH = ROUTING_ROOT / "road_graph_metadata.json"
ROADS_ONLY_DIR = ROUTING_ROOT / "source"
PROVINCE_PBF_DIR = ROUTING_ROOT / "pbf"
FMI_DIR = ROUTING_ROOT / "fmi"
METADATA_DIR = ROUTING_ROOT / "metadata"

CREATE_ROADS_ONLY_PBF = True
EXTRA_ROUTING_SAFETY_KM = float(os.environ.get("NC_EXTRA_ROUTING_SAFETY_KM", "0"))
if EXTRA_ROUTING_SAFETY_KM < 0:
    raise ValueError("NC_EXTRA_ROUTING_SAFETY_KM 不能 < 0")
KM_PER_DEG_LAT_CONSERVATIVE = 110.0
KM_PER_DEG_LON_EQUATOR_CONSERVATIVE = 111.0
OVERWRITE_ROADS_ONLY = RUN_POLICY == "force_rebuild"
OVERWRITE_PROVINCE_PBF = RUN_POLICY == "force_rebuild"
BUILD_FMI = False
RUST_PROJECT_DIR = PROJECT_ROOT / "osm_batch_router_v2"
OSM_CH_PRE_PACKAGE = "osm_ch_pre"
OVERWRITE_FMI = RUN_POLICY == "force_rebuild"

def log(msg: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label}不存在：{path}")


def safe_filename(name: str) -> str:
    s = re.sub(r'[\\/:*?"<>|]+', "_", str(name))
    return s.strip().replace(" ", "_")


def json_load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def json_dump(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def run_command(cmd: list[str], cwd: Path | None = None) -> None:

    log("执行：" + " ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd))

    result = subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd) if cwd is not None else None,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"命令执行失败，returncode={result.returncode}\n"
            + " ".join(str(x) for x in cmd)
        )


# =============================================================================

# =============================================================================

def expand_bbox_conservatively(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    buffer_km: float,
):


    lat_pad = buffer_km / KM_PER_DEG_LAT_CONSERVATIVE

    out_min_lat = max(-90.0, min_lat - lat_pad)
    out_max_lat = min(90.0, max_lat + lat_pad)

    max_abs_lat = max(
        abs(out_min_lat),
        abs(out_max_lat),
    )

    cos_lat = math.cos(math.radians(max_abs_lat))

    if cos_lat < 1e-6:
        lon_pad = 180.0
    else:
        lon_pad = (
            buffer_km
            / (
                KM_PER_DEG_LON_EQUATOR_CONSERVATIVE
                * cos_lat
            )
        )

    out_min_lon = max(-180.0, min_lon - lon_pad)
    out_max_lon = min(180.0, max_lon + lon_pad)

    return (
        out_min_lon,
        out_min_lat,
        out_max_lon,
        out_max_lat,
    )


def bbox_to_osmium_string(bbox) -> str:
    min_lon, min_lat, max_lon, max_lat = bbox

    return (
        f"{min_lon:.8f},"
        f"{min_lat:.8f},"
        f"{max_lon:.8f},"
        f"{max_lat:.8f}"
    )


# =============================================================================

# =============================================================================

def prepare_roads_only_pbf(osmium_exe: str) -> Path:

    ROADS_ONLY_DIR.mkdir(parents=True, exist_ok=True)

    stem = OSM_SOURCE_PBF.name
    if stem.endswith(".osm.pbf"):
        base = stem[:-8]
    elif stem.endswith(".pbf"):
        base = stem[:-4]
    else:
        base = OSM_SOURCE_PBF.stem

    out_path = (
        ROADS_ONLY_DIR
        / f"{base}_highways.osm.pbf"
    )

    if not CREATE_ROADS_ONLY_PBF:
        log("CREATE_ROADS_ONLY_PBF=False，直接使用原始全国 PBF。")
        return OSM_SOURCE_PBF

    input_fingerprint = build_fingerprint(
        config={
            "stage": "0_2_roads_only",
            "year": YEAR,
            "create_roads_only_pbf": CREATE_ROADS_ONLY_PBF,
        },
        files=[OSM_SOURCE_PBF],
    )

    if (
        not OVERWRITE_ROADS_ONLY
        and cache_valid(
            out_path, input_fingerprint,
            event_payload={"stage":"0_2_roads_only","year":YEAR},
        )
    ):
        log(
            "全国 highway-only PBF 缓存指纹一致，跳过："
            f"{out_path} "
            f"({out_path.stat().st_size / 1024**3:.2f} GB)"
        )
        return out_path

    if out_path.exists() and not OVERWRITE_ROADS_ONLY:
        log("全国 highway-only PBF 缓存失配，自动重建。")

    if out_path.exists():
        out_path.unlink()

    log("")
    log("=" * 88)
    log("Step 1/2：全国 OSM 过滤为 highway-only PBF")
    log("=" * 88)

    t0 = time.time()

    cmd = [
        osmium_exe,
        "tags-filter",
        str(OSM_SOURCE_PBF),
        "w/highway",
        "-o",
        str(out_path),
        "--overwrite",
    ]

    run_command(cmd)

    elapsed = time.time() - t0

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(
            f"highway-only PBF 没有正确生成：{out_path}"
        )

    log(
        f"highway-only 完成："
        f"{out_path.stat().st_size / 1024**3:.2f} GB，"
        f"耗时={elapsed / 60:.1f} min"
    )
    write_cache_meta(
        out_path, input_fingerprint,
        payload={"source_osm_pbf": str(OSM_SOURCE_PBF)},
    )

    return out_path


# =============================================================================

# =============================================================================

def extract_province_pbf(
    osmium_exe: str,
    source_pbf: Path,
    province_id: int,
    province_name: str,
    routing_bbox,
    input_fingerprint: str,
):
    PROVINCE_PBF_DIR.mkdir(parents=True, exist_ok=True)

    out_path = (
        PROVINCE_PBF_DIR
        / f"province_{province_id:03d}_{safe_filename(province_name)}_roads.osm.pbf"
    )

    if (
        not OVERWRITE_PROVINCE_PBF
        and cache_valid(
            out_path, input_fingerprint,
            event_payload={"stage":"0_2_province_pbf","year":YEAR,"province_id":province_id,"province_name":province_name},
        )
    ):
        return out_path, 0.0, "existing_validated"

    if out_path.exists() and not OVERWRITE_PROVINCE_PBF:
        log(f"{province_name}: 省级 PBF 缓存失配，自动重建")

    if out_path.exists():
        out_path.unlink()

    t0 = time.time()

    bbox_str = bbox_to_osmium_string(routing_bbox)

    cmd = [
        osmium_exe,
        "extract",
        "-b",
        bbox_str,
        "-s",
        "complete_ways",
        "-o",
        str(out_path),
        "--overwrite",
        str(source_pbf),
    ]

    run_command(cmd)

    elapsed = time.time() - t0

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(
            f"{province_name} 路网 PBF 没有正确生成：{out_path}"
        )

    write_cache_meta(
        out_path, input_fingerprint,
        payload={
            "province_id": province_id,
            "province_name": province_name,
            "routing_bbox": list(routing_bbox),
        },
    )
    return out_path, elapsed, "ok"


# =============================================================================

# =============================================================================

def build_fmi_if_requested(
    pbf_path: Path,
    province_id: int,
    province_name: str,
):
    if not BUILD_FMI:
        return None, 0.0, "not_requested"

    ensure_exists(
        RUST_PROJECT_DIR / "Cargo.toml",
        "Rust workspace Cargo.toml",
    )

    cargo_exe = resolve_cargo(PROJECT_ROOT)

    FMI_DIR.mkdir(parents=True, exist_ok=True)

    final_fmi = (
        FMI_DIR
        / f"province_{province_id:03d}_{safe_filename(province_name)}_roads.osm.pbf.fmi"
    )

    rust_sources = [RUST_PROJECT_DIR / "Cargo.toml"]
    lock = RUST_PROJECT_DIR / "Cargo.lock"
    if lock.exists():
        rust_sources.append(lock)
    src_root = RUST_PROJECT_DIR / "src"
    if src_root.exists():
        rust_sources.extend(sorted(src_root.rglob("*.rs")))
    fmi_fp = build_fingerprint(
        config={
            "stage":"0_2_optional_fmi",
            "year":YEAR,
            "province_id":province_id,
            "province_name":province_name,
            "package":OSM_CH_PRE_PACKAGE,
        },
        files=[pbf_path, *rust_sources],
    )
    if not OVERWRITE_FMI and cache_valid(
        final_fmi, fmi_fp,
        event_payload={"stage":"0_2_optional_fmi","year":YEAR,"province_id":province_id,"province_name":province_name},
    ):
        return final_fmi, 0.0, "existing_validated"


    generated_fmi = Path(str(pbf_path) + ".fmi")

    if generated_fmi.exists():
        generated_fmi.unlink()

    t0 = time.time()

    cmd = [
        cargo_exe,
        "run",
        "--release",
        "-p",
        OSM_CH_PRE_PACKAGE,
        str(pbf_path),
    ]

    run_command(
        cmd,
        cwd=RUST_PROJECT_DIR,
    )

    elapsed = time.time() - t0

    if not generated_fmi.exists():
        raise RuntimeError(
            f"osm_ch_pre 执行结束但未找到预期 FMI："
            f"{generated_fmi}"
        )

    if final_fmi.exists():
        final_fmi.unlink()

    shutil.move(
        str(generated_fmi),
        str(final_fmi),
    )
    write_cache_meta(
        final_fmi, fmi_fp,
        payload={"province_id":province_id,"province_name":province_name,"package":OSM_CH_PRE_PACKAGE},
    )

    return final_fmi, elapsed, "ok"


# =============================================================================

# =============================================================================

def main():
    total_t0 = time.time()
    log("=" * 88)
    log("0_2_prepare_road_graph 开始")
    log(f"ANALYSIS_YEAR={YEAR} | OSM_SNAPSHOT={OSM_SNAPSHOT_DATE} | source={OSM_SOURCE_PBF.name}")
    log(f"ANALYSIS_MODE={ANALYSIS_MODE} | SEARCH_THRESHOLD_MIN={MAX_ANALYSIS_TIME_MIN:g}")
    log(f"ROUTING_BUFFER_KM={ROUTING_BUFFER_KM:.3f} | EXTRA={EXTRA_ROUTING_SAFETY_KM:.3f}")
    log("无 OD candidate 阶段；无 snap gate；路网 buffer 不含 +1/+1 km。")
    log("=" * 88)

    ensure_exists(OSM_SOURCE_PBF, "中国 OSM PBF")
    ensure_exists(PROVINCES_PATH, "省界标准表")
    ROUTING_ROOT.mkdir(parents=True, exist_ok=True)

    stage_fingerprint = build_fingerprint(
        config={
            "stage": "0_2_prepare_road_graph",
            "year": YEAR,
            "analysis_year": YEAR,
            "osm_snapshot_year": OSM_SNAPSHOT_YEAR,
            "osm_snapshot_date": OSM_SNAPSHOT_DATE,
            "routing_buffer_km": ROUTING_BUFFER_KM,
            "extra_routing_safety_km": EXTRA_ROUTING_SAFETY_KM,
            "create_roads_only_pbf": CREATE_ROADS_ONLY_PBF,
            "build_fmi": BUILD_FMI,
            "only_provinces": ONLY_PROVINCES,
        },
        files=[OSM_SOURCE_PBF, PROVINCES_PATH, Path(__file__).resolve()],
    )
    stage_status = prepare_stage_directory(
        ROUTING_ROOT, stage_fingerprint, run_policy=RUN_POLICY,
        payload={
            "stage": "0_2_prepare_road_graph",
            "year": YEAR,
            "analysis_year": YEAR,
            "osm_snapshot_year": OSM_SNAPSHOT_YEAR,
            "osm_snapshot_date": OSM_SNAPSHOT_DATE,
        },
    )
    log(f"阶段目录状态：{stage_status}")

    PROVINCE_PBF_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    osmium_exe = resolve_osmium(PROJECT_ROOT)
    log(f"osmium = {osmium_exe}")
    source_pbf = prepare_roads_only_pbf(osmium_exe)

    provinces = pd.read_parquet(
        PROVINCES_PATH,
        columns=["province_id", "province_name", "min_lon", "min_lat", "max_lon", "max_lat"],
    ).copy()
    provinces["province_id"] = pd.to_numeric(provinces["province_id"], errors="raise").astype(int)

    if ONLY_PROVINCES is not None:
        wanted = set(map(str, ONLY_PROVINCES))
        provinces = provinces[provinces["province_name"].astype(str).isin(wanted)].copy()
        missing = wanted - set(provinces["province_name"].astype(str))
        if missing:
            raise RuntimeError(f"ONLY_PROVINCES 中省名未找到：{sorted(missing)}")
    elif REQUIRE_COMPLETE_RUN_WHEN_UNFILTERED and len(provinces) < 31:
        raise RuntimeError(f"省界标准表仅 {len(provinces)} 个省级单元，正式运行应检查是否完整。")

    routing_buffer_km = float(ROUTING_BUFFER_KM + EXTRA_ROUTING_SAFETY_KM)
    rows = []
    for seq, r in provinces.sort_values("province_id").reset_index(drop=True).iterrows():
        pid = int(r["province_id"]); pname = str(r["province_name"])
        bbox = expand_bbox_conservatively(
            float(r["min_lon"]), float(r["min_lat"]), float(r["max_lon"]), float(r["max_lat"]), routing_buffer_km
        )
        pbf_fp = build_fingerprint(
            config={
                "stage":"0_2_province_pbf", "year":YEAR,
                "osm_snapshot_year":OSM_SNAPSHOT_YEAR, "osm_snapshot_date":OSM_SNAPSHOT_DATE,
                "province_id":pid, "routing_bbox":list(bbox),
            },
            files=[source_pbf],
        )
        log(f"[{seq+1}/{len(provinces)}] {pname}: 裁取路网")
        pbf, elapsed, status = extract_province_pbf(osmium_exe, source_pbf, pid, pname, bbox, pbf_fp)
        fmi, fmi_seconds, fmi_status = build_fmi_if_requested(pbf, pid, pname)
        rows.append({
            "analysis_year": YEAR, "osm_snapshot_year": OSM_SNAPSHOT_YEAR,
            "osm_snapshot_date": OSM_SNAPSHOT_DATE,
            "province_id": pid, "province_name": pname,
            "routing_buffer_km": routing_buffer_km,
            "min_lon": bbox[0], "min_lat": bbox[1], "max_lon": bbox[2], "max_lat": bbox[3],
            "pbf_path": str(pbf), "pbf_size_gb": pbf.stat().st_size / 1024**3,
            "extract_seconds": elapsed, "pbf_status": status,
            "fmi_path": str(fmi) if fmi else None, "fmi_seconds": fmi_seconds, "fmi_status": fmi_status,
        })

    summary = pd.DataFrame(rows).sort_values("province_id")
    ROAD_GRAPH_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(ROAD_GRAPH_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    metadata = {
        "year": YEAR,
        "analysis_year": YEAR,
        "osm_snapshot_year": OSM_SNAPSHOT_YEAR,
        "osm_snapshot_date": OSM_SNAPSHOT_DATE,
        "osm_snapshot_alignment": "next_year_jan1_as_analysis_year_end",
        "source_pbf": str(OSM_SOURCE_PBF),
        "analysis_mode": ANALYSIS_MODE,
        "search_threshold_min": MAX_ANALYSIS_TIME_MIN,
        "routing_buffer_km": routing_buffer_km, "candidate_od_stage": False,
        "snap_gate": False, "province_count": int(len(summary)),
        "elapsed_seconds": time.time() - total_t0,
    }
    import json
    ROAD_GRAPH_METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"完成：{ROAD_GRAPH_SUMMARY_PATH}")

if __name__ == "__main__":
    if is_year_worker():
        main()
    else:
        run_script_for_all_years(__file__)
