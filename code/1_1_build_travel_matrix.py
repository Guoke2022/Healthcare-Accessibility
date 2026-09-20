# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import (
    ALLOW_CROSS_PROVINCE_HOSPITALS, PREPARED_ROOT, ROAD_GRAPH_ROOT, TRAVEL_TIME_ROOT,
    ANALYSIS_MODE, RUN_POLICY, MAX_ANALYSIS_TIME_MIN, MAX_ROAD_SPEED_KMH, UNDIRECTED, PROJECT_ROOT,
    SPEED_PROFILES, ROUTER_THREADS, get_run_year, is_year_worker, run_script_for_all_years,
    ONLY_PROVINCES, REQUIRE_COMPLETE_RUN_WHEN_UNFILTERED, service_scope_tag,
    COMPONENT_RESCUE_ENABLED, TINY_COMPONENT_MAX_ROAD_KM,
    COMPONENT_RESCUE_MIN_SIZE_RATIO, COMPONENT_RESCUE_MIN_GRID_COUNT,
    COMPONENT_RESCUE_MIN_GRID_RATIO, COMPONENT_RESCUE_MAX_EXTRA_KM,
)
from utils.cache import (
    build_fingerprint, cache_valid, file_sha256, write_cache_meta, prepare_stage_directory,
)
from utils.input_signatures import (
    hospital_routing_signature,
    population_routing_signature_for_path,
)
from utils.executables import resolve_cargo

YEAR = get_run_year()
SERVICE_SCOPE = service_scope_tag()

# Shapley / counterfactual component overrides.
# Normal main-analysis execution leaves these unset, so all three component years == YEAR.
ROAD_YEAR = int(os.environ.get("NC_COMPONENT_ROAD_YEAR", YEAR))
POPULATION_YEAR = int(os.environ.get("NC_COMPONENT_POPULATION_YEAR", YEAR))
HOSPITAL_YEAR = int(os.environ.get("NC_COMPONENT_HOSPITAL_YEAR", YEAR))
SCENARIO_LABEL = os.environ.get("NC_COMPONENT_SCENARIO", "").strip()

HOSPITALS_PATH = PREPARED_ROOT / str(HOSPITAL_YEAR) / "hospitals.parquet"
POPULATION_PARTS_DIR = PREPARED_ROOT / str(POPULATION_YEAR) / "population_parts"
PROVINCES_PATH = PREPARED_ROOT / "common" / "provinces.parquet"
ROAD_GRAPH_SUMMARY_PATH = ROAD_GRAPH_ROOT / str(ROAD_YEAR) / "road_graph_summary.csv"
ROAD_GRAPH_METADATA_PATH = ROAD_GRAPH_ROOT / str(ROAD_YEAR) / "road_graph_metadata.json"
ROAD_GRAPH_MANIFEST = ROAD_GRAPH_ROOT / str(ROAD_YEAR) / "_stage_input.json"

_output_override = os.environ.get("NC_TRAVEL_TIME_OUTPUT_ROOT_OVERRIDE", "").strip()
OUTPUT_ROOT = (
    Path(_output_override).expanduser().resolve()
    if _output_override
    else TRAVEL_TIME_ROOT / str(YEAR) / SERVICE_SCOPE
)
RUST_ROUTER_DIR = Path(os.environ.get("NC_RUST_ROUTER_DIR", PROJECT_ROOT / "osm_batch_router_v2")).expanduser().resolve()
PARQUET_CHUNK_ROWS = 2_000_000
PARQUET_COMPRESSION = "zstd"
OVERWRITE = RUN_POLICY == "force_rebuild"
ROUTER_PROTOCOL_VERSION = "component_rescue_v2"
KEEP_BINARY_TEMP = False

# Bump only when Python-side routing semantics/input serialization changes.
ROUTING_CACHE_SCHEMA_VERSION = 3

GRID_INPUT_DTYPE = np.dtype([("grid_id", "<i8"), ("lon", "<f8"), ("lat", "<f8")], align=False)
HOSPITAL_INPUT_DTYPE = np.dtype([("hospital_row", "<i4"), ("lon", "<f8"), ("lat", "<f8")], align=False)
TRAVEL_DTYPE = np.dtype([("hospital_row", "<i4"), ("grid_id", "<i8"), ("travel_time_min", "<f4")], align=False)
NEAREST_DTYPE = np.dtype([("grid_id", "<i8"), ("nearest_hospital_time_min", "<f4")], align=False)
GRID_SNAP_DTYPE = np.dtype([
    ("grid_id", "<i8"), ("snap_kind", "u1"), ("node_id", "<u4"), ("segment_id", "<u4"),
    ("edge_fraction", "<f4"), ("snap_distance_km", "<f4"), ("component_id", "<u4"),
], align=False)
HOSPITAL_SNAP_DTYPE = np.dtype([
    ("hospital_row", "<i4"), ("snap_kind", "u1"), ("node_id", "<u4"), ("segment_id", "<u4"),
    ("edge_fraction", "<f4"), ("snap_distance_km", "<f4"), ("component_id", "<u4"),
], align=False)
assert GRID_INPUT_DTYPE.itemsize == 24 and HOSPITAL_INPUT_DTYPE.itemsize == 20
assert TRAVEL_DTYPE.itemsize == 16 and NEAREST_DTYPE.itemsize == 12
assert GRID_SNAP_DTYPE.itemsize == 29 and HOSPITAL_SNAP_DTYPE.itemsize == 25


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def safe_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", str(name)).strip().replace(" ", "_")


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label}不存在：{path}")


def run_command(cmd, cwd=None, env=None) -> None:
    log("执行：" + " ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd))
    p = subprocess.run([str(x) for x in cmd], cwd=str(cwd) if cwd else None, env=env, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"命令执行失败，returncode={p.returncode}")


def verify_router_protocol(exe: Path) -> None:
    p = subprocess.run([str(exe), "--protocol-version"], capture_output=True, text=True, check=False)
    got = (p.stdout or "").strip()
    if p.returncode != 0 or got != ROUTER_PROTOCOL_VERSION:
        raise RuntimeError(
            "Rust router 与 Python 1_1 版本不匹配。\n"
            f"期望协议: {ROUTER_PROTOCOL_VERSION}\n"
            f"实际 stdout: {got!r}\n"
            f"stderr: {(p.stderr or '').strip()!r}"
        )


def router_source_digest(source_files: list[Path]) -> str:

    h = hashlib.sha256()
    for src in sorted((Path(x).resolve() for x in source_files), key=lambda x: str(x).lower()):
        h.update(str(src.name).encode("utf-8"))
        h.update(b"\0")
        with src.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()


def build_router() -> Path:
    ensure_exists(RUST_ROUTER_DIR / "Cargo.toml", "Rust Cargo.toml")
    ensure_exists(RUST_ROUTER_DIR / "src" / "main.rs", "Rust main.rs")
    exe = RUST_ROUTER_DIR / "target" / "release" / ("osm_batch_router.exe" if os.name == "nt" else "osm_batch_router")
    stamp = RUST_ROUTER_DIR / "target" / "release" / ".osm_batch_router_source.sha256"
    source_files = [RUST_ROUTER_DIR / "Cargo.toml", RUST_ROUTER_DIR / "src" / "main.rs"]
    lock = RUST_ROUTER_DIR / "Cargo.lock"
    if lock.exists(): source_files.append(lock)
    digest = router_source_digest(source_files)
    stamped = stamp.read_text(encoding="ascii").strip() if stamp.exists() else ""
    needs_build = (not exe.exists()) or (stamped != digest)

    if os.environ.get("NC_RUST_ROUTER_PREBUILT", "0") == "1":
        if needs_build:
            raise RuntimeError(
                "runner 声明 Rust router 已预编译，但 binary/source 内容哈希不一致。"
                "请重新启动 reconstruct_from_raw_inputs.py，让 HPC runner 串行重编译 Rust router。"
            )
        verify_router_protocol(exe)
        return exe

    if needs_build:
        cargo = resolve_cargo(PROJECT_ROOT)
        log("Rust router 源码内容变化或缺少 build stamp：清理该 crate 缓存后强制重编译...")
        run_command([cargo, "clean", "-p", "osm_batch_router"], cwd=RUST_ROUTER_DIR)
        run_command([cargo, "build", "--release", "--bin", "osm_batch_router"], cwd=RUST_ROUTER_DIR)
        ensure_exists(exe, "Rust release binary")
        verify_router_protocol(exe)
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(digest + "\n", encoding="ascii")
    else:
        verify_router_protocol(exe)
    return exe


def write_grid_input(path: Path, out: Path) -> int:
    df = pd.read_parquet(path, columns=["grid_id", "lon", "lat"])
    arr = np.empty(len(df), dtype=GRID_INPUT_DTYPE)
    for c in ["grid_id", "lon", "lat"]:
        arr[c] = df[c].to_numpy(dtype=arr.dtype[c])
    arr.tofile(out)
    return len(arr)


def write_hospital_input(rows: np.ndarray, hospitals_indexed: pd.DataFrame, out: Path) -> int:
    rows = np.asarray(rows, dtype=np.int64)
    missing = rows[~np.isin(rows, hospitals_indexed.index.to_numpy(dtype=np.int64))]
    if len(missing):
        raise KeyError(f"hospital_row 在 hospitals.parquet 找不到：{missing[:20].tolist()}")
    sub = hospitals_indexed.loc[rows]
    arr = np.empty(len(sub), dtype=HOSPITAL_INPUT_DTYPE)
    arr["hospital_row"] = sub.index.to_numpy(dtype=np.int32)
    arr["lon"] = sub["lon"].to_numpy(dtype=np.float64)
    arr["lat"] = sub["lat"].to_numpy(dtype=np.float64)
    arr.tofile(out)
    return len(arr)


def structured_binary_to_parquet(binary_path: Path, parquet_path: Path, dtype: np.dtype, types: dict[str, pa.DataType]) -> int:
    ensure_exists(binary_path, "Rust binary output")
    size = binary_path.stat().st_size
    if size % dtype.itemsize:
        raise RuntimeError(f"binary record size 不匹配：{binary_path}, size={size}, record={dtype.itemsize}")
    n = size // dtype.itemsize
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    if n == 0:
        pq.write_table(pa.table({k: pa.array([], type=types[k]) for k in dtype.names}), parquet_path, compression=PARQUET_COMPRESSION)
        return 0
    mm = np.memmap(binary_path, mode="r", dtype=dtype)
    writer = None
    try:
        for start in range(0, n, PARQUET_CHUNK_ROWS):
            chunk = mm[start:min(start+PARQUET_CHUNK_ROWS, n)]
            table = pa.Table.from_arrays([pa.array(np.asarray(chunk[k]), type=types[k]) for k in dtype.names], names=dtype.names)
            if writer is None:
                writer = pq.ParquetWriter(parquet_path, table.schema, compression=PARQUET_COMPRESSION, use_dictionary=False)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
        del mm
    return int(n)


TRAVEL_TYPES = {"hospital_row":pa.int32(), "grid_id":pa.int64(), "travel_time_min":pa.float32()}
NEAREST_TYPES = {"grid_id":pa.int64(), "nearest_hospital_time_min":pa.float32()}
GRID_SNAP_TYPES = {"grid_id":pa.int64(), "snap_kind":pa.uint8(), "node_id":pa.uint32(), "segment_id":pa.uint32(), "edge_fraction":pa.float32(), "snap_distance_km":pa.float32(), "component_id":pa.uint32()}
HOSP_SNAP_TYPES = {"hospital_row":pa.int32(), "snap_kind":pa.uint8(), "node_id":pa.uint32(), "segment_id":pa.uint32(), "edge_fraction":pa.float32(), "snap_distance_km":pa.float32(), "component_id":pa.uint32()}


def snap_stats(path: Path, id_col: str) -> dict:
    x = pd.read_parquet(path, columns=[id_col, "snap_kind", "snap_distance_km"])
    d = pd.to_numeric(x["snap_distance_km"], errors="coerce").to_numpy(float)
    finite = d[np.isfinite(d)]
    return {
        "n": int(len(x)), "n_edge_snap": int((x["snap_kind"] == 0).sum()), "n_node_snap": int((x["snap_kind"] == 1).sum()),
        "snap_mean_km": float(np.mean(finite)) if len(finite) else np.nan,
        "snap_median_km": float(np.median(finite)) if len(finite) else np.nan,
        "snap_p95_km": float(np.quantile(finite, .95)) if len(finite) else np.nan,
        "snap_max_km": float(np.max(finite)) if len(finite) else np.nan,
    }


def select_hospital_rows(pid: int, road_row: pd.Series, hospitals: pd.DataFrame) -> tuple[np.ndarray, str]:
    valid = hospitals["is_valid_for_routing"].fillna(False).astype(bool)
    if not ALLOW_CROSS_PROVINCE_HOSPITALS:
        mask = valid & (pd.to_numeric(hospitals["province_id_spatial"], errors="coerce") == pid)
        return hospitals.loc[mask, "source_row"].to_numpy(dtype=np.int32), "same_province_hospitals"


    mask = valid & hospitals["lon"].between(float(road_row["min_lon"]), float(road_row["max_lon"])) & hospitals["lat"].between(float(road_row["min_lat"]), float(road_row["max_lat"]))
    return hospitals.loc[mask, "source_row"].to_numpy(dtype=np.int32), "routing_bbox_hospital_pool"


def process_province(row: pd.Series, hospitals_indexed: pd.DataFrame, hospitals_flat: pd.DataFrame, router_binary: Path, router_source_files: list[Path]) -> list[dict]:
    pid = int(row["province_id"]); pname = str(row["province_name"]); safe = safe_filename(pname)
    population_path = POPULATION_PARTS_DIR / f"province_{pid:03d}_{pname}.parquet"
    pbf_path = Path(row["pbf_path"])
    ensure_exists(population_path, f"{pname} population")
    ensure_exists(pbf_path, f"{pname} road PBF")
    hospital_rows, hospital_scope_note = select_hospital_rows(pid, row, hospitals_flat)
    if len(hospital_rows) == 0:
        raise RuntimeError(f"{pname}: routing hospital pool 为空")

    matrix_paths = {p: OUTPUT_ROOT/p/"matrix"/f"province_{pid:03d}_{pname}.parquet" for p in SPEED_PROFILES}
    nearest_paths = {p: OUTPUT_ROOT/p/"nearest"/f"province_{pid:03d}_{pname}.parquet" for p in SPEED_PROFILES}
    grid_snap_path = OUTPUT_ROOT/"snap_qc"/f"province_{pid:03d}_{pname}_grid_snap.parquet"
    hosp_snap_path = OUTPUT_ROOT/"snap_qc"/f"province_{pid:03d}_{pname}_hospital_snap.parquet"
    router_summary_path = OUTPUT_ROOT/"router_qc"/f"province_{pid:03d}_{pname}.json"
    road_stats_path = OUTPUT_ROOT/"router_qc"/f"province_{pid:03d}_{pname}_road_class_stats.csv"
    component_stats_path = OUTPUT_ROOT/"router_qc"/f"province_{pid:03d}_{pname}_component_stats.csv"

    # Semantic dependency split: routing consumes grid support/coordinates and hospital
    # routing fields only. Beds/population magnitudes are deliberately excluded here.
    pop_routing_sig = population_routing_signature_for_path(population_path)
    selected_hospitals = hospitals_indexed.loc[np.asarray(hospital_rows, dtype=np.int64)].reset_index(drop=True)
    hosp_routing_sig = hospital_routing_signature(selected_hospitals)

    fp = build_fingerprint(
        config={"stage":"1_1_build_travel_matrix","cache_schema_version":ROUTING_CACHE_SCHEMA_VERSION,
                "year":YEAR,"road_year":ROAD_YEAR,"population_year":POPULATION_YEAR,
                "hospital_year":HOSPITAL_YEAR,"scenario":SCENARIO_LABEL or None,
                "province_id":pid,"service_scope":SERVICE_SCOPE,
                "profiles":SPEED_PROFILES,"cutoff":MAX_ANALYSIS_TIME_MIN,"max_speed":MAX_ROAD_SPEED_KMH,
                "undirected":UNDIRECTED,"router_threads":ROUTER_THREADS,"hospital_scope":hospital_scope_note,
                "component_rescue_enabled":COMPONENT_RESCUE_ENABLED,
                "tiny_component_max_road_km":TINY_COMPONENT_MAX_ROAD_KM,
                "component_rescue_min_size_ratio":COMPONENT_RESCUE_MIN_SIZE_RATIO,
                "component_rescue_min_grid_count":COMPONENT_RESCUE_MIN_GRID_COUNT,
                "component_rescue_min_grid_ratio":COMPONENT_RESCUE_MIN_GRID_RATIO,
                "component_rescue_max_extra_km":COMPONENT_RESCUE_MAX_EXTRA_KM,
                "population_routing_signature":pop_routing_sig,
                "hospital_routing_signature":hosp_routing_sig},
        files=[pbf_path, *router_source_files],
    )
    primary = matrix_paths[SPEED_PROFILES[0]]
    extras = [*nearest_paths.values(), grid_snap_path, hosp_snap_path, router_summary_path, road_stats_path, component_stats_path, *[matrix_paths[p] for p in SPEED_PROFILES[1:]]]
    if not OVERWRITE and cache_valid(
        primary, fp, extra_outputs=extras,
        event_payload={"stage":"1_1_province","year":YEAR,"province_id":pid,"province_name":pname},
    ):
        meta = json.loads(Path(str(primary)+".meta.json").read_text(encoding="utf-8"))
        return meta.get("summary_rows", [])

    temp_dir = OUTPUT_ROOT/"_tmp"/f"province_{pid:03d}_{safe}"
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    rust_out = temp_dir/"rust"; rust_out.mkdir(parents=True)
    grids_bin=temp_dir/"grids.bin"; hospitals_bin=temp_dir/"hospitals.bin"
    n_grids=write_grid_input(population_path, grids_bin); n_hospitals=write_hospital_input(hospital_rows,hospitals_indexed,hospitals_bin)
    log(f"{pname}: grids={n_grids:,}, routing_hospitals={n_hospitals:,}, scope={hospital_scope_note}")
    t0=time.time()
    run_command([
        router_binary, pbf_path, grids_bin, hospitals_bin, rust_out,
        MAX_ANALYSIS_TIME_MIN, int(MAX_ROAD_SPEED_KMH), str(UNDIRECTED).lower(),
        ROUTER_THREADS, ",".join(SPEED_PROFILES),
        str(COMPONENT_RESCUE_ENABLED).lower(),
        TINY_COMPONENT_MAX_ROAD_KM,
        COMPONENT_RESCUE_MIN_SIZE_RATIO,
        COMPONENT_RESCUE_MIN_GRID_COUNT,
        COMPONENT_RESCUE_MIN_GRID_RATIO,
        COMPONENT_RESCUE_MAX_EXTRA_KM,
    ])
    route_seconds=time.time()-t0

    router_summary=json.loads((rust_out/"router_summary.json").read_text(encoding="utf-8"))
    if router_summary.get("snap_gate_applied") is not False:
        raise RuntimeError(f"{pname}: Rust summary 显示 snap gate 未正确关闭")
    expected_rescue = {
        "component_rescue_enabled": COMPONENT_RESCUE_ENABLED,
        "tiny_component_max_road_km": float(TINY_COMPONENT_MAX_ROAD_KM),
        "component_rescue_min_size_ratio": float(COMPONENT_RESCUE_MIN_SIZE_RATIO),
        "component_rescue_min_grid_count": int(COMPONENT_RESCUE_MIN_GRID_COUNT),
        "component_rescue_min_grid_ratio": float(COMPONENT_RESCUE_MIN_GRID_RATIO),
        "component_rescue_max_extra_km": float(COMPONENT_RESCUE_MAX_EXTRA_KM),
    }
    for k, v in expected_rescue.items():
        got = router_summary.get(k)
        if isinstance(v, bool):
            if got is not v:
                raise RuntimeError(f"{pname}: Rust {k}={got!r} != Python {v!r}")
        elif got is None or not np.isclose(float(got), float(v), rtol=0, atol=1e-9):
            raise RuntimeError(f"{pname}: Rust {k}={got!r} != Python {v!r}")
    router_summary_path.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(rust_out/"router_summary.json",router_summary_path)
    shutil.copy2(rust_out/"road_class_stats.csv",road_stats_path)
    shutil.copy2(rust_out/"component_stats.csv",component_stats_path)
    structured_binary_to_parquet(rust_out/"grid_snap.bin",grid_snap_path,GRID_SNAP_DTYPE,GRID_SNAP_TYPES)
    structured_binary_to_parquet(rust_out/"hospital_snap.bin",hosp_snap_path,HOSPITAL_SNAP_DTYPE,HOSP_SNAP_TYPES)
    gs=snap_stats(grid_snap_path,"grid_id"); hs=snap_stats(hosp_snap_path,"hospital_row")

    profile_map={x["profile"]:x for x in router_summary["profiles"]}; rows=[]
    for profile in SPEED_PROFILES:
        n_records=structured_binary_to_parquet(rust_out/f"travel_{profile}.bin",matrix_paths[profile],TRAVEL_DTYPE,TRAVEL_TYPES)
        n_nearest=structured_binary_to_parquet(rust_out/f"nearest_{profile}.bin",nearest_paths[profile],NEAREST_DTYPE,NEAREST_TYPES)
        if n_nearest!=n_grids: raise RuntimeError(f"{pname}/{profile}: nearest rows {n_nearest} != grids {n_grids}")
        ps=profile_map[profile]
        rows.append({"province_id":pid,"province_name":pname,"profile":profile,"n_grids":n_grids,"n_routing_hospitals":n_hospitals,
                     "hospital_scope_note":hospital_scope_note,"n_travel_records_le_cutoff":n_records,"route_searches":ps["route_searches"],
                     "skipped_hospitals_no_target_component":ps["skipped_hospitals_no_target_component"],
                     "nearest_reachable_grids":ps["nearest_reachable_grids"],"routing_seconds_profile":ps["routing_seconds"],"nearest_seconds_profile":ps["nearest_seconds"],
                     "grid_edge_snap_count":gs["n_edge_snap"],"grid_node_snap_count":gs["n_node_snap"],"grid_snap_mean_km":gs["snap_mean_km"],"grid_snap_p95_km":gs["snap_p95_km"],"grid_snap_max_km":gs["snap_max_km"],
                     "hospital_edge_snap_count":hs["n_edge_snap"],"hospital_node_snap_count":hs["n_node_snap"],"hospital_snap_mean_km":hs["snap_mean_km"],"hospital_snap_p95_km":hs["snap_p95_km"],"hospital_snap_max_km":hs["snap_max_km"],
                     "component_count":router_summary.get("component_count"),"tiny_component_count":router_summary.get("tiny_component_count"),
                     "grid_component_rescue_count":router_summary.get("grid_component_rescue_count",0),
                     "hospital_component_rescue_count":router_summary.get("hospital_component_rescue_count",0),
                     "component_rescue_enabled":router_summary.get("component_rescue_enabled"),
                     "snap_gate_applied":False,"matrix_path":str(matrix_paths[profile]),"nearest_path":str(nearest_paths[profile]),"total_router_call_seconds":route_seconds,"status":"ok"})
    write_cache_meta(primary,fp,payload={
        "summary_rows":rows,
        "router_source_sha256":file_sha256(RUST_ROUTER_DIR/"src"/"main.rs"),
        "population_routing_signature":pop_routing_sig,
        "hospital_routing_signature":hosp_routing_sig,
        "routing_cache_schema_version":ROUTING_CACHE_SCHEMA_VERSION,
    })
    if not KEEP_BINARY_TEMP: shutil.rmtree(temp_dir,ignore_errors=True)
    return rows


def main() -> None:
    log("="*96); log("1_1_build_travel_matrix 开始")
    log(
        f"YEAR={YEAR} | road={ROAD_YEAR} | population={POPULATION_YEAR} | hospital={HOSPITAL_YEAR} "
        f"| scenario={SCENARIO_LABEL or 'main'} | scope={SERVICE_SCOPE} "
        f"| profiles={SPEED_PROFILES} | router_threads={ROUTER_THREADS}"
    )
    log("snapping=street segment；motorway=node only；motorway_link=edge allowed；snap gate=OFF")
    log(
        f"component rescue={COMPONENT_RESCUE_ENABLED} | "
        f"tiny<{TINY_COMPONENT_MAX_ROAD_KM:g} km | "
        f"road_ratio>={COMPONENT_RESCUE_MIN_SIZE_RATIO:g}x | "
        f"min_raw_grids={COMPONENT_RESCUE_MIN_GRID_COUNT} | "
        f"grid_ratio>={COMPONENT_RESCUE_MIN_GRID_RATIO:g}x | "
        f"max_extra={COMPONENT_RESCUE_MAX_EXTRA_KM:g} km"
    )
    ensure_exists(HOSPITALS_PATH,"hospitals.parquet");ensure_exists(POPULATION_PARTS_DIR,"population_parts");ensure_exists(ROAD_GRAPH_SUMMARY_PATH,"road_graph_summary.csv");ensure_exists(ROAD_GRAPH_METADATA_PATH,"road_graph_metadata.json");ensure_exists(ROAD_GRAPH_MANIFEST,"0_2 stage manifest")
    road_metadata=json.loads(ROAD_GRAPH_METADATA_PATH.read_text(encoding="utf-8"))
    osm_snapshot_year=road_metadata.get("osm_snapshot_year")
    osm_snapshot_date=road_metadata.get("osm_snapshot_date")
    if int(road_metadata.get("analysis_year", road_metadata.get("year", ROAD_YEAR))) != ROAD_YEAR:
        raise RuntimeError(
            f"0_2 road metadata analysis_year 与 ROAD_YEAR={ROAD_YEAR} 不一致：{road_metadata}"
        )
    log(f"OSM snapshot={osm_snapshot_date or osm_snapshot_year} | source={Path(road_metadata.get('source_pbf','')).name}")
    router_binary=build_router(); source_files=[RUST_ROUTER_DIR/"Cargo.toml", RUST_ROUTER_DIR/"src"/"main.rs"]
    lock=RUST_ROUTER_DIR/"Cargo.lock"; source_files += [lock] if lock.exists() else []
    # Stage-level fingerprint contains only dependencies that invalidate *all* provinces.
    # Hospital/population changes are validated per province below, so a one-row beds edit
    # cannot wipe the whole 1_1 directory.
    stage_fp=build_fingerprint(config={
        "stage":"1_1_build_travel_matrix","cache_schema_version":ROUTING_CACHE_SCHEMA_VERSION,
        "year":YEAR,"road_year":ROAD_YEAR,
        "population_year":POPULATION_YEAR,"hospital_year":HOSPITAL_YEAR,
        "scenario":SCENARIO_LABEL or None,"scope":SERVICE_SCOPE,
        "profiles":SPEED_PROFILES,"cutoff":MAX_ANALYSIS_TIME_MIN,"max_speed":MAX_ROAD_SPEED_KMH,
        "undirected":UNDIRECTED,"router_threads":ROUTER_THREADS,"only_provinces":ONLY_PROVINCES,
        "snap_gate":False,"motorway_edge_snap":False,"motorway_link_edge_snap":True,
        "component_rescue_enabled":COMPONENT_RESCUE_ENABLED,
        "tiny_component_max_road_km":TINY_COMPONENT_MAX_ROAD_KM,
        "component_rescue_min_size_ratio":COMPONENT_RESCUE_MIN_SIZE_RATIO,
        "component_rescue_min_grid_count":COMPONENT_RESCUE_MIN_GRID_COUNT,
        "component_rescue_min_grid_ratio":COMPONENT_RESCUE_MIN_GRID_RATIO,
        "component_rescue_max_extra_km":COMPONENT_RESCUE_MAX_EXTRA_KM,
    },files=[ROAD_GRAPH_MANIFEST,Path(__file__).resolve(),*source_files])
    status=prepare_stage_directory(
        OUTPUT_ROOT,
        stage_fp,
        run_policy=RUN_POLICY,
        payload={
            "stage":"1_1_build_travel_matrix",
            "year":YEAR,
            "road_year":ROAD_YEAR,
            "population_year":POPULATION_YEAR,
            "hospital_year":HOSPITAL_YEAR,
            "scenario":SCENARIO_LABEL or None,
        },
    );log(f"阶段目录状态：{status}")

    hospitals=pd.read_parquet(HOSPITALS_PATH,columns=["source_row","lon","lat","is_valid_for_routing","province_id_spatial"]).copy()
    hospitals["source_row"]=pd.to_numeric(hospitals["source_row"],errors="raise").astype(np.int32)
    hospitals_indexed=hospitals.set_index("source_row",drop=False).sort_index()
    road=pd.read_csv(ROAD_GRAPH_SUMMARY_PATH)
    if ONLY_PROVINCES is not None: road=road[road["province_name"].astype(str).isin(set(map(str,ONLY_PROVINCES)))].copy()
    if ONLY_PROVINCES is None and REQUIRE_COMPLETE_RUN_WHEN_UNFILTERED:
        expected=pd.read_parquet(PROVINCES_PATH,columns=["province_name"])["province_name"].astype(str).nunique()
        if road["province_name"].astype(str).nunique()!=expected: raise RuntimeError(f"road_graph_summary 省份不完整：{road['province_name'].nunique()} != {expected}")
    all_rows=[]
    for i,(_,r) in enumerate(road.sort_values("province_id").iterrows(),1):
        log(f"[{i}/{len(road)}] {r['province_name']}")
        all_rows.extend(process_province(r,hospitals_indexed,hospitals,router_binary,source_files))
    summary=pd.DataFrame(all_rows); summary.to_csv(OUTPUT_ROOT/"travel_matrix_summary.csv",index=False,encoding="utf-8-sig")
    metadata={
        "year":YEAR, "analysis_year":YEAR,
        "road_year":ROAD_YEAR,
        "population_year":POPULATION_YEAR,
        "hospital_year":HOSPITAL_YEAR,
        "scenario":SCENARIO_LABEL or None,
        "analysis_mode":ANALYSIS_MODE,
        "routing_cutoff_min":MAX_ANALYSIS_TIME_MIN,
        "osm_snapshot_year":osm_snapshot_year, "osm_snapshot_date":osm_snapshot_date,
        "osm_snapshot_alignment":road_metadata.get("osm_snapshot_alignment"),
        "osm_source_pbf":road_metadata.get("source_pbf"),
        "service_scope":SERVICE_SCOPE,"profiles":SPEED_PROFILES,"snap_gate_applied":False,
        "snap_method":"street_segment_projection_except_motorway_node_only_with_tiny_component_rescue",
        "motorway_link_edge_snap_allowed":True,
        "component_rescue_enabled":COMPONENT_RESCUE_ENABLED,
        "tiny_component_max_road_km":TINY_COMPONENT_MAX_ROAD_KM,
        "component_rescue_min_size_ratio":COMPONENT_RESCUE_MIN_SIZE_RATIO,
        "component_rescue_min_grid_count":COMPONENT_RESCUE_MIN_GRID_COUNT,
        "component_rescue_min_grid_ratio":COMPONENT_RESCUE_MIN_GRID_RATIO,
        "component_rescue_max_extra_km":COMPONENT_RESCUE_MAX_EXTRA_KM,
        "od_candidate_stage":False,"router_threads":ROUTER_THREADS
    }
    (OUTPUT_ROOT/"travel_matrix_metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    log(f"完成：{OUTPUT_ROOT/'travel_matrix_summary.csv'}")


if __name__ == "__main__":
    if is_year_worker(): main()
    else: run_script_for_all_years(__file__)
