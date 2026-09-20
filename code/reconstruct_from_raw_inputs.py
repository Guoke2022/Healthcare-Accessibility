# -*- coding: utf-8 -*-
"""Optional high-performance workflow for reconstructing the analysis from raw geospatial inputs."""


from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import (
    ANALYSIS_MODE,
    CODE_ROOT,
    PROJECT_ROOT,
    RESULT_ROOT,
    MAIN_RESULT_ROOT,
    MAX_ANALYSIS_TIME_MIN,
    MAX_ROAD_SPEED_KMH,
    ROUTING_BUFFER_KM,
    CPU_BUDGET,
    ROUTER_THREADS,
    YEAR_ENV_VAR,
    YEAR_PARALLEL_WORKERS,
    YEARS,
    POPULATION_DATASET,
    population_dataset_label,
    WORLDPOP_R2025A_ROOT,
)
from utils.executables import executable_version, resolve_cargo, resolve_osmium
from utils.inequality_schema import INEQUALITY_SCHEMA_VERSION


ROUTER_PROTOCOL_VERSION = "component_rescue_v2"


def _router_source_digest(sources: list[Path]) -> str:
    h = hashlib.sha256()
    for src in sorted((Path(x).resolve() for x in sources), key=lambda x: str(x).lower()):
        h.update(src.name.encode("utf-8")); h.update(b"\0")
        with src.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()


def _router_build_stamp(rust_dir: Path) -> Path:
    return rust_dir / "target" / "release" / ".osm_batch_router_source.sha256"


def _verify_router_protocol(router_exe: Path) -> None:
    p = subprocess.run([str(router_exe), "--protocol-version"], capture_output=True, text=True, check=False)
    got = (p.stdout or "").strip()
    if p.returncode != 0 or got != ROUTER_PROTOCOL_VERSION:
        raise RuntimeError(
            f"Rust router 协议不匹配：expected={ROUTER_PROTOCOL_VERSION!r}, got={got!r}, "
            f"stderr={(p.stderr or '').strip()!r}"
        )

YEAR_PRE_ACCESS_STAGES = [
    "0_1_prepare_base_data.py",
    "0_2_prepare_road_graph.py",
    "1_1_build_travel_matrix.py",
]
YEAR_ACCESS_STAGES = ["1_2_calculate_accessibility.py"]
YEAR_STAGES = YEAR_PRE_ACCESS_STAGES + YEAR_ACCESS_STAGES


POST_STAGES = [
    "2_1_multiscale_match.py",
    "2_2_multiscale_statistics.py",
    "2_2_precompute_urban_rural_stats.py",
        "3_1_prepare_ci_data.py",
    "3_2_calculate_ci.py",
]

SHAPLEY_STAGE = "4_2_shapley_decomposition.py"
CORE_ANALYSIS_STAGES = [
    "4_1_city_dynamics.py",
    "prepare_hospital_expansion.py",
    "build_see_cie_regression_panel.py",
    "run_see_cie_regressions.py",
]


MAIN_ADDITIONAL_EXPERIMENT_STAGES = [
    "run_province_fe_robustness.py",
    "run_spatial_robustness.py",
]


MAIN_FIGURE_STAGES = [
    "fig1_1_travel_time_boxplot.py",
    "fig1_2_travel_time_province_panels.py",
    "fig1_3_travel_time_threshold_pies.py",
    "fig2_1_time_accessibility_rank.py",
    "fig2_3_city_county_accessibility_change.py",
    "fig3_1_inequality_trends.py",
    "fig3_2_multiscale_accessibility_gap.py",
    "fig4_1_frequency_distribution_2014_2024.py",
    "fig4_2_ci_plots.py",
    "fig5_1_pathway_share_final.py",
    "fig5_2_accessibility_scale_reversal_box.py",
    "fig5_3_see_cie_effect_plots.py",
    "plot_shapley_decomposition.py",
]

FINAL_STAGES: list[str] = []
INTEGRITY_STAGE: str | None = None


RESOLVED_TOOL_ENV: dict[str, str] = {}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


# Shapley is part of the formal main pipeline and can also be requested by an
# isolated sensitivity workflow. Main-only figures / Province-FE / Moran / SEM
# remain restricted to ANALYSIS_MODE=main so sensitivity reruns do not spend
# hours reproducing outputs that are unrelated to the sensitivity contrast.
RUN_SHAPLEY = _env_flag("NC_RUN_SHAPLEY", True)


RUN_ID = time.strftime("%Y%m%d_%H%M%S")
LOG_ROOT = RESULT_ROOT / "logs" / f"pipeline_{RUN_ID}"
MASTER_LOG = LOG_ROOT / "pipeline.log"
TIMING_CSV = LOG_ROOT / "timing.csv"
STAGE_LOG_ROOT = LOG_ROOT / "stages"
CACHE_EVENT_ROOT = LOG_ROOT / "cache_events"
RUN_PROVENANCE_CSV = LOG_ROOT / "run_provenance.csv"
_LOG_LOCK = threading.Lock()
_TIMING_LOCK = threading.Lock()


def _init_logs() -> None:
    STAGE_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_EVENT_ROOT.mkdir(parents=True, exist_ok=True)
    if not TIMING_CSV.exists():
        with TIMING_CSV.open("w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow([
                "year", "stage", "start_time", "end_time",
                "elapsed_seconds", "elapsed_min", "status", "stage_log",
            ])


def write_run_context() -> None:
    """Helper for write_run_context."""
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis_mode": ANALYSIS_MODE,
        "result_root": str(RESULT_ROOT),
        "main_result_root": str(MAIN_RESULT_ROOT),
        "project_root": str(PROJECT_ROOT),
        "search_threshold_min": float(MAX_ANALYSIS_TIME_MIN),
        "max_road_speed_kmh": float(MAX_ROAD_SPEED_KMH),
        "routing_buffer_km": float(ROUTING_BUFFER_KM),
        "years": list(YEARS),
        "population_dataset": POPULATION_DATASET,
        "population_dataset_label": population_dataset_label(),
        "worldpop_r2025a_root": str(WORLDPOP_R2025A_ROOT) if POPULATION_DATASET == "worldpop_r2025a_v1" else None,
        "speed_profile": os.environ.get("NC_SPEED_PROFILE", "chn_osm_default"),
        "service_scope": "cross_province" if os.environ.get("NC_ALLOW_CROSS_PROVINCE_HOSPITALS", "0").strip().lower() in {"1", "true", "yes", "y"} else "same_province",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runner": str(Path(__file__).resolve()),
        "inequality_schema_version": INEQUALITY_SCHEMA_VERSION,
        "formal_inequality_metrics": ["gini", "theil", "atkinson_05"],
        "inequality_population_universe": "finite_acc_ge_0_and_positive_population",
    }
    path = RESULT_ROOT / "_run_context.json"
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with _LOG_LOCK:
        print(line, flush=True)
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        with MASTER_LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def aggregate_cache_events() -> Path:
    """Merge per-process cache events into one human-readable provenance CSV."""
    _init_logs()
    rows = []
    for p in sorted(CACHE_EVENT_ROOT.glob("*.json")):
        try:
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    columns = [
        "timestamp", "timestamp_ns", "pid", "thread_id", "stage", "year",
        "province_id", "province_name", "profile", "scenario", "output",
        "cache_status", "reason", "input_fingerprint", "previous_fingerprint",
    ]
    with RUN_PROVENANCE_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for row in sorted(rows, key=lambda x: int(x.get("timestamp_ns", 0))):
            w.writerow(row)
    return RUN_PROVENANCE_CSV


def record_timing(*, year, stage: str, start_ts: float, end_ts: float, status: str, stage_log: Path | None = None) -> None:
    _init_logs()
    row = [
        year, stage, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_ts)),
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_ts)),
        round(end_ts - start_ts, 3), round((end_ts - start_ts) / 60.0, 3),
        status, str(stage_log) if stage_log else "",
    ]
    with _TIMING_LOCK:
        with TIMING_CSV.open("a", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(row)


def script_path(name: str) -> Path:
    p = CODE_ROOT / name
    if not p.exists():
        raise FileNotFoundError(f"缺少脚本：{p}")
    return p


def _tail_log(path: Path, max_lines: int = 60) -> str:
    """Return the tail of a stage log for immediate failure diagnostics."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"<无法读取阶段日志：{exc}>"
    lines = text.splitlines()
    tail = lines[-max_lines:]
    return "\n".join(tail)


def run_stage(name: str, *, year: int | None = None) -> None:
    env = os.environ.copy()
    env.update(RESOLVED_TOOL_ENV)
    env["NC_ROUTER_THREADS"] = str(ROUTER_THREADS)
    env["NC_CACHE_EVENT_DIR"] = str(CACHE_EVENT_ROOT)
    if year is None:
        env.pop(YEAR_ENV_VAR, None)
        tag = "全时段"
        log_tag = "all"
    else:
        env[YEAR_ENV_VAR] = str(year)
        tag = str(year)
        log_tag = str(year)

    _init_logs()
    stage_log = STAGE_LOG_ROOT / f"{log_tag}_{Path(name).stem}.log"
    t0 = time.time()
    log(f"START {tag} | {name} | detail_log={stage_log}")
    status = "ok"
    try:
        with stage_log.open("w", encoding="utf-8", errors="replace") as fout:
            subprocess.run(
                [sys.executable, "-u", str(script_path(name))],
                check=True, env=env, stdout=fout, stderr=subprocess.STDOUT,
            )
    except subprocess.CalledProcessError as e:
        status = f"failed:{e.returncode}"
        tail = _tail_log(stage_log)
        log(f"STAGE FAILED {tag} | {name} | returncode={e.returncode}")
        if tail:
            log("----- stage log tail -----")
            for line in tail.splitlines():
                log(line)
            log("----- end stage log tail -----")
        raise RuntimeError(
            f"阶段失败：year={tag}, stage={name}, returncode={e.returncode}; "
            f"详见 {stage_log}；阶段日志末尾已打印到主终端"
        ) from e
    finally:
        t1 = time.time()
        record_timing(year=tag, stage=name, start_ts=t0, end_ts=t1, status=status, stage_log=stage_log)
    log(f"DONE  {tag} | {name} | elapsed={(time.time()-t0)/60.0:.2f} min")


def _rust_router_paths() -> tuple[Path, Path, list[Path]]:
    """Helper for _rust_router_paths."""
    rust_dir = Path(
        os.environ.get("NC_RUST_ROUTER_DIR", PROJECT_ROOT / "osm_batch_router_v2")
    ).expanduser().resolve()
    cargo_toml = rust_dir / "Cargo.toml"
    main_rs = rust_dir / "src" / "main.rs"
    if not cargo_toml.exists() or not main_rs.exists():
        raise FileNotFoundError(f"Rust router 源码不完整：{rust_dir}")
    sources = [cargo_toml, main_rs]
    lock = rust_dir / "Cargo.lock"
    if lock.exists():
        sources.append(lock)
    router_exe = rust_dir / "target" / "release" / (
        "osm_batch_router.exe" if os.name == "nt" else "osm_batch_router"
    )
    return rust_dir, router_exe, sources


def preflight_external_dependencies() -> None:
    """Helper for preflight_external_dependencies."""


    global RESOLVED_TOOL_ENV

    osmium = resolve_osmium(PROJECT_ROOT)
    osmium_ver = executable_version(osmium)
    RESOLVED_TOOL_ENV["NC_OSMIUM_EXE"] = osmium
    log(f"依赖 OK | osmium={osmium} | {osmium_ver}")

    rust_dir, router_exe, sources = _rust_router_paths()
    digest = _router_source_digest(sources)
    stamp = _router_build_stamp(rust_dir)
    stamped = stamp.read_text(encoding="ascii").strip() if stamp.exists() else ""
    needs_build = (not router_exe.exists()) or (stamped != digest)
    if not needs_build:
        try:
            _verify_router_protocol(router_exe)
        except RuntimeError:
            needs_build = True
    if needs_build:
        cargo = resolve_cargo(PROJECT_ROOT)
        cargo_ver = executable_version(cargo)
        RESOLVED_TOOL_ENV["NC_CARGO_EXE"] = cargo
        log(f"依赖 OK | cargo={cargo} | {cargo_ver} | Rust router 需要强制重编译")
    else:
        log(f"Rust router 内容哈希与协议均一致：{router_exe}")


def prebuild_rust_router_once() -> None:
    """Helper for prebuild_rust_router_once."""
    rust_dir, router_exe, sources = _rust_router_paths()
    digest = _router_source_digest(sources)
    stamp = _router_build_stamp(rust_dir)
    stamped = stamp.read_text(encoding="ascii").strip() if stamp.exists() else ""
    needs_build = (not router_exe.exists()) or (stamped != digest)
    if not needs_build:
        try:
            _verify_router_protocol(router_exe)
        except RuntimeError:
            needs_build = True

    if needs_build:
        cargo = RESOLVED_TOOL_ENV.get("NC_CARGO_EXE") or resolve_cargo(PROJECT_ROOT)
        RESOLVED_TOOL_ENV["NC_CARGO_EXE"] = cargo
        log("START Rust | 源码哈希变化/协议不匹配，清理 crate 缓存并强制重编译 osm_batch_router")
        env = {**os.environ, **RESOLVED_TOOL_ENV}
        try:
            subprocess.run([cargo, "clean", "-p", "osm_batch_router"], cwd=str(rust_dir), check=True, env=env)
            subprocess.run([cargo, "build", "--release", "--bin", "osm_batch_router"], cwd=str(rust_dir), check=True, env=env)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Rust 核心路由器编译失败：returncode={e.returncode}。"
                "年度并行尚未启动，请先修复 Rust 源码。"
            ) from e
        if not router_exe.exists():
            raise FileNotFoundError(f"cargo build 成功但未找到 release binary：{router_exe}")
        _verify_router_protocol(router_exe)
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(digest + "\n", encoding="ascii")
        log(f"DONE  Rust | {router_exe} | protocol={ROUTER_PROTOCOL_VERSION}")
    else:
        _verify_router_protocol(router_exe)
        log(f"Rust router 无需重编译：{router_exe}")


    RESOLVED_TOOL_ENV["NC_RUST_ROUTER_PREBUILT"] = "1"


def prepare_common_inputs_once() -> None:
    """Helper for prepare_common_inputs_once."""
    env = os.environ.copy()
    env.update(RESOLVED_TOOL_ENV)
    env.pop(YEAR_ENV_VAR, None)
    env["NC_PREPARE_COMMON_ONLY"] = "1"
    env["NC_ROUTER_THREADS"] = str(ROUTER_THREADS)
    env["NC_CACHE_EVENT_DIR"] = str(CACHE_EVENT_ROOT)
    name = "0_1_prepare_base_data.py"
    _init_logs()
    stage_log = STAGE_LOG_ROOT / "common_0_1_prepare_common.log"
    t0 = time.time()
    status = "ok"
    log(f"START common | 0_1 共享省界预处理 | detail_log={stage_log}")
    try:
        with stage_log.open("w", encoding="utf-8", errors="replace") as fout:
            subprocess.run([sys.executable, str(script_path(name))], check=True, env=env, stdout=fout, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        status = f"failed:{e.returncode}"
        raise RuntimeError(
            f"共享数据预处理失败：stage={name}, returncode={e.returncode}; 详见 {stage_log}"
        ) from e
    finally:
        record_timing(year="common", stage="0_1_prepare_common", start_ts=t0, end_ts=time.time(), status=status, stage_log=stage_log)
    log(f"DONE  common | 0_1 共享省界预处理 | elapsed={(time.time()-t0)/60.0:.2f} min")


_YEAR_TOTAL_START: dict[int, float] = {}
_YEAR_TOTAL_LOCK = threading.Lock()


def _mark_year_start(year: int) -> None:
    with _YEAR_TOTAL_LOCK:
        _YEAR_TOTAL_START.setdefault(int(year), time.time())


def _record_year_total(year: int, status: str) -> None:
    with _YEAR_TOTAL_LOCK:
        t0 = _YEAR_TOTAL_START.pop(int(year), None)
    if t0 is not None:
        record_timing(
            year=year,
            stage="YEAR_TOTAL",
            start_ts=t0,
            end_ts=time.time(),
            status=status,
        )


def run_year_stage_group(year: int, stages: list[str], *, final_group: bool) -> int:
    """Run one dependency-safe group of annual stages."""
    _mark_year_start(year)
    status = "ok"
    try:
        for stage in stages:
            run_stage(stage, year=year)
        return year
    except Exception:
        status = "failed"
        raise
    finally:
        if final_group:
            _record_year_total(year, status)
        elif status != "ok":
            _record_year_total(year, status)


def _run_year_group_parallel(stages: list[str], *, phase: str, final_group: bool) -> None:
    workers = min(YEAR_PARALLEL_WORKERS, len(YEARS))
    log(
        f"年度阶段启动 | phase={phase} | years={YEARS[0]}-{YEARS[-1]} | "
        f"stages={stages} | year_workers={workers} | router_threads/year={ROUTER_THREADS} | "
        f"cpu_budget={CPU_BUDGET}"
    )

    if workers <= 1:
        for year in YEARS:
            run_year_stage_group(year, stages, final_group=final_group)
            log(f"YEAR PHASE COMPLETE {year} | phase={phase}")
        return

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"year-{phase.lower()}") as ex:
        futures = {
            ex.submit(run_year_stage_group, y, stages, final_group=final_group): y
            for y in YEARS
        }
        try:
            for fut in as_completed(futures):
                y = futures[fut]
                fut.result()
                log(f"YEAR PHASE COMPLETE {y} | phase={phase}")
        except Exception:
            for f in futures:
                f.cancel()
            raise


def run_years_parallel() -> None:
    # Phase A: every year's 1_1 travel matrix must exist before any 1_2 starts.
    _run_year_group_parallel(
        YEAR_PRE_ACCESS_STAGES,
        phase="PRE_ACCESS_0_1_TO_1_1",
        final_group=False,
    )

    # 1_2 topology fallback can reference later-year 1_1 matrices
    # (e.g. 2014 -> 2015/2016; 2015/2016 -> later references).
    log(
        "CROSS-YEAR BARRIER PASSED | all years completed 0_1 -> 0_2 -> 1_1; "
        "starting 1_2 accessibility"
    )

    # Phase B: now all cross-year reference matrices and their cache dependencies exist.
    _run_year_group_parallel(
        YEAR_ACCESS_STAGES,
        phase="ACCESSIBILITY_1_2",
        final_group=True,
    )

    for year in YEARS:
        log(f"YEAR COMPLETE {year}")

def main() -> None:
    _init_logs()
    pipeline_t0 = time.time()
    pipeline_status = "ok"
    log("=" * 96)
    log("Optional HPC raw-geospatial reconstruction workflow")
    write_run_context()
    log(
        f"ANALYSIS_MODE={ANALYSIS_MODE}; RESULT_ROOT={RESULT_ROOT}; "
        f"POPULATION={population_dataset_label()}; YEARS={YEARS[0]}-{YEARS[-1]}; "
        f"SEARCH_THRESHOLD={MAX_ANALYSIS_TIME_MIN:g} min; ROUTING_BUFFER={ROUTING_BUFFER_KM:g} km"
    )
    log(
        f"CPU_BUDGET={CPU_BUDGET}; YEAR_PARALLEL_WORKERS={YEAR_PARALLEL_WORKERS}; "
        f"ROUTER_THREADS={ROUTER_THREADS}"
    )
    log("核心结构：0_1 -> 0_2 -> 1_1 -> 1_2；不存在 OD-candidate 阶段，也不存在 snap gate。")
    log("=" * 96)

    try:
        # Cheap static method/config self-check before any expensive preprocessing/routing.
        preflight_external_dependencies()
        prebuild_rust_router_once()
        prepare_common_inputs_once()
        run_years_parallel()


        for stage in POST_STAGES:
            run_stage(stage)


        if RUN_SHAPLEY:
            run_stage(SHAPLEY_STAGE)
        else:
            log("NC_RUN_SHAPLEY=0：跳过 Shapley decomposition。")

        for stage in CORE_ANALYSIS_STAGES:
            run_stage(stage)

        if ANALYSIS_MODE == "main":
            for stage in MAIN_ADDITIONAL_EXPERIMENT_STAGES:
                run_stage(stage)
            for stage in MAIN_FIGURE_STAGES:
                run_stage(stage)
        else:
            log(f"{ANALYSIS_MODE}：复用正式统计/模型主链，但跳过正文绘图和 main-only 附加实验。")

        for stage in FINAL_STAGES:
            run_stage(stage)

    except Exception as e:
        pipeline_status = "failed"
        log(f"PIPELINE FAILED | {type(e).__name__}: {e}")
        raise
    finally:
        record_timing(
            year="ALL", stage="PIPELINE_TOTAL", start_ts=pipeline_t0,
            end_ts=time.time(), status=pipeline_status
        )
        log(f"PIPELINE STATUS={pipeline_status} | 总耗时={(time.time()-pipeline_t0)/3600.0:.2f} h")
        prov_path = aggregate_cache_events()
        log(f"日志目录：{LOG_ROOT}")
        log(f"阶段耗时表：{TIMING_CSV}")
        log(f"缓存/重算 provenance：{prov_path}")


if __name__ == "__main__":
    main()
