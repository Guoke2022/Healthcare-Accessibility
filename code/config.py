# -*- coding: utf-8 -*-
"""Shared configuration module for the analysis code.

This file is **not an executable workflow entry point**. Run ``python reproduce.py``
from the repository root for the released workflow. The larger upstream geospatial
workflow can be run with ``python code/reconstruct_from_raw_inputs.py`` after
supplying the required source datasets.

Responsibilities
----------------
1. Resolve repository/data/result/code paths from the repository layout or ``NC_*`` overrides.
2. Define analysis years, service scope, speed profile and computational settings.
3. Define canonical stage-output locations shared by analysis scripts.
4. Keep path and analysis configuration centralized instead of hard-coding machine paths.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# =============================================================================

# =============================================================================

# Public-repository layout: <repo>/code/config.py, so parents[1] is the repo root.
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("NC_PROJECT_ROOT", DEFAULT_PROJECT_ROOT)).expanduser().resolve()
DATA_ROOT = Path(os.environ.get("NC_DATA_ROOT", PROJECT_ROOT / "data")).expanduser().resolve()
CODE_ROOT = Path(os.environ.get("NC_CODE_ROOT", PROJECT_ROOT / "code")).expanduser().resolve()


MAIN_RESULT_ROOT = Path(os.environ.get("NC_MAIN_RESULT_ROOT", PROJECT_ROOT / "result")).expanduser().resolve()
SENSITIVITY_RESULT_ROOT = Path(os.environ.get("NC_SENSITIVITY_RESULT_ROOT", PROJECT_ROOT / "result_sensitivity")).expanduser().resolve()
ANALYSIS_MODE = os.environ.get("NC_ANALYSIS_MODE", "main").strip().lower()
ALLOWED_ANALYSIS_MODES = {
    "main",
    "threshold_sensitivity",
    "population_sensitivity",
    "fixed_road_sensitivity",
    "branch_threshold_sensitivity",
    "cross_province_sensitivity",
}
if ANALYSIS_MODE not in ALLOWED_ANALYSIS_MODES:
    raise ValueError(f"NC_ANALYSIS_MODE={ANALYSIS_MODE!r} 无效；可选：{sorted(ALLOWED_ANALYSIS_MODES)}")

RESULT_ROOT = Path(os.environ.get("NC_RESULT_ROOT", MAIN_RESULT_ROOT)).expanduser().resolve()
if ANALYSIS_MODE != "main":
    if "NC_RESULT_ROOT" not in os.environ:
        raise ValueError(f"{ANALYSIS_MODE} 模式必须显式设置 NC_RESULT_ROOT，防止误写主分析目录")
    if RESULT_ROOT == MAIN_RESULT_ROOT or MAIN_RESULT_ROOT in RESULT_ROOT.parents or RESULT_ROOT in MAIN_RESULT_ROOT.parents:
        raise ValueError(f"{ANALYSIS_MODE} 模式的 RESULT_ROOT 必须与主分析目录完全隔离，不能相同或互为父子目录")


PREPARED_ROOT = RESULT_ROOT / "0_1_prepared_data"
ROAD_GRAPH_ROOT = RESULT_ROOT / "0_2_road_graph"
TRAVEL_TIME_ROOT = RESULT_ROOT / "1_1_travel_time"
ACCESSIBILITY_ROOT = RESULT_ROOT / "1_2_accessibility"
MULTISCALE_MATCHED_ROOT = RESULT_ROOT / "2_1_multiscale_matched"
MULTISCALE_ANALYSIS_ROOT = RESULT_ROOT / "2_2_multiscale_analysis"
CI_MATCHED_ROOT = RESULT_ROOT / "3_1_ci_matched"
CI_ANALYSIS_ROOT = RESULT_ROOT / "3_2_ci_analysis"

# =============================================================================

# =============================================================================


MAIN_ANALYSIS_YEARS = list(range(2014, 2025))
POPULATION_SENSITIVITY_YEARS = list(range(2015, 2025))
YEARS = list(POPULATION_SENSITIVITY_YEARS if ANALYSIS_MODE == "population_sensitivity" else MAIN_ANALYSIS_YEARS)
YEAR_ENV_VAR = "PIPELINE_YEAR"

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------


#   2014 accessibility -> china-150101.osm.pbf
#   2015 accessibility -> china-160101.osm.pbf
#   ...
#   2024 accessibility -> china-250101.osm.pbf


OSM_SNAPSHOT_YEAR_OFFSET = 1


def get_osm_snapshot_year(analysis_year: int) -> int:
    """Helper for get_osm_snapshot_year."""


    year = int(analysis_year)
    fixed_raw = os.environ.get("NC_FIXED_OSM_ANALYSIS_YEAR", "").strip()
    if fixed_raw:
        if ANALYSIS_MODE != "fixed_road_sensitivity":
            raise ValueError(
                "NC_FIXED_OSM_ANALYSIS_YEAR 仅允许在 fixed_road_sensitivity 模式中使用"
            )
        fixed_year = int(fixed_raw)
        if fixed_year not in MAIN_ANALYSIS_YEARS:
            raise ValueError(
                f"NC_FIXED_OSM_ANALYSIS_YEAR={fixed_year} 必须位于 "
                f"{MAIN_ANALYSIS_YEARS[0]}~{MAIN_ANALYSIS_YEARS[-1]}"
            )
        year = fixed_year
    return year + OSM_SNAPSHOT_YEAR_OFFSET


def get_osm_snapshot_date(analysis_year: int) -> str:
    """Helper for get_osm_snapshot_date."""
    return f"{get_osm_snapshot_year(analysis_year)}-01-01"


def get_osm_snapshot_path(analysis_year: int) -> Path:
    """Helper for get_osm_snapshot_path."""
    snapshot_year = get_osm_snapshot_year(analysis_year)
    return DATA_ROOT / "osm" / f"china-{snapshot_year % 100:02d}0101.osm.pbf"

# =============================================================================

# =============================================================================


MAIN_ANALYSIS_TIME_MIN = 60.0
MAX_ANALYSIS_TIME_MIN = float(os.environ.get("NC_MAX_ANALYSIS_TIME_MIN", str(MAIN_ANALYSIS_TIME_MIN)))
MAX_ROAD_SPEED_KMH = 120.0
if not math.isfinite(MAX_ANALYSIS_TIME_MIN) or MAX_ANALYSIS_TIME_MIN <= 0:
    raise ValueError("NC_MAX_ANALYSIS_TIME_MIN 必须为有限正数")
if ANALYSIS_MODE != "threshold_sensitivity" and not math.isclose(MAX_ANALYSIS_TIME_MIN, MAIN_ANALYSIS_TIME_MIN, rel_tol=0.0, abs_tol=1e-12):
    label = {
        "main": "主分析",
        "population_sensitivity": "人口数据源敏感性分析",
        "fixed_road_sensitivity": "固定路网敏感性分析",
        "branch_threshold_sensitivity": "分院区床位阈值敏感性分析",
        "cross_province_sensitivity": "放开省界敏感性分析",
    }.get(ANALYSIS_MODE, ANALYSIS_MODE)
    raise ValueError(
        f"{label}搜索阈值固定为 {MAIN_ANALYSIS_TIME_MIN:g} min。"
        "如需改变阈值，请在独立 threshold-sensitivity workflow 中显式设置对应模式；"
        "人口源敏感性只允许改变 population dataset。"
    )


ROUTING_BUFFER_KM = MAX_ANALYSIS_TIME_MIN / 60.0 * MAX_ROAD_SPEED_KMH


SEARCH_THRESHOLD_SENSITIVITY_VALUES_MIN = (30.0, 45.0, 60.0, 90.0, 120.0)

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
POPULATION_DATASET = os.environ.get("NC_POPULATION_DATASET", "landscan").strip().lower()
ALLOWED_POPULATION_DATASETS = {"landscan", "worldpop_r2025a_v1"}
if POPULATION_DATASET not in ALLOWED_POPULATION_DATASETS:
    raise ValueError(f"NC_POPULATION_DATASET={POPULATION_DATASET!r} 无效；可选：{sorted(ALLOWED_POPULATION_DATASETS)}")


if ANALYSIS_MODE != "population_sensitivity" and POPULATION_DATASET != "landscan":
    raise ValueError(f"{ANALYSIS_MODE} 只能使用 LandScan；WorldPop 仅允许在 population_sensitivity 模式中运行")

WORLDPOP_R2025A_ROOT = Path(
    os.environ.get("NC_WORLDPOP_R2025A_ROOT", DATA_ROOT / "worldpop" / "R2025A version v1")
).expanduser().resolve()

def get_population_raster_path(analysis_year: int) -> Path:
    """Helper for get_population_raster_path."""
    year = int(analysis_year)
    if POPULATION_DATASET == "landscan":
        return DATA_ROOT / "landscan" / f"landscan-global-{year}.tif"
    if year < 2015:
        raise ValueError("WorldPop R2025A v1 中国 1-km 数据从 2015 年开始，不能用于 2014")
    return WORLDPOP_R2025A_ROOT / f"chn_pop_{year}_CN_1km_R2025A_UA_v1.tif"

def population_dataset_label() -> str:
    return "LandScan" if POPULATION_DATASET == "landscan" else "WorldPop R2025A version v1"


MOTORWAY_EDGE_SNAP_ALLOWED = False
MOTORWAY_LINK_EDGE_SNAP_ALLOWED = True

# -----------------------------------------------------------------------------
# Tiny-component rescue
# -----------------------------------------------------------------------------


#


#


COMPONENT_RESCUE_ENABLED = os.environ.get("NC_COMPONENT_RESCUE_ENABLED", "1").strip().lower() in {"1", "true", "yes", "y"}
TINY_COMPONENT_MAX_ROAD_KM = float(os.environ.get("NC_TINY_COMPONENT_MAX_ROAD_KM", "20.0"))
COMPONENT_RESCUE_MIN_SIZE_RATIO = float(os.environ.get("NC_COMPONENT_RESCUE_MIN_SIZE_RATIO", "10.0"))
COMPONENT_RESCUE_MIN_GRID_COUNT = int(os.environ.get("NC_COMPONENT_RESCUE_MIN_GRID_COUNT", "500"))
COMPONENT_RESCUE_MIN_GRID_RATIO = float(os.environ.get("NC_COMPONENT_RESCUE_MIN_GRID_RATIO", "3.0"))
COMPONENT_RESCUE_MAX_EXTRA_KM = float(os.environ.get("NC_COMPONENT_RESCUE_MAX_EXTRA_KM", "1.0"))
if TINY_COMPONENT_MAX_ROAD_KM <= 0:
    raise ValueError("NC_TINY_COMPONENT_MAX_ROAD_KM 必须 > 0")
if COMPONENT_RESCUE_MIN_SIZE_RATIO <= 1:
    raise ValueError("NC_COMPONENT_RESCUE_MIN_SIZE_RATIO 必须 > 1")
if COMPONENT_RESCUE_MIN_GRID_COUNT <= 0:
    raise ValueError("NC_COMPONENT_RESCUE_MIN_GRID_COUNT 必须 > 0")
if COMPONENT_RESCUE_MIN_GRID_RATIO <= 1:
    raise ValueError("NC_COMPONENT_RESCUE_MIN_GRID_RATIO 必须 > 1")
if COMPONENT_RESCUE_MAX_EXTRA_KM < 0:
    raise ValueError("NC_COMPONENT_RESCUE_MAX_EXTRA_KM 必须 >= 0")


UNDIRECTED = os.environ.get("NC_UNDIRECTED", "1").strip().lower() in {"1", "true", "yes", "y"}


SPEED_PROFILE = os.environ.get("NC_SPEED_PROFILE", "chn_osm_default").strip()
ALLOWED_SPEED_PROFILES = {"legacy", "chn_osm_default"}
if SPEED_PROFILE not in ALLOWED_SPEED_PROFILES:
    raise ValueError(
        f"SPEED_PROFILE={SPEED_PROFILE!r} 无效；"
        f"可选：{sorted(ALLOWED_SPEED_PROFILES)}"
    )


SPEED_PROFILES = [SPEED_PROFILE]


ALLOW_CROSS_PROVINCE_HOSPITALS = os.environ.get("NC_ALLOW_CROSS_PROVINCE_HOSPITALS", "0").strip().lower() in {"1", "true", "yes", "y"}


if ANALYSIS_MODE == "cross_province_sensitivity":
    if not ALLOW_CROSS_PROVINCE_HOSPITALS:
        raise ValueError("cross_province_sensitivity 必须设置 NC_ALLOW_CROSS_PROVINCE_HOSPITALS=1")
elif ALLOW_CROSS_PROVINCE_HOSPITALS:
    raise ValueError(
        f"{ANALYSIS_MODE} 不允许打开跨省医院；请使用独立 cross-province sensitivity workflow"
    )


R_GT100_POLICY = os.environ.get("NC_R_GT100_POLICY", "keep").strip().lower()  # allowed: "keep", "exclude_legacy"
if R_GT100_POLICY not in {"keep", "exclude_legacy"}:
    raise ValueError("NC_R_GT100_POLICY 只能是 'keep' 或 'exclude_legacy'")


R_TOPOLOGY_QC_ENABLED = os.environ.get(
    "NC_R_TOPOLOGY_QC_ENABLED", os.environ.get("NC_R_TOPOLOGY_FATAL_ENABLED", "1")
).strip().lower() in {"1", "true", "yes", "y"}
R_TOPOLOGY_SUSPECT_THRESHOLD = float(os.environ.get(
    "NC_R_TOPOLOGY_SUSPECT_THRESHOLD", os.environ.get("NC_R_TOPOLOGY_FATAL_THRESHOLD", "500")
))
R_TOPOLOGY_SUSPECT_MAX_REACHABLE_GRIDS = int(os.environ.get(
    "NC_R_TOPOLOGY_SUSPECT_MAX_REACHABLE_GRIDS", os.environ.get("NC_R_TOPOLOGY_FATAL_MAX_REACHABLE_GRIDS", "10")
))
if R_TOPOLOGY_SUSPECT_THRESHOLD <= 0:
    raise ValueError("NC_R_TOPOLOGY_SUSPECT_THRESHOLD 必须 > 0")
if R_TOPOLOGY_SUSPECT_MAX_REACHABLE_GRIDS < 0:
    raise ValueError("NC_R_TOPOLOGY_SUSPECT_MAX_REACHABLE_GRIDS 必须 >= 0")


ONLY_PROVINCES = None


REQUIRE_COMPLETE_RUN_WHEN_UNFILTERED = True

# =============================================================================

# =============================================================================


CPU_BUDGET = int(os.environ.get("NC_CPU_BUDGET", "100"))
if CPU_BUDGET < 1:
    raise ValueError("NC_CPU_BUDGET 必须 >= 1")
YEAR_PARALLEL_WORKERS = int(os.environ.get("NC_YEAR_WORKERS", min(4, len(YEARS), max(1, CPU_BUDGET))))
if YEAR_PARALLEL_WORKERS < 1:
    raise ValueError("NC_YEAR_WORKERS 必须 >= 1")
ROUTER_THREADS = int(os.environ.get("NC_ROUTER_THREADS", max(1, CPU_BUDGET // YEAR_PARALLEL_WORKERS)))


SHAPLEY_SCENARIO_WORKERS = int(os.environ.get("NC_SHAPLEY_SCENARIO_WORKERS", min(3, max(1, CPU_BUDGET))))
if SHAPLEY_SCENARIO_WORKERS < 1:
    raise ValueError("NC_SHAPLEY_SCENARIO_WORKERS 必须 >= 1")
_SHAPLEY_CPU_RESERVE = min(max(2, SHAPLEY_SCENARIO_WORKERS * 2), max(0, CPU_BUDGET - SHAPLEY_SCENARIO_WORKERS))
SHAPLEY_ROUTER_THREADS = int(os.environ.get(
    "NC_SHAPLEY_ROUTER_THREADS",
    max(1, (CPU_BUDGET - _SHAPLEY_CPU_RESERVE) // SHAPLEY_SCENARIO_WORKERS),
))
if SHAPLEY_ROUTER_THREADS < 1:
    raise ValueError("NC_SHAPLEY_ROUTER_THREADS 必须 >= 1")


ROAD_AUDIT_YEAR_WORKERS = int(os.environ.get("NC_ROAD_AUDIT_YEAR_WORKERS", min(3, len(YEARS), max(1, CPU_BUDGET))))
ROAD_AUDIT_RUST_THREADS = int(os.environ.get("NC_ROAD_AUDIT_RUST_THREADS", max(1, CPU_BUDGET // max(1, ROAD_AUDIT_YEAR_WORKERS))))
EXPORT_YEAR_WORKERS = int(os.environ.get("NC_EXPORT_YEAR_WORKERS", min(4, len(YEARS), max(1, CPU_BUDGET))))
MULTISCALE_YEAR_WORKERS = int(os.environ.get("NC_MULTISCALE_YEAR_WORKERS", min(6, len(YEARS), max(1, CPU_BUDGET))))
MULTISCALE_STATS_YEAR_WORKERS = int(os.environ.get("NC_MULTISCALE_STATS_YEAR_WORKERS", min(6, len(YEARS), max(1, CPU_BUDGET))))
if MULTISCALE_STATS_YEAR_WORKERS < 1:
    raise ValueError("NC_MULTISCALE_STATS_YEAR_WORKERS 必须 >= 1")


MAKE_PLOTS = os.environ.get("NC_MAKE_PLOTS", "1").strip().lower() not in {"0", "false", "no"}
PLOT_DPI = int(os.environ.get("NC_PLOT_DPI", "600"))

# =============================================================================

# =============================================================================

# auto_clean：


#

RUN_POLICY = "auto_clean"  # allowed: "auto_clean", "force_rebuild"
if RUN_POLICY not in {"auto_clean", "force_rebuild"}:
    raise ValueError("RUN_POLICY 只能是 'auto_clean' 或 'force_rebuild'")


def service_scope_tag() -> str:
    return "cross_province" if ALLOW_CROSS_PROVINCE_HOSPITALS else "same_province"


# =============================================================================

# =============================================================================

def get_run_year() -> int:
    """Helper for get_run_year."""


    raw = os.environ.get(YEAR_ENV_VAR)
    if raw is None:
        return YEARS[-1]

    year = int(raw)
    if year not in YEARS:
        raise ValueError(
            f"{YEAR_ENV_VAR}={year} 不在允许年份 {YEARS[0]}~{YEARS[-1]} 内"
        )
    return year


def is_year_worker() -> bool:
    """Helper for is_year_worker."""
    return YEAR_ENV_VAR in os.environ


def run_script_for_all_years(script_path, years=None, *, workers=None, extra_env=None) -> None:
    """Helper for run_script_for_all_years."""


    years = list(YEARS if years is None else years)
    if not years:
        return
    script_path = Path(script_path).resolve()
    workers = min(int(YEAR_PARALLEL_WORKERS if workers is None else workers), len(years))
    if workers < 1:
        raise ValueError("workers 必须 >= 1")
    extra_env = {} if extra_env is None else {str(k): str(v) for k, v in extra_env.items()}

    def _run(year: int) -> int:
        env = os.environ.copy()
        env.update(extra_env)
        env[YEAR_ENV_VAR] = str(year)
        env["NC_ROUTER_THREADS"] = str(ROUTER_THREADS)
        print(f"[年度 worker] {script_path.name} | YEAR={year}", flush=True)
        subprocess.run([sys.executable, str(script_path)], check=True, env=env)
        return year

    if workers <= 1:
        for year in years:
            _run(year)
        return

    print(
        f"年度并行：script={script_path.name}, workers={workers}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="year") as ex:
        futures = {ex.submit(_run, year): year for year in years}
        try:
            for fut in as_completed(futures):
                year = futures[fut]
                fut.result()
                print(f"[年度完成] {script_path.name} | YEAR={year}", flush=True)
        except Exception:
            for fut in futures:
                fut.cancel()
            raise


# =============================================================================

# =============================================================================

# =============================================================================

# =============================================================================


MULTISCALE_ROOT = MULTISCALE_MATCHED_ROOT
ANALYSIS_ROOT = MULTISCALE_ANALYSIS_ROOT

# =============================================================================

# =============================================================================

SERVICE_SCOPES = [service_scope_tag()]
PROFILES = list(SPEED_PROFILES)

# =============================================================================

# =============================================================================

ADMIN_BOUNDARY_VINTAGE = int(os.environ.get("NC_ADMIN_BOUNDARY_VINTAGE", "2023"))
ADMIN_SHP = Path(os.environ.get("NC_ADMIN_SHP", DATA_ROOT / "行政区划" / "T2023年初县级.shp"))


ADMIN_COLS = ["县级", "县级码", "地级", "地级码", "省级", "省级码"]
ENABLE_CITY_LEVEL = True
# Public repository may release three fixed boundary layers directly rather than
# reconstructing province/city geometry by dissolving counties.
PROVINCE_SHP = Path(os.environ.get("NC_PROVINCE_SHP", DATA_ROOT / "行政区划" / "2023年省级.shp"))
CITY_SHP = Path(os.environ.get("NC_CITY_SHP", DATA_ROOT / "行政区划" / "T2023年初地级.shp"))

# =============================================================================

# =============================================================================


APPLY_LEGACY_POSTPROCESS = os.environ.get("NC_APPLY_LEGACY_POSTPROCESS", "0").strip().lower() in {"1", "true", "yes", "y"}
LEGACY_IQR_THRESHOLD = 3.0
LEGACY_IDW_K = 50
LEGACY_IDW_POWER = 2.0
VALID_TIME_MIN = 0.0


WRITE_COMPAT_NATIONAL_CSV = False

# =============================================================================

# =============================================================================


EASTERN_REGION = [
    "北京市", "天津市", "河北省", "上海市", "江苏省",
    "浙江省", "福建省", "山东省", "广东省", "海南省",
]
COASTAL = EASTERN_REGION  # deprecated compatibility alias

NORTHERN = [
    "黑龙江省", "吉林省", "辽宁省", "河北省", "北京市", "天津市",
    "内蒙古自治区", "新疆维吾尔自治区", "甘肃省", "宁夏回族自治区",
    "山西省", "陕西省", "青海省", "山东省", "河南省",
]

# =============================================================================

# =============================================================================

ENABLE_URBAN_RURAL = False
ENABLE_POOR_COUNTY = False
POOR_COUNTY_XLSX = DATA_ROOT / "行政区划" / "全国832个国家级贫困县名单、摘帽整理数据.xlsx"

YEAR_TO_GURS = {
    2014: DATA_ROOT / "城乡数据" / "GURS_100m" / "GURS_2015_EPSG4326.tif",
    2015: DATA_ROOT / "城乡数据" / "GURS_100m" / "GURS_2015_EPSG4326.tif",
    2016: DATA_ROOT / "城乡数据" / "GURS_100m" / "GURS_2015_EPSG4326.tif",
    2019: DATA_ROOT / "城乡数据" / "GURS_100m" / "GURS_2020_EPSG4326.tif",
    2020: DATA_ROOT / "城乡数据" / "GURS_100m" / "GURS_2020_EPSG4326.tif",
    2021: DATA_ROOT / "城乡数据" / "GURS_100m" / "GURS_2020_EPSG4326.tif",
}

# =============================================================================

# =============================================================================

STAT_GROUPS = {
    "national": None,
    "provincial": "省级",

    "city": "city_name_norm",
    "county": "县级码",
    "Eastern_Region": "Eastern_Region",
    "North_South": "North_South",
    "city_level": "city_level",
    # "Urban_Rural": "Urban_Rural",
    # "Poor_County": "Poor_County",
}


# =============================================================================

# =============================================================================

CI_YEARS = list(YEARS)
GDP_XLSX = Path(os.environ.get("NC_CI_GDP_XLSX", DATA_ROOT / "中国城市数据库v202603版（2000-2024年）.xlsx"))
GDP_SHEET = "sheet1"
MINORITY_CSV = Path(os.environ.get("NC_CI_MINORITY_CSV", DATA_ROOT / "少数民族数据" / "七普县级少数民族人口比重.csv"))
MUNICIPALITIES = ["北京市", "上海市", "天津市", "重庆市"]
REGIONS = ["Eastern", "NotEastern"]
LEGACY_MINORITY_GT_ZERO = os.environ.get("NC_LEGACY_MINORITY_GT_ZERO", "0").strip().lower() in {"1", "true", "yes", "y"}
ETHNIC_THRESHOLD_PCT = float(os.environ.get("NC_CI_ETHNIC_THRESHOLD_PCT", "9.0"))
CITY_LEVEL_MERGE_MAP = {"Type I Large City":"Large City","Type II Large City":"Large City","Medium-sized City":"Medium/Small City","Small City":"Medium/Small City"}
MEGA_LABEL, SUPER_LABEL, COMBINED_GROUP_NAME = "Mega City", "Super City", "Mega/Super City"
HIGHLIGHT_YEARS = None


COUNTY_GDP_XLSX = Path(os.environ.get(
    "NC_COUNTY_GDP_XLSX",
    DATA_ROOT / "county_gdp_panel_137_cities_clean_2014_2023.xlsx",
)).expanduser().resolve()
COUNTY_GDP_SHEET = os.environ.get("NC_COUNTY_GDP_SHEET", "county_gdp_panel")


# =============================================================================

# =============================================================================

def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


# =============================================================================

# =============================================================================
ANALYSIS_YEARS = list(YEARS)
BASE_YEAR = min(ANALYSIS_YEARS)
END_YEAR = max(ANALYSIS_YEARS)
CHANGE_YEARS = [y for y in ANALYSIS_YEARS if y > BASE_YEAR]
SERVICE_SCOPE = service_scope_tag()
PROFILE = SPEED_PROFILE


SOCIO_END_YEAR = int(os.environ.get("NC_SOCIO_END_YEAR", END_YEAR))
if not BASE_YEAR <= SOCIO_END_YEAR <= END_YEAR:
    raise ValueError(f"SOCIO_END_YEAR={SOCIO_END_YEAR} 必须位于 {BASE_YEAR}~{END_YEAR}")

# =============================================================================

# =============================================================================

HOSPITAL_DATA_DIR = _env_path("NC_HOSPITAL_DATA_DIR", DATA_ROOT / "beds")


POP_FLOW_CSV = _env_path("NC_POP_FLOW_CSV", DATA_ROOT / "城市迁徙" / "地级市人口流动率14-23.csv")
POP_FLOW_XLSX = _env_path("NC_POP_FLOW_XLSX", DATA_ROOT / "城市迁徙" / "地级市-人口流动率（2000-2024年）.xlsx")
POP_FLOW_SHEET = os.environ.get("NC_POP_FLOW_SHEET", "回归填补")


GDP_CITY_CSV = _env_path("NC_GDP_CITY_CSV", DATA_ROOT / "城市级人均GDP" / "人均GDP14-23.csv")
GDP_CITY_RAW_CSV = _env_path("NC_GDP_CITY_RAW_CSV", DATA_ROOT / "城市级人均GDP" / "人均GDP.csv")
CITY_DATABASE_XLSX = _env_path("NC_CITY_DATABASE_XLSX", DATA_ROOT / "中国城市数据库v202603版（2000-2024年）.xlsx")
CITY_DATABASE_SHEET = os.environ.get("NC_CITY_DATABASE_SHEET", "") or None


COUNTY_SHP = _env_path("NC_COUNTY_SHP", ADMIN_SHP)

# =============================================================================

# =============================================================================
CITY_DYNAMICS_ROOT = RESULT_ROOT / "4_1_city_dynamics"
HOSPITAL_CHANGES_ROOT = RESULT_ROOT / "hospital_changes"
SEE_CIE_ANNUAL_ROOT = RESULT_ROOT / "see_cie_annual"
SEE_CIE_PANEL_ROOT = RESULT_ROOT / "see_cie_regression_panel"
SEE_CIE_REGRESSION_ROOT = RESULT_ROOT / "see_cie_regression"
FIGURE_ROOT = RESULT_ROOT / "Figure"
FIGURE5_ROOT = FIGURE_ROOT / "Figure 5"
SHAPLEY_FIGURE_ROOT = FIGURE_ROOT / "Shapley_decomposition"

# =============================================================================

# =============================================================================


MODE_COASTAL = [
    "辽宁省", "河北省", "天津市", "山东省", "江苏省", "上海市",
    "浙江省", "福建省", "广东省", "广西壮族自治区", "海南省",
]
CITY_ORDER_4 = ["Medium/Small City", "Large City", "Super City", "Mega City"]


ALLOW_FUZZY_HOSPITAL_MATCH = os.environ.get("NC_ALLOW_FUZZY_HOSPITAL_MATCH", "0").strip().lower() in {"1", "true", "yes", "y"}
HOSPITAL_FUZZY_MATCH_MAX_KM = float(os.environ.get("NC_HOSPITAL_FUZZY_MATCH_MAX_KM", "1.0"))
HOSPITAL_NAME_SIMILARITY_MIN = float(os.environ.get("NC_HOSPITAL_NAME_SIMILARITY_MIN", "0.80"))
HOSPITAL_EXACT_MATCH_REVIEW_KM = float(os.environ.get("NC_HOSPITAL_EXACT_MATCH_REVIEW_KM", "20.0"))
if HOSPITAL_FUZZY_MATCH_MAX_KM <= 0 or HOSPITAL_EXACT_MATCH_REVIEW_KM <= 0 or not 0 <= HOSPITAL_NAME_SIMILARITY_MIN <= 1:
    raise ValueError("医院匹配阈值配置无效")

