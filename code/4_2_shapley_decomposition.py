# -*- coding: utf-8 -*-
"""4_2 Shapley decomposition for endpoint changes in accessibility and inequality.

Supported modes
---------------
three_component
    Standard decomposition with Road (R), Population (P), and Hospital supply (H).
    All 2^3 endpoint/counterfactual combinations are evaluated.

fixed_road_two_component
    Robustness decomposition with mapped road-network conditions held fixed at the
    configured modern-road state. Only Population and Hospital supply are Shapley
    players, giving four H/P states. This mode is used by 6_4 and must not be
    interpreted as estimating a causal road-infrastructure contribution.

The baseline/target years are controlled by NC_SHAPLEY_BASE_YEAR and
NC_SHAPLEY_TARGET_YEAR (default: min/max of the active analysis years), allowing
2014-2024 threshold/branch analyses and 2015-2024 population-source sensitivity.

Primary outputs
---------------
<RESULT_ROOT>/shapley_summary.csv
<RESULT_ROOT>/4_2_shapley_decomposition/scenario_stats.csv
<RESULT_ROOT>/4_2_shapley_decomposition/shapley_summary.csv
"""
from __future__ import annotations

import itertools
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    ACCESSIBILITY_ROOT,
    ANALYSIS_MODE,
    PREPARED_ROOT,
    ROAD_GRAPH_ROOT,
    RESULT_ROOT,
    RUN_POLICY,
    SHAPLEY_ROUTER_THREADS,
    SHAPLEY_SCENARIO_WORKERS,
    SPEED_PROFILE,
    YEARS,
    service_scope_tag,
)
from utils.multiscale import calculate_accessibility_stats, validate_complete_province_parts


BASE_YEAR = int(os.environ.get("NC_SHAPLEY_BASE_YEAR", str(min(YEARS))))
TARGET_YEAR = int(os.environ.get("NC_SHAPLEY_TARGET_YEAR", str(max(YEARS))))
SHAPLEY_MODE = os.environ.get("NC_SHAPLEY_MODE", "three_component").strip().lower()
ALLOWED_SHAPLEY_MODES = {"three_component", "fixed_road_two_component"}
if SHAPLEY_MODE not in ALLOWED_SHAPLEY_MODES:
    raise ValueError(f"NC_SHAPLEY_MODE={SHAPLEY_MODE!r} invalid; choose from {sorted(ALLOWED_SHAPLEY_MODES)}")

SERVICE_SCOPE = service_scope_tag()
PROFILE = SPEED_PROFILE
CODE_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = RESULT_ROOT / "4_2_shapley_decomposition"
SCENARIO_ROOT = OUTPUT_ROOT / "scenarios"
LOG_ROOT = OUTPUT_ROOT / "logs"
SUMMARY_ROOT_PATH = RESULT_ROOT / "shapley_summary.csv"

# Standard three-component decomposition.  The fixed-road robustness uses a
# dedicated two-component H/P decomposition while the mapped road state is held
# fixed by the sensitivity-only config hook.
if SHAPLEY_MODE == "three_component":
    SCENARIOS = {
        "A000": {"road": BASE_YEAR, "population": BASE_YEAR, "hospital": BASE_YEAR, "legacy_label": "base road + base pop + base hospital"},
        "A001": {"road": BASE_YEAR, "population": BASE_YEAR, "hospital": TARGET_YEAR, "legacy_label": "base road + base pop + target hospital"},
        "A010": {"road": BASE_YEAR, "population": TARGET_YEAR, "hospital": BASE_YEAR, "legacy_label": "base road + target pop + base hospital"},
        "A011": {"road": BASE_YEAR, "population": TARGET_YEAR, "hospital": TARGET_YEAR, "legacy_label": "base road + target pop + target hospital"},
        "A100": {"road": TARGET_YEAR, "population": BASE_YEAR, "hospital": BASE_YEAR, "legacy_label": "target road + base pop + base hospital"},
        "A101": {"road": TARGET_YEAR, "population": BASE_YEAR, "hospital": TARGET_YEAR, "legacy_label": "target road + base pop + target hospital"},
        "A110": {"road": TARGET_YEAR, "population": TARGET_YEAR, "hospital": BASE_YEAR, "legacy_label": "target road + target pop + base hospital"},
        "A111": {"road": TARGET_YEAR, "population": TARGET_YEAR, "hospital": TARGET_YEAR, "legacy_label": "target road + target pop + target hospital"},
    }
    MIXED_CODES = ["A001", "A010", "A011", "A100", "A101", "A110"]
    FACTORS = ("road", "population", "hospital")
    FACTOR_BITS = {"road": 0, "population": 1, "hospital": 2}
    BASE_CODE, TARGET_CODE = "A000", "A111"
else:
    # Road is not a player here.  All four states use the same fixed road
    # network; only population and hospital supply switch between endpoints.
    SCENARIOS = {
        "A000": {"road": TARGET_YEAR, "population": BASE_YEAR, "hospital": BASE_YEAR, "legacy_label": "fixed road + base pop + base hospital"},
        "A001": {"road": TARGET_YEAR, "population": BASE_YEAR, "hospital": TARGET_YEAR, "legacy_label": "fixed road + base pop + target hospital"},
        "A010": {"road": TARGET_YEAR, "population": TARGET_YEAR, "hospital": BASE_YEAR, "legacy_label": "fixed road + target pop + base hospital"},
        "A011": {"road": TARGET_YEAR, "population": TARGET_YEAR, "hospital": TARGET_YEAR, "legacy_label": "fixed road + target pop + target hospital"},
    }
    MIXED_CODES = ["A001", "A010"]
    FACTORS = ("population", "hospital")
    FACTOR_BITS = {"population": 1, "hospital": 2}
    BASE_CODE, TARGET_CODE = "A000", "A011"

OUTCOMES = {
    "accessibility": "pop_median",
    "gini": "pop_gini",
    "theil": "pop_theil",
    "atkinson_05": "pop_atkinson_05",
}


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label}不存在：{path}")


def scenario_dir(code: str) -> Path:
    return SCENARIO_ROOT / code


def scenario_travel_root(code: str) -> Path:
    return scenario_dir(code) / "1_1_travel_time" / SERVICE_SCOPE


def scenario_access_root(code: str) -> Path:
    return scenario_dir(code) / "1_2_accessibility" / SERVICE_SCOPE


def main_access_root(year: int) -> Path:
    return ACCESSIBILITY_ROOT / str(year) / SERVICE_SCOPE


def validate_prerequisites() -> None:
    allowed_modes = {
        "main",
        "threshold_sensitivity",
        "population_sensitivity",
        "fixed_road_sensitivity",
        "branch_threshold_sensitivity",
        "cross_province_sensitivity",
    }
    if ANALYSIS_MODE not in allowed_modes:
        raise RuntimeError(f"4_2 Shapley does not support analysis_mode={ANALYSIS_MODE!r}")
    if BASE_YEAR == TARGET_YEAR:
        raise ValueError("Shapley baseline and target years must differ")
    if BASE_YEAR not in YEARS or TARGET_YEAR not in YEARS:
        raise ValueError(
            f"Shapley years {BASE_YEAR}->{TARGET_YEAR} must lie within the active analysis years {YEARS[0]}->{YEARS[-1]}"
        )
    if SHAPLEY_MODE == "fixed_road_two_component" and not os.environ.get("NC_FIXED_OSM_ANALYSIS_YEAR", "").strip():
        raise RuntimeError(
            "fixed_road_two_component requires NC_FIXED_OSM_ANALYSIS_YEAR so the road state is truly held fixed"
        )
    for y in sorted({BASE_YEAR, TARGET_YEAR}):
        ensure_exists(PREPARED_ROOT / str(y) / "hospitals.parquet", f"{y} hospitals.parquet")
        ensure_exists(PREPARED_ROOT / str(y) / "population_parts", f"{y} population_parts")
        ensure_exists(ROAD_GRAPH_ROOT / str(y) / "road_graph_summary.csv", f"{y} road graph")
        ensure_exists(ROAD_GRAPH_ROOT / str(y) / "_stage_input.json", f"{y} road graph manifest")
        ensure_exists(main_access_root(y) / PROFILE / "grids", f"{y} main accessibility grids")


def validate_base_endpoint_topology_fallback() -> None:
    """Confirm annual-road endpoint fallback lineage when that fallback is expected."""
    if SHAPLEY_MODE == "fixed_road_two_component":
        log("Fixed-road H/P decomposition: annual topology fallback is intentionally disabled; skip fallback-count check.")
        return
    path = main_access_root(BASE_YEAR) / PROFILE / "hospital_R.parquet"
    ensure_exists(path, f"{BASE_YEAR} main hospital_R")
    h = pd.read_parquet(path)

    legacy_repair_cols = {
        "osm_ratio_repair_applied",
        "R_per_10000_before_osm_repair",
    } & set(h.columns)
    if legacy_repair_cols:
        raise RuntimeError(
            f"{BASE_YEAR} 主分析仍包含旧的 R-repair 字段 {sorted(legacy_repair_cols)}。\n"
            "请用新的 1_2_calculate_accessibility.py 重跑 2014 主分析后再运行 Shapley。"
        )

    required = "osm_topology_fallback_applied"
    if required not in h.columns:
        raise RuntimeError(
            f"{BASE_YEAR} 主分析尚未使用新的 targeted OSM topology fallback。\n"
            "请先用新的 1_2_calculate_accessibility.py 重跑 2014；不需要重跑 1_1。"
        )

    n = int(h[required].fillna(False).astype(bool).sum())
    if n <= 0:
        raise RuntimeError(
            f"{BASE_YEAR} hospital_R 中没有任何 topology fallback 记录；"
            "与当前预设异常清单不一致，请检查 1_2 日志。"
        )
    log(f"A000 endpoint topology fallback 已确认：{BASE_YEAR} 共 {n} 家 hospital-road pairs。")


def run_script(script: Path, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        p = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(CODE_ROOT),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if p.returncode != 0:
        raise RuntimeError(f"{script.name} 失败，returncode={p.returncode}；详见 {log_path}")


def run_mixed_scenario(code: str) -> dict:
    spec = SCENARIOS[code]
    t0 = time.time()
    sdir = scenario_dir(code)
    sdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # Force direct execution of 1_1/1_2 main() rather than their all-year dispatchers.
    env["PIPELINE_YEAR"] = str(TARGET_YEAR)
    env["NC_COMPONENT_ROAD_YEAR"] = str(spec["road"])
    env["NC_COMPONENT_POPULATION_YEAR"] = str(spec["population"])
    env["NC_COMPONENT_HOSPITAL_YEAR"] = str(spec["hospital"])
    env["NC_COMPONENT_SCENARIO"] = code
    env["NC_TRAVEL_TIME_OUTPUT_ROOT_OVERRIDE"] = str(scenario_travel_root(code))
    env["NC_TRAVEL_TIME_INPUT_ROOT_OVERRIDE"] = str(scenario_travel_root(code))
    env["NC_ACCESSIBILITY_OUTPUT_ROOT_OVERRIDE"] = str(scenario_access_root(code))
    # Mixed scenarios are parallelized at the scenario level, so each child router must
    # receive only its allocated CPU share instead of inheriting the main-analysis value.
    env["NC_ROUTER_THREADS"] = str(SHAPLEY_ROUTER_THREADS)

    log(
        f"{code}: road={spec['road']}, population={spec['population']}, "
        f"hospital={spec['hospital']} ({spec['legacy_label']}) | "
        f"router_threads={SHAPLEY_ROUTER_THREADS}"
    )
    log(
        f"{code}: targeted OSM topology fallback由 1_2 统一处理；"
        "不插值/截断 R，不做 accessibility winsorization。"
    )
    run_script(CODE_ROOT / "1_1_build_travel_matrix.py", env, LOG_ROOT / f"{code}_1_1.log")
    run_script(CODE_ROOT / "1_2_calculate_accessibility.py", env, LOG_ROOT / f"{code}_1_2.log")
    return {"scenario": code, "elapsed_seconds": time.time() - t0, "status": "ok"}


def grid_parts(access_root: Path) -> list[Path]:
    parts = sorted((access_root / PROFILE / "grids").glob("province_*.parquet"))
    validate_complete_province_parts(parts, f"Shapley grids: {access_root}")
    return parts


def national_accessibility_stats(access_root: Path) -> dict:
    parts = grid_parts(access_root)
    frames = []
    for p in parts:
        use = ["population", "accessibility"]
        # Current 1_2 always writes grid_snap_distance_km, but retain compatibility.
        try:
            z = pd.read_parquet(p, columns=use + ["grid_snap_distance_km"])
        except Exception:
            z = pd.read_parquet(p, columns=use)
            z["grid_snap_distance_km"] = np.nan
        z = z.rename(columns={"population": "pop", "accessibility": "acc"})
        frames.append(z[["pop", "acc", "grid_snap_distance_km"]])
    df = pd.concat(frames, ignore_index=True)
    return calculate_accessibility_stats(df)


def collect_scenario_stats() -> pd.DataFrame:
    rows = []
    for code, spec in SCENARIOS.items():
        if code == BASE_CODE:
            root = main_access_root(BASE_YEAR)
            source_kind = "analysis_endpoint"
        elif code == TARGET_CODE:
            root = main_access_root(TARGET_YEAR)
            source_kind = "analysis_endpoint"
        else:
            root = scenario_access_root(code)
            source_kind = "mixed_counterfactual"
        ensure_exists(root / PROFILE / "grids", f"{code} accessibility grids")
        stats = national_accessibility_stats(root)
        rows.append({
            "scenario": code,
            "legacy_label": spec["legacy_label"],
            "road_year": spec["road"],
            "population_year": spec["population"],
            "hospital_year": spec["hospital"],
            "source_kind": source_kind,
            "profile": PROFILE,
            "service_scope": SERVICE_SCOPE,
            "pop_median": stats.get("pop_median"),
            "pop_mean": stats.get("pop_mean"),
            "pop_gini": stats.get("pop_gini"),
            "pop_theil": stats.get("pop_theil"),
            "pop_atkinson_05": stats.get("pop_atkinson_05"),
            "zero_access_pop_pct": stats.get("zero_access_pop_pct"),
            "pop_num": stats.get("pop_num"),
            "grid_num": stats.get("grid_num"),
            "accessibility_root": str(root),
        })
    out = pd.DataFrame(rows).sort_values("scenario").reset_index(drop=True)
    for col in OUTCOMES.values():
        if not np.isfinite(pd.to_numeric(out[col], errors="coerce")).all():
            bad = out.loc[~np.isfinite(pd.to_numeric(out[col], errors="coerce")), ["scenario", col]]
            raise RuntimeError(f"Shapley scenario {col} 存在非有限值：\n{bad}")
    return out


def state_code(active: set[str]) -> str:
    bits = ["0", "0", "0"]
    for f in active:
        bits[FACTOR_BITS[f]] = "1"
    return "A" + "".join(bits)


def exact_shapley(values: dict[str, float]) -> dict[str, float]:
    """Exact Shapley over all permutations of the active factor set."""
    phi = {f: 0.0 for f in FACTORS}
    perms = list(itertools.permutations(FACTORS))
    for perm in perms:
        active: set[str] = set()
        current = float(values[state_code(active)])
        for f in perm:
            active.add(f)
            nxt = float(values[state_code(active)])
            phi[f] += nxt - current
            current = nxt
    return {f: phi[f] / len(perms) for f in FACTORS}


def selftest_shapley_formula() -> None:
    """Synthetic efficiency/additivity checks for the active factor set."""
    if SHAPLEY_MODE == "three_component":
        additive = {}
        interactive = {}
        for code in SCENARIOS:
            r, p, h = map(int, code[1:])
            additive[code] = 10.0 + 2.0*r + 3.0*p + 5.0*h
            interactive[code] = (
                1.0 + 2.0*r + 3.0*p + 5.0*h
                + 7.0*r*p + 11.0*r*h + 13.0*p*h + 17.0*r*p*h
            )
        phi = exact_shapley(additive)
        expected = {"road": 2.0, "population": 3.0, "hospital": 5.0}
        for f in FACTORS:
            if not np.isclose(phi[f], expected[f], rtol=0.0, atol=1e-12):
                raise RuntimeError(f"Shapley synthetic additive self-test failed for {f}: {phi[f]} != {expected[f]}")
        total2 = interactive[TARGET_CODE] - interactive[BASE_CODE]
        phi2 = exact_shapley(interactive)
        if not np.isclose(sum(phi2.values()), total2, rtol=0.0, atol=1e-12):
            raise RuntimeError("Shapley synthetic interaction efficiency self-test failed")
    else:
        additive = {}
        interactive = {}
        for code in SCENARIOS:
            pbit, hbit = int(code[2]), int(code[3])
            additive[code] = 10.0 + 3.0*pbit + 5.0*hbit
            interactive[code] = 1.0 + 3.0*pbit + 5.0*hbit + 7.0*pbit*hbit
        phi = exact_shapley(additive)
        expected = {"population": 3.0, "hospital": 5.0}
        for f in FACTORS:
            if not np.isclose(phi[f], expected[f], rtol=0.0, atol=1e-12):
                raise RuntimeError(f"Two-factor Shapley synthetic test failed for {f}: {phi[f]} != {expected[f]}")
        total2 = interactive[TARGET_CODE] - interactive[BASE_CODE]
        phi2 = exact_shapley(interactive)
        if not np.isclose(sum(phi2.values()), total2, rtol=0.0, atol=1e-12):
            raise RuntimeError("Two-factor Shapley interaction efficiency self-test failed")

def legacy_equal_four_marginal(values: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    """Reproduce the old script: 1/4-weight marginals, then proportional rescaling."""
    A = values
    raw = {
        "road": ((A["A100"]-A["A000"]) + (A["A110"]-A["A010"]) + (A["A101"]-A["A001"]) + (A["A111"]-A["A011"])) / 4.0,
        "population": ((A["A010"]-A["A000"]) + (A["A110"]-A["A100"]) + (A["A011"]-A["A001"]) + (A["A111"]-A["A101"])) / 4.0,
        "hospital": ((A["A001"]-A["A000"]) + (A["A101"]-A["A100"]) + (A["A011"]-A["A010"]) + (A["A111"]-A["A110"])) / 4.0,
    }
    total = A["A111"] - A["A000"]
    raw_sum = sum(raw.values())
    if not math.isfinite(raw_sum) or abs(raw_sum) < 1e-15:
        adjusted = {f: np.nan for f in FACTORS}
    else:
        scale = total / raw_sum
        adjusted = {f: raw[f] * scale for f in FACTORS}
    return raw, adjusted


def decompose(stats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexed = stats.set_index("scenario")
    exact_rows = []
    legacy_rows = []
    for outcome, col in OUTCOMES.items():
        values = {code: float(indexed.loc[code, col]) for code in SCENARIOS}
        total = values[TARGET_CODE] - values[BASE_CODE]
        exact = exact_shapley(values)
        exact_sum = sum(exact.values())
        if SHAPLEY_MODE == "three_component":
            legacy_raw, legacy_adj = legacy_equal_four_marginal(values)
            legacy_raw_sum = sum(legacy_raw.values())
        else:
            legacy_raw = legacy_adj = {f: np.nan for f in FACTORS}
            legacy_raw_sum = np.nan

        for factor in FACTORS:
            share = 100.0 * exact[factor] / total if abs(total) > 1e-15 else np.nan
            improvement_sign = 1.0 if outcome == "accessibility" else -1.0
            exact_rows.append({
                "outcome": outcome,
                "factor": factor,
                "factor_display": "Hospital supply" if factor == "hospital" else factor.title(),
                "contribution_abs": exact[factor],
                "contribution_to_improvement": exact[factor] * improvement_sign,
                "share_pct": share,
                "baseline_value": values[BASE_CODE],
                "target_value": values[TARGET_CODE],
                "total_change": total,
                "total_improvement": total * improvement_sign,
                "sum_check_residual": exact_sum - total,
                "method": f"exact_shapley_{math.factorial(len(FACTORS))}_permutation_average",
                "shapley_mode": SHAPLEY_MODE,
                "base_year": BASE_YEAR,
                "target_year": TARGET_YEAR,
                "profile": PROFILE,
                "service_scope": SERVICE_SCOPE,
            })
            legacy_share = 100.0 * legacy_adj[factor] / total if abs(total) > 1e-15 and np.isfinite(legacy_adj[factor]) else np.nan
            exact_share = share
            legacy_rows.append({
                "outcome": outcome,
                "factor": factor,
                "exact_contribution_abs": exact[factor],
                "exact_share_pct": exact_share,
                "legacy_raw_equal_quarter_contribution": legacy_raw[factor],
                "legacy_raw_sum_minus_total": legacy_raw_sum - total,
                "legacy_rescaled_contribution": legacy_adj[factor],
                "legacy_rescaled_share_pct": legacy_share,
                "exact_minus_legacy_share_pct": exact_share - legacy_share if np.isfinite(legacy_share) else np.nan,
            })

        if not np.isclose(exact_sum, total, rtol=1e-10, atol=1e-10):
            raise RuntimeError(
                f"Exact Shapley efficiency check failed for {outcome}: sum(phi)={exact_sum}, total={total}"
            )

    return pd.DataFrame(exact_rows), pd.DataFrame(legacy_rows)


def write_method_note() -> None:
    lines = [
        "# Shapley decomposition (current implementation)",
        "",
        f"- Period: {BASE_YEAR} to {TARGET_YEAR}.",
        f"- Mode: {SHAPLEY_MODE}.",
        f"- Active factors: {', '.join(FACTORS)}.",
        "- Hospital supply means the annual hospital configuration used by the main model: operating hospital locations/presence plus observed bed capacities. The old code called this factor `Bed`.",
        f"- Counterfactuals: all 2^{len(FACTORS)} = {len(SCENARIOS)} combinations; endpoint states reuse the ordinary analysis outputs and mixed states rerun the current routing and Ga2SFCA code.",
        f"- Formal estimator: exact Shapley value, computed over all {math.factorial(len(FACTORS))} factor-order permutations. No post-hoc scaling is applied.",
        "- In fixed_road_two_component mode, mapped road-network conditions are held fixed and are not a Shapley player; only Hospital and Population are decomposed.",
        "- The legacy 1/4+rescale audit is written only for the standard three-component decomposition.",
        "- Population is not only a weighting variable: its year selects the LandScan demand surface used in S1/S2 and therefore belongs inside each counterfactual recomputation.",
        "- Mixed scenarios are isolated under `scenarios/`; they never overwrite ordinary annual results.",
        "- Confirmed OSM topology failures are handled by the same targeted travel-vector fallback in annual and counterfactual runs. Only W_ij for flagged hospital-road pairs is replaced by a reliable adjacent-road-year vector; R_j is then recomputed normally from scenario-specific population and hospital supply.",
        "- No direct R interpolation/capping and no accessibility winsorization are used in the formal Shapley analysis.",
        "",
    ]
    (OUTPUT_ROOT / "README_SHAPLEY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    t0 = time.time()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    SCENARIO_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    selftest_shapley_formula()
    validate_prerequisites()
    validate_base_endpoint_topology_fallback()
    log(f"Shapley {BASE_YEAR}->{TARGET_YEAR} | mode={SHAPLEY_MODE} | profile={PROFILE} | scope={SERVICE_SCOPE}")
    log(f"Exact Shapley over {math.factorial(len(FACTORS))} permutations; active factors={FACTORS}.")

    scenario_workers = min(SHAPLEY_SCENARIO_WORKERS, len(MIXED_CODES))
    log(
        f"mixed scenarios parallelism={scenario_workers} | "
        f"router_threads/scenario={SHAPLEY_ROUTER_THREADS} | "
        f"max_router_threads={scenario_workers * SHAPLEY_ROUTER_THREADS}"
    )
    run_rows_by_code: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=scenario_workers, thread_name_prefix="shapley") as pool:
        future_to_code = {pool.submit(run_mixed_scenario, code): code for code in MIXED_CODES}
        for future in as_completed(future_to_code):
            code = future_to_code[future]
            # future.result() propagates any scenario failure and stops formal decomposition.
            row = future.result()
            run_rows_by_code[code] = row
            log(f"{code}: completed in {row['elapsed_seconds'] / 60:.1f} min")

    # Preserve the canonical scenario order in the audit CSV regardless of completion order.
    run_rows = [run_rows_by_code[code] for code in MIXED_CODES]
    pd.DataFrame(run_rows).to_csv(OUTPUT_ROOT / "scenario_run_status.csv", index=False, encoding="utf-8-sig")

    stats = collect_scenario_stats()
    stats.to_csv(OUTPUT_ROOT / "scenario_stats.csv", index=False, encoding="utf-8-sig")
    exact, legacy = decompose(stats)
    exact.to_csv(OUTPUT_ROOT / "shapley_summary.csv", index=False, encoding="utf-8-sig")
    # Root-level canonical file is what 7_1 consumes.
    exact.to_csv(SUMMARY_ROOT_PATH, index=False, encoding="utf-8-sig")
    legacy_path = OUTPUT_ROOT / "shapley_legacy_comparison.csv"
    if SHAPLEY_MODE == "three_component":
        legacy.to_csv(legacy_path, index=False, encoding="utf-8-sig")
    elif legacy_path.exists():
        legacy_path.unlink()

    manifest = {
        "created_at": datetime.now().isoformat(),
        "base_year": BASE_YEAR,
        "target_year": TARGET_YEAR,
        "analysis_mode": ANALYSIS_MODE,
        "shapley_mode": SHAPLEY_MODE,
        "active_factors": list(FACTORS),
        "base_code": BASE_CODE,
        "target_code": TARGET_CODE,
        "profile": PROFILE,
        "service_scope": SERVICE_SCOPE,
        "run_policy": RUN_POLICY,
        "factor_semantics": {
            "road": "road network / routing conditions from selected year",
            "population": "active population-demand surface from selected year",
            "hospital": "hospital presence/location plus observed bed capacity from selected year",
        },
        "formal_method": f"exact Shapley; average marginal contribution over all {math.factorial(len(FACTORS))} permutations",
        "legacy_method": ("equal 1/4 weighting of four marginals plus proportional rescaling (audit only)" if SHAPLEY_MODE == "three_component" else "not applicable for fixed-road two-component decomposition"),
        "osm_topology_fallback": {
            "level": "hospital-to-grid travel-time vector (W_ij)",
            "scope": "pre-specified topology failures plus suspicious new-hospital x old-road counterfactual pairs",
            "R_handling": "R_j recomputed normally from scenario-specific D_j; no interpolation or cap",
            "accessibility_winsorization": False,
        },
        "scenarios": SCENARIOS,
        "elapsed_seconds": time.time() - t0,
    }
    (OUTPUT_ROOT / "shapley_metadata.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_method_note()
    log(f"完成：{SUMMARY_ROOT_PATH}")


if __name__ == "__main__":
    main()
