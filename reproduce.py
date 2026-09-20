#!/usr/bin/env python3
"""Reproduce the manuscript figures and SEE/CIE regression results from released data."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
CODE_ROOT = REPO_ROOT / "code"
REPRO_ROOT = REPO_ROOT / "data" / "reproduction"
WORK_ROOT = REPO_ROOT / ".reproduction_work"
OUTPUT_ROOT = REPO_ROOT / "outputs"
PUBLIC_SCOPE = "same_province"
PUBLIC_PROFILE = "chn_osm_default"

FIGURE_SCRIPTS = [
    "fig1_1_travel_time_boxplot.py",
    "fig1_2_travel_time_province_panels.py",
    "fig1_3_travel_time_threshold_pies.py",
    "fig2_1_time_accessibility_rank.py",
    "fig2_3_city_county_accessibility_change.py",
    "fig3_1_inequality_trends.py",
    "fig3_2_multiscale_accessibility_gap.py",
    "fig4_1_frequency_distribution_2014_2024.py",
    "fig4_2_ci_plots.py",
    "recompute_shapley_from_scenarios.py",
    "plot_shapley_decomposition.py",
]

SEE_CIE_SCRIPTS = [
    "fig5_1_pathway_share_final.py",
    "fig5_2_accessibility_scale_reversal_box.py",
    "run_see_cie_regressions.py",
    "fig5_3_see_cie_effect_plots.py",
]


def _find_boundary(layer: str) -> Path | None:
    root = REPRO_ROOT / "static" / "administrative_boundaries"
    path = root / f"china_{layer}_2023.shp"
    return path if path.exists() else None


def _validate_csv_columns(path: Path, required_columns: set[str], errors: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        errors.append(f"missing release input: {path.relative_to(REPO_ROOT)}")
        return None
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        errors.append(f"cannot read {path.relative_to(REPO_ROOT)}: {exc}")
        return None
    missing = sorted(required_columns - set(df.columns))
    if missing:
        errors.append(f"{path.relative_to(REPO_ROOT)} missing columns: {missing}")
    if df.empty:
        errors.append(f"{path.relative_to(REPO_ROOT)} is empty")
    return df


def _validate_year_endpoints(df: pd.DataFrame | None, path: Path, errors: list[str]) -> None:
    if df is None or "Year" not in df.columns:
        return
    years = set(pd.to_numeric(df["Year"], errors="coerce").dropna().astype(int).tolist())
    for year in (2014, 2024):
        if year not in years:
            errors.append(f"{path.relative_to(REPO_ROOT)} does not contain year {year}")


def validate_release_inputs() -> list[str]:
    errors: list[str] = []

    csv_requirements: dict[Path, set[str]] = {
        REPRO_ROOT / "2_2_multiscale_analysis" / "travel_time" / "national_travel_time_stats.csv": {"Year", "pop_median"},
        REPRO_ROOT / "2_2_multiscale_analysis" / "travel_time" / "provincial_travel_time_stats.csv": {"Year", "省级", "省级码", "pop_median", "pop_pct_lt_60"},
        REPRO_ROOT / "2_2_multiscale_analysis" / "travel_time" / "city_travel_time_stats.csv": {"Year", "地级", "地级码", "pop_median"},
        REPRO_ROOT / "2_2_multiscale_analysis" / "travel_time" / "county_travel_time_stats.csv": {"Year", "县级", "县级码", "pop_median"},
        REPRO_ROOT / "2_2_multiscale_analysis" / "accessibility" / "national_acc_stats.csv": {"Year", "pop_median", "pop_gini", "pop_theil", "pop_atkinson_05"},
        REPRO_ROOT / "2_2_multiscale_analysis" / "accessibility" / "provincial_acc_stats.csv": {"Year", "省级", "省级码", "pop_median"},
        REPRO_ROOT / "2_2_multiscale_analysis" / "accessibility" / "city_acc_stats.csv": {"Year", "地级", "地级码", "pop_median"},
        REPRO_ROOT / "2_2_multiscale_analysis" / "accessibility" / "county_acc_stats.csv": {"Year", "县级", "县级码", "pop_median"},
        REPRO_ROOT / "2_2_multiscale_analysis" / "accessibility" / "city_level_acc_stats.csv": {"Year", "city_level", "pop_median"},
        REPRO_ROOT / "2_2_multiscale_analysis" / "derived_groups" / "Coastal_Inland_acc_stats.csv": {"Year", "Coastal_Inland", "pop_median"},
        REPRO_ROOT / "2_2_multiscale_analysis" / "derived_groups" / "Urban_Rural_acc_stats.csv": {"Year", "Urban_Rural", "pop_median"},
        REPRO_ROOT / "3_2_ci_analysis" / "acc_CI_results_by_GDP.csv": {"year", "CI"},
        REPRO_ROOT / "3_2_ci_analysis" / "acc_CI_results_by_GDP_region.csv": {"year", "region", "CI"},
        REPRO_ROOT / "3_2_ci_analysis" / "acc_CI_results_by_GDP_city_level.csv": {"year", "city_level", "CI"},
        REPRO_ROOT / "3_2_ci_analysis" / "plot_inputs" / "kde_curves.csv": {"variable", "year", "x", "density"},
        REPRO_ROOT / "3_2_ci_analysis" / "plot_inputs" / "ci_curve_points.csv": {"curve_family", "group_name", "rank_var", "year", "x", "y", "ci"},
        REPRO_ROOT / "4_2_shapley_decomposition" / "scenario_stats.csv": {"scenario", "pop_median", "pop_gini", "pop_theil", "pop_atkinson_05"},
        REPRO_ROOT / "4_2_shapley_decomposition" / "shapley_summary.csv": {"outcome", "factor", "contribution_abs", "share_pct"},
        REPRO_ROOT / "see_cie_regression_panel" / "city_2014_2024_index.csv": {
            "省级", "地级", "city_level", "city_SEE", "city_CIE", "city_TotalNetExpansion",
            "acc_2014", "acc_2024", "gini_2014", "gini_2024", "theil_2014", "theil_2024",
            "atkinson_05_2014", "atkinson_05_2024", "ln_FiscalRevenue_pc_2014",
        },
    }

    loaded: dict[Path, pd.DataFrame | None] = {}
    for path, columns in csv_requirements.items():
        loaded[path] = _validate_csv_columns(path, columns, errors)

    for path, df in loaded.items():
        # Urban/rural summaries intentionally cover only the GURS-supported years
        # (2014-2016 and 2019-2021), so they do not have a 2024 endpoint.
        if "Urban_Rural_acc_stats.csv" in path.name:
            continue
        if "2_2_multiscale_analysis" in str(path):
            _validate_year_endpoints(df, path, errors)
        elif "acc_CI_results" in path.name and df is not None and "year" in df.columns:
            years = set(pd.to_numeric(df["year"], errors="coerce").dropna().astype(int).tolist())
            for year in (2014, 2024):
                if year not in years:
                    errors.append(f"{path.relative_to(REPO_ROOT)} does not contain year {year}")

    scenario_path = REPRO_ROOT / "4_2_shapley_decomposition" / "scenario_stats.csv"
    scenario_df = loaded.get(scenario_path)
    if scenario_df is not None and "scenario" in scenario_df.columns:
        expected = {f"A{r}{p}{h}" for r in (0, 1) for p in (0, 1) for h in (0, 1)}
        actual = set(scenario_df["scenario"].astype(str))
        if actual != expected or len(scenario_df) != 8:
            errors.append(
                "Shapley scenario table must contain exactly A000-A111 once each; "
                f"missing={sorted(expected-actual)}, extra={sorted(actual-expected)}, rows={len(scenario_df)}"
            )
        elif scenario_df["scenario"].astype(str).duplicated().any():
            errors.append("Shapley scenario table contains duplicate scenario codes")

    panel_path = REPRO_ROOT / "see_cie_regression_panel" / "city_2014_2024_index.csv"
    panel_df = loaded.get(panel_path)
    if panel_df is not None and {"省级", "地级"}.issubset(panel_df.columns):
        duplicates = panel_df.duplicated(["省级", "地级"], keep=False)
        if duplicates.any():
            errors.append(f"SEE/CIE panel contains {int(duplicates.sum())} duplicate province-city rows")

    for layer in ["province", "city", "county"]:
        shp = _find_boundary(layer)
        if shp is None:
            errors.append(f"missing {layer} boundary shapefile")
            continue
        for suffix in [".shx", ".dbf", ".prj", ".cpg"]:
            if not shp.with_suffix(suffix).exists():
                errors.append(f"missing shapefile sidecar: {shp.with_suffix(suffix).relative_to(REPO_ROOT)}")
    return errors


def _replace_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def clean_work() -> None:
    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)


def clean_outputs() -> None:
    if OUTPUT_ROOT.exists():
        for item in OUTPUT_ROOT.iterdir():
            if item.name == "README.md":
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()


def stage_inputs() -> None:
    clean_work()
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    _replace_tree(
        REPRO_ROOT / "2_2_multiscale_analysis",
        WORK_ROOT / "2_2_multiscale_analysis" / PUBLIC_SCOPE / PUBLIC_PROFILE,
    )
    _replace_tree(
        REPRO_ROOT / "3_2_ci_analysis",
        WORK_ROOT / "3_2_ci_analysis" / PUBLIC_SCOPE / PUBLIC_PROFILE,
    )
    _replace_tree(REPRO_ROOT / "4_2_shapley_decomposition", WORK_ROOT / "4_2_shapley_decomposition")
    _replace_tree(REPRO_ROOT / "see_cie_regression_panel", WORK_ROOT / "see_cie_regression_panel")


def runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "NC_PROJECT_ROOT": str(REPO_ROOT),
        "NC_DATA_ROOT": str(REPO_ROOT / "data"),
        "NC_MAIN_RESULT_ROOT": str(WORK_ROOT),
        "NC_RESULT_ROOT": str(WORK_ROOT),
        "NC_ANALYSIS_MODE": "main",
        "NC_PUBLIC_REPRODUCTION": "1",
        "NC_FIGURE_SERVICE_SCOPE": PUBLIC_SCOPE,
        "NC_FIGURE_SPEED_PROFILE": PUBLIC_PROFILE,
        "NC_SPEED_PROFILE": PUBLIC_PROFILE,
        "NC_MAKE_PLOTS": "1",
    })
    county = _find_boundary("county")
    city = _find_boundary("city")
    province = _find_boundary("province")
    if county:
        env["NC_ADMIN_SHP"] = str(county)
        env["NC_COUNTY_SHP"] = str(county)
    if city:
        env["NC_CITY_SHP"] = str(city)
    if province:
        env["NC_PROVINCE_SHP"] = str(province)
    return env


def run_scripts(scripts: list[str], env: dict[str, str]) -> None:
    for name in scripts:
        script = CODE_ROOT / name
        if not script.exists():
            raise FileNotFoundError(f"Missing script: {script}")
        print(f"\n>>> {name}", flush=True)
        subprocess.run([sys.executable, str(script)], cwd=REPO_ROOT, env=env, check=True)


def _copy_tree_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)


def collect_outputs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    figure_root = WORK_ROOT / "Figure"
    for number in range(1, 6):
        _copy_tree_if_exists(figure_root / f"Figure {number}", OUTPUT_ROOT / "figures" / f"Figure_{number}")
    _copy_tree_if_exists(figure_root / "Shapley_decomposition", OUTPUT_ROOT / "figures" / "Shapley_decomposition")
    _copy_tree_if_exists(figure_root / "Map_layers", OUTPUT_ROOT / "map_layers")
    _copy_tree_if_exists(WORK_ROOT / "see_cie_regression", OUTPUT_ROOT / "regression")

    generated = sorted(
        str(path.relative_to(OUTPUT_ROOT)).replace("\\", "/")
        for path in OUTPUT_ROOT.rglob("*")
        if path.is_file() and path.name != "generated_files.txt"
    )
    (OUTPUT_ROOT / "generated_files.txt").write_text("\n".join(generated) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=["all", "figures", "see-cie"], default="all")
    parser.add_argument("--check-only", action="store_true", help="Validate released inputs and exit.")
    parser.add_argument("--clean-only", action="store_true", help="Remove generated work/output files and exit.")
    args = parser.parse_args()

    if args.clean_only:
        clean_work()
        clean_outputs()
        print("Generated reproduction files removed.")
        return

    errors = validate_release_inputs()
    if errors:
        print("Reproduction-data validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        raise SystemExit(2)
    print("Reproduction-data validation passed.")
    if args.check_only:
        return

    stage_inputs()
    env = runtime_env()
    if args.section in {"all", "figures"}:
        run_scripts(FIGURE_SCRIPTS, env)
    if args.section in {"all", "see-cie"}:
        run_scripts(SEE_CIE_SCRIPTS, env)
    collect_outputs()
    print(f"\nReproduction completed. See: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
