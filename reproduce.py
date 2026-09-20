#!/usr/bin/env python3
"""Reproduce the manuscript figures and SEE/CIE regression results from released data."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

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


def validate_release_inputs() -> list[str]:
    errors: list[str] = []
    required = [
        REPRO_ROOT / "2_2_multiscale_analysis" / "travel_time" / "national_travel_time_stats.csv",
        REPRO_ROOT / "2_2_multiscale_analysis" / "travel_time" / "provincial_travel_time_stats.csv",
        REPRO_ROOT / "2_2_multiscale_analysis" / "travel_time" / "city_travel_time_stats.csv",
        REPRO_ROOT / "2_2_multiscale_analysis" / "travel_time" / "county_travel_time_stats.csv",
        REPRO_ROOT / "2_2_multiscale_analysis" / "accessibility" / "national_acc_stats.csv",
        REPRO_ROOT / "2_2_multiscale_analysis" / "accessibility" / "provincial_acc_stats.csv",
        REPRO_ROOT / "2_2_multiscale_analysis" / "accessibility" / "city_acc_stats.csv",
        REPRO_ROOT / "2_2_multiscale_analysis" / "accessibility" / "county_acc_stats.csv",
        REPRO_ROOT / "2_2_multiscale_analysis" / "accessibility" / "city_level_acc_stats.csv",
        REPRO_ROOT / "2_2_multiscale_analysis" / "derived_groups" / "Coastal_Inland_acc_stats.csv",
        REPRO_ROOT / "2_2_multiscale_analysis" / "derived_groups" / "Urban_Rural_acc_stats.csv",
        REPRO_ROOT / "3_2_ci_analysis" / "acc_CI_results_by_GDP.csv",
        REPRO_ROOT / "3_2_ci_analysis" / "acc_CI_results_by_GDP_region.csv",
        REPRO_ROOT / "3_2_ci_analysis" / "acc_CI_results_by_GDP_city_level.csv",
        REPRO_ROOT / "3_2_ci_analysis" / "plot_inputs" / "kde_curves.csv",
        REPRO_ROOT / "3_2_ci_analysis" / "plot_inputs" / "ci_curve_points.csv",
        REPRO_ROOT / "4_2_shapley_decomposition" / "shapley_summary.csv",
        REPRO_ROOT / "see_cie_regression_panel" / "city_2014_2024_index.csv",
    ]
    for path in required:
        if not path.exists():
            errors.append(f"missing release input: {path.relative_to(REPO_ROOT)}")

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
    shutil.copy2(
        WORK_ROOT / "4_2_shapley_decomposition" / "shapley_summary.csv",
        WORK_ROOT / "shapley_summary.csv",
    )


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
