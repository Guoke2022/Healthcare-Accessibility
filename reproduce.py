#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduce manuscript figures and downstream SEE/CIE analyses.

The public workflow starts from compact released intermediate data under
``data/reproduction``. Large national routing, Ga2SFCA construction, GURS
classification, grid-level CI matching, and hospital-level SEE/CIE construction
are intentionally outside the default reproduction path.
"""
from __future__ import annotations

import argparse
import json
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
    "5_4_see_cie_descriptives.py",
    "fig5_2_accessibility_scale_reversal_box.py",
    "5_6_see_cie_regression.py",
    "fig5_3_see_cie_effect_plots.py",
    "fig5_1_pathway_share_final.py",
]


def _find_boundary(layer: str) -> Path | None:
    root = REPRO_ROOT / "static" / "administrative_boundaries"
    preferred = root / f"china_{layer}_2023.shp"
    if preferred.exists():
        return preferred
    candidates = sorted(root.glob(f"*{layer}*.shp")) if root.exists() else []
    return candidates[0] if len(candidates) == 1 else None


def validate_release_inputs() -> list[str]:
    errors: list[str] = []

    multiscale = REPRO_ROOT / "2_2_multiscale_analysis"
    required_multiscale = [
        "travel_time/national_travel_time_stats.csv",
        "travel_time/provincial_travel_time_stats.csv",
        "travel_time/city_travel_time_stats.csv",
        "travel_time/county_travel_time_stats.csv",
        "accessibility/national_acc_stats.csv",
        "accessibility/provincial_acc_stats.csv",
        "accessibility/city_acc_stats.csv",
        "accessibility/county_acc_stats.csv",
        "accessibility/city_level_acc_stats.csv",
        "derived_groups/Coastal_Inland_acc_stats.csv",
        "derived_groups/Urban_Rural_acc_stats.csv",
    ]
    for relative in required_multiscale:
        path = multiscale / relative
        if not path.exists():
            errors.append(f"missing multiscale release file: {path}")

    ci = REPRO_ROOT / "3_2_ci_analysis"
    required_ci = [
        "acc_CI_results_by_GDP.csv",
        "plot_inputs/kde_curves.csv",
        "plot_inputs/ci_curve_points.csv",
    ]
    for relative in required_ci:
        path = ci / relative
        if not path.exists():
            errors.append(f"missing Figure 4 release file: {path}")

    shapley = REPRO_ROOT / "4_2_shapley_decomposition" / "shapley_summary.csv"
    if not shapley.exists():
        errors.append(f"missing Shapley plot input: {shapley}")

    annual = REPRO_ROOT / "5_3_see_cie_annual"
    city_files = sorted(annual.glob("city_SEE_CIE_*.csv")) if annual.exists() else []
    if len(city_files) != 10:
        errors.append(f"expected 10 annual city SEE/CIE files (2015-2024) in {annual}; found {len(city_files)}")

    panel = REPRO_ROOT / "5_5_see_cie_regression_panel" / "city_2014_2024_index.csv"
    if not panel.exists():
        errors.append(f"missing SEE/CIE regression panel: {panel}")

    for layer in ["province", "city", "county"]:
        shp = _find_boundary(layer)
        if shp is None:
            errors.append(f"missing {layer} boundary shapefile")
            continue
        for suffix in [".shx", ".dbf", ".prj", ".cpg"]:
            sidecar = shp.with_suffix(suffix)
            if not sidecar.exists():
                errors.append(f"missing shapefile sidecar: {sidecar}")

    return errors


def _replace_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def stage_inputs() -> None:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    _replace_tree(
        REPRO_ROOT / "2_2_multiscale_analysis",
        WORK_ROOT / "2_2_multiscale_analysis" / PUBLIC_SCOPE / PUBLIC_PROFILE,
    )
    _replace_tree(
        REPRO_ROOT / "3_2_ci_analysis",
        WORK_ROOT / "3_2_ci_analysis" / PUBLIC_SCOPE / PUBLIC_PROFILE,
    )
    for name in ["4_2_shapley_decomposition", "5_3_see_cie_annual", "5_5_see_cie_regression_panel"]:
        _replace_tree(REPRO_ROOT / name, WORK_ROOT / name)

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
    boundaries = {layer: _find_boundary(layer) for layer in ["province", "city", "county"]}
    if boundaries["county"] is not None:
        env["NC_ADMIN_SHP"] = str(boundaries["county"])
        env["NC_COUNTY_SHP"] = str(boundaries["county"])
    if boundaries["city"] is not None:
        env["NC_CITY_SHP"] = str(boundaries["city"])
    if boundaries["province"] is not None:
        env["NC_PROVINCE_SHP"] = str(boundaries["province"])
    return env


def run_scripts(scripts: list[str], env: dict[str, str]) -> None:
    for name in scripts:
        script = CODE_ROOT / name
        if not script.exists():
            raise FileNotFoundError(f"Missing repository script: {script}")
        print(f"\n>>> {name}", flush=True)
        subprocess.run([sys.executable, str(script)], cwd=str(REPO_ROOT), env=env, check=True)


def _reset_generated_outputs(section: str) -> None:
    targets = []
    if section in {"all", "figures"}:
        targets.extend([
            OUTPUT_ROOT / "figures" / "Figure_1",
            OUTPUT_ROOT / "figures" / "Figure_2",
            OUTPUT_ROOT / "figures" / "Figure_3",
            OUTPUT_ROOT / "figures" / "Figure_4",
            OUTPUT_ROOT / "figures" / "Shapley_decomposition",
            OUTPUT_ROOT / "map_layers",
        ])
    if section in {"all", "see-cie"}:
        targets.extend([
            OUTPUT_ROOT / "figures" / "Figure_5",
            OUTPUT_ROOT / "regression",
            OUTPUT_ROOT / "tables",
        ])
    for path in targets:
        if path.exists():
            shutil.rmtree(path)
    manifest = OUTPUT_ROOT / "generated_files.txt"
    if manifest.exists():
        manifest.unlink()


def _copy_tree_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)


def collect_outputs(section: str) -> list[Path]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if section in {"all", "figures"}:
        figure_root = WORK_ROOT / "Figure"
        for number in [1, 2, 3, 4]:
            _copy_tree_if_exists(figure_root / f"Figure {number}", OUTPUT_ROOT / "figures" / f"Figure_{number}")
        _copy_tree_if_exists(figure_root / "Shapley_decomposition", OUTPUT_ROOT / "figures" / "Shapley_decomposition")
        _copy_tree_if_exists(figure_root / "Map_layers", OUTPUT_ROOT / "map_layers")

    if section in {"all", "see-cie"}:
        _copy_tree_if_exists(WORK_ROOT / "Figure" / "Figure 5", OUTPUT_ROOT / "figures" / "Figure_5")
        _copy_tree_if_exists(WORK_ROOT / "5_6_see_cie_regression", OUTPUT_ROOT / "regression")
        table_source = WORK_ROOT / "5_4_see_cie_descriptive"
        table_target = OUTPUT_ROOT / "tables"
        table_target.mkdir(parents=True, exist_ok=True)
        for name in ["city_accessibility_absolute_relative_change.csv", "city_level_accessibility_change_summary.csv"]:
            source = table_source / name
            if source.exists():
                shutil.copy2(source, table_target / name)

    files = sorted(path for path in OUTPUT_ROOT.rglob("*") if path.is_file() and path.name != "README.md")
    (OUTPUT_ROOT / "generated_files.txt").write_text(
        "\n".join(str(path.relative_to(OUTPUT_ROOT)) for path in files) + "\n",
        encoding="utf-8",
    )
    return files


def clean_generated() -> None:
    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    if OUTPUT_ROOT.exists():
        for child in OUTPUT_ROOT.iterdir():
            if child.name != "README.md":
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section",
        choices=["all", "figures", "see-cie"],
        default="all",
        help="Run the full public reproduction, manuscript figures 1-4/Shapley only, or SEE/CIE/Figure 5 only.",
    )
    parser.add_argument("--check-only", action="store_true", help="Validate released reproduction inputs and exit.")
    parser.add_argument("--clean-only", action="store_true", help="Remove generated outputs and temporary work files, then exit.")
    parser.add_argument("--keep-work", action="store_true", help="Keep the temporary .reproduction_work directory for debugging.")
    args = parser.parse_args()

    if args.clean_only:
        clean_generated()
        print("Generated outputs and temporary work files removed.")
        return

    manifest = REPRO_ROOT / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing reproduction manifest: {manifest}")
    json.loads(manifest.read_text(encoding="utf-8"))

    errors = validate_release_inputs()
    if errors:
        print("Reproduction-data validation failed:\n", file=sys.stderr)
        for item in errors:
            print(f"  - {item}", file=sys.stderr)
        raise SystemExit(2)

    print("Reproduction-data validation passed.")
    if args.check_only:
        return

    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    _reset_generated_outputs(args.section)
    stage_inputs()
    env = runtime_env()

    if args.section in {"all", "figures"}:
        run_scripts(FIGURE_SCRIPTS, env)
    if args.section in {"all", "see-cie"}:
        run_scripts(SEE_CIE_SCRIPTS, env)

    generated = collect_outputs(args.section)
    if not args.keep_work and WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)

    print("\nReproduction completed successfully.")
    print(f"Generated files: {len(generated)}")
    print(f"Output directory: {OUTPUT_ROOT}")
    print("See outputs/generated_files.txt for the complete file list.")


if __name__ == "__main__":
    main()
