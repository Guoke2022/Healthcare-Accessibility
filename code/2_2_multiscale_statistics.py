# -*- coding: utf-8 -*-
"""Module utilities for 2 2 multiscale statistics."""


from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from utils.cache import (
    build_fingerprint,
    clean_directory,
    prepare_stage_directory,
    stage_manifest_path,
)

from config import (
    YEARS,
    SERVICE_SCOPES,
    PROFILES,
    STAT_GROUPS,
    RUN_POLICY,
    MULTISCALE_ROOT,
    MULTISCALE_STATS_YEAR_WORKERS,
)
from utils.inequality_schema import INEQUALITY_SCHEMA_VERSION
from utils.multiscale import (
    read_matched_year,
    calculate_travel_time_stats,
    calculate_accessibility_stats,
    stats_output_dir,
)

CITY_LEVEL_MERGE_MAP = {
    "Type I Large City": "Large City",
    "Type II Large City": "Large City",
    "Medium-sized City": "Medium/Small City",
    "Small City": "Medium/Small City",
}


def extra_columns(groupby_col: str | None) -> list[str]:
    if groupby_col == "县级码":
        return ["县级"]
    if groupby_col in {"地级", "city_name_norm"}:
        return ["地级码"]
    if groupby_col == "省级":
        return ["省级码"]
    return []


def _expected_outputs(scope: str, profile: str) -> list[Path]:
    outputs: list[Path] = []
    for group_name in STAT_GROUPS:
        outputs.append(stats_output_dir(scope, profile, "travel_time") / f"{group_name}_travel_time_stats.csv")
        outputs.append(stats_output_dir(scope, profile, "accessibility") / f"{group_name}_acc_stats.csv")
    return outputs


def _input_manifests(scope: str, profile: str) -> list[Path]:
    manifests = [
        stage_manifest_path(MULTISCALE_ROOT / str(year) / scope / profile)
        for year in YEARS
    ]
    missing = [p for p in manifests if not p.exists()]
    if missing:
        sample = "\n".join(str(p) for p in missing[:5])
        raise FileNotFoundError(
            "2_2 缺少 2_1 stage manifest；请先完整运行 2_1_multiscale_match.py。\n"
            f"示例缺失：\n{sample}"
        )
    return manifests


def _stage_fingerprint(scope: str, profile: str) -> str:
    code_root = Path(__file__).resolve().parent
    manifests = _input_manifests(scope, profile)
    return build_fingerprint(
        config={
            "stage": "2_2_multiscale_statistics",
            "years": list(YEARS),
            "scope": scope,
            "profile": profile,
            "stat_groups": dict(STAT_GROUPS),
            "city_level_merge_map": dict(CITY_LEVEL_MERGE_MAP),

            "io_layout_version": 4,
            "inequality_schema_version": INEQUALITY_SCHEMA_VERSION,
            "inequality_population_universe": "finite_acc_ge_0_and_positive_population",
            "theil_city_decomposition": "exact_additive_within_between_by_city_name_norm",
        },
        files=[
            *manifests,
            Path(__file__).resolve(),
            code_root / "utils" / "multiscale.py",
            code_root / "utils" / "inequality_metrics.py",
            code_root / "utils" / "inequality_schema.py",
        ],
    )


def _all_required_columns() -> list[str]:
    # city_name_norm is required even for non-city grouping rows because every
    # accessibility result now carries an exact within-city/between-city Theil decomposition.
    cols = {"pop", "travel_time", "acc", "grid_snap_distance_km", "city_name_norm"}
    for groupby_col in STAT_GROUPS.values():
        if groupby_col:
            cols.add(groupby_col)
        cols.update(extra_columns(groupby_col))
    return sorted(cols)


def _new_results() -> dict[tuple[str, str], list[dict]]:
    return {
        (stat_kind, group_name): []
        for group_name in STAT_GROUPS
        for stat_kind in ("travel_time", "accessibility")
    }


def _extra_values(group: pd.DataFrame, extras: list[str]) -> dict:
    out = {}
    for extra in extras:
        vals = group[extra].dropna()
        out[extra] = vals.iloc[0] if len(vals) else pd.NA
    return out


def _append_normal_group_rows(
    *,
    df: pd.DataFrame,
    year: int,
    group_name: str,
    groupby_col: str | None,
    results: dict[tuple[str, str], list[dict]],
) -> None:
    """Helper for _append_normal_group_rows."""
    travel_sink = results[("travel_time", group_name)]
    acc_sink = results[("accessibility", group_name)]

    if groupby_col is None:
        travel_sink.append({"Year": year, **calculate_travel_time_stats(df)})
        acc_sink.append({"Year": year, **calculate_accessibility_stats(df)})
        return

    extras = extra_columns(groupby_col)
    for group_value, group in df.groupby(groupby_col, dropna=False):
        identity = {
            "Year": year,
            groupby_col: group_value,
            **_extra_values(group, extras),
        }
        travel_sink.append({**identity, **calculate_travel_time_stats(group)})
        acc_sink.append({**identity, **calculate_accessibility_stats(group)})


def _append_city_level_rows(
    *,
    df: pd.DataFrame,
    year: int,
    group_name: str,
    results: dict[tuple[str, str], list[dict]],
) -> None:
    """Helper for _append_city_level_rows."""


    travel_sink = results[("travel_time", group_name)]
    acc_sink = results[("accessibility", group_name)]

    for group_value, group in df.groupby("city_level", dropna=False):
        travel_sink.append({
            "Year": year,
            "city_level": group_value,
            **calculate_travel_time_stats(group),
        })

    merged_level = df["city_level"].map(CITY_LEVEL_MERGE_MAP).fillna(df["city_level"])
    for group_value, group in df.groupby(merged_level, dropna=False):
        acc_sink.append({
            "Year": year,
            "city_level": group_value,
            **calculate_accessibility_stats(group),
        })


def _process_year(
    year: int,
    scope: str,
    profile: str,
    columns: list[str],
) -> tuple[int, dict[tuple[str, str], list[dict]], int]:
    """Helper for _process_year."""
    try:
        df = read_matched_year(year, scope, profile, columns=columns)
    except (FileNotFoundError, KeyError) as e:
        raise RuntimeError(f"2_2 无法读取 {year}/{scope}/{profile}: {e}") from e

    n_grids = len(df)
    results = _new_results()
    for group_name, groupby_col in STAT_GROUPS.items():
        if groupby_col == "city_level":
            _append_city_level_rows(
                df=df,
                year=year,
                group_name=group_name,
                results=results,
            )
        else:
            _append_normal_group_rows(
                df=df,
                year=year,
                group_name=group_name,
                groupby_col=groupby_col,
                results=results,
            )
    del df
    return year, results, n_grids


def _merge_year_results(
    target: dict[tuple[str, str], list[dict]],
    yearly: dict[tuple[str, str], list[dict]],
) -> None:
    for key in target:
        target[key].extend(yearly[key])


def _write_results(scope: str, profile: str, results: dict[tuple[str, str], list[dict]]) -> None:
    for group_name, groupby_col in STAT_GROUPS.items():
        for stat_kind in ("travel_time", "accessibility"):
            result_df = pd.DataFrame(results[(stat_kind, group_name)])

            if (
                group_name == "city"
                and groupby_col == "city_name_norm"
                and "city_name_norm" in result_df.columns
            ):
                result_df = result_df.rename(columns={"city_name_norm": "地级"})

            if stat_kind == "travel_time":
                filename = f"{group_name}_travel_time_stats.csv"
            else:
                filename = f"{group_name}_acc_stats.csv"
            out_path = stats_output_dir(scope, profile, stat_kind) / filename
            result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"完成：{out_path} | rows={len(result_df):,}", flush=True)


def _run_scope_profile(scope: str, profile: str) -> None:
    stage_root = stats_output_dir(scope, profile, "travel_time").parent
    fingerprint = _stage_fingerprint(scope, profile)
    expected = _expected_outputs(scope, profile)

    status = prepare_stage_directory(
        stage_root,
        fingerprint,
        run_policy=RUN_POLICY,
        payload={"stage": "2_2_multiscale_statistics", "scope": scope, "profile": profile},
    )

    if status == "reused_same_input" and all(p.exists() for p in expected):
        print(
            f"2_2 cache hit | scope={scope} | profile={profile} | "
            f"输入未变且 {len(expected)} 个统计 CSV 完整，整阶段跳过。",
            flush=True,
        )
        return

    if status == "reused_same_input":

        missing = [p.name for p in expected if not p.exists()]
        print(
            f"2_2 输出不完整（missing={missing[:5]}），清空并重建当前 scope/profile。",
            flush=True,
        )
        clean_directory(stage_root)
        prepare_stage_directory(
            stage_root,
            fingerprint,
            run_policy="auto_clean",
            payload={"stage": "2_2_multiscale_statistics", "scope": scope, "profile": profile},
        )

    workers = min(int(MULTISCALE_STATS_YEAR_WORKERS), len(YEARS))
    print(f"\n{'=' * 90}", flush=True)
    print(
        f"2_2 | scope={scope} | profile={profile} | "
        f"year_workers={workers} | single-sort distribution stats",
        flush=True,
    )
    print("=" * 90, flush=True)

    columns = _all_required_columns()
    results = _new_results()

    if workers <= 1:
        for i, year in enumerate(YEARS, 1):
            finished_year, yearly, n_grids = _process_year(year, scope, profile, columns)
            _merge_year_results(results, yearly)
            print(
                f"2_2 [{i}/{len(YEARS)}] {finished_year}: "
                f"loaded {n_grids:,} grids once; all groups complete.",
                flush=True,
            )
    else:
        completed: dict[int, dict[tuple[str, str], list[dict]]] = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_year = {
                pool.submit(_process_year, year, scope, profile, columns): year
                for year in YEARS
            }
            done_count = 0
            for future in as_completed(future_to_year):
                year = future_to_year[future]
                try:
                    finished_year, yearly, n_grids = future.result()
                except Exception as e:
                    raise RuntimeError(
                        f"2_2 年份并行任务失败 | year={year} | scope={scope} | profile={profile}"
                    ) from e
                completed[finished_year] = yearly
                done_count += 1
                print(
                    f"2_2 [{done_count}/{len(YEARS)}] {finished_year}: "
                    f"loaded {n_grids:,} grids once; all groups complete.",
                    flush=True,
                )


        for year in YEARS:
            _merge_year_results(results, completed[year])

    _write_results(scope, profile, results)


def main():
    for scope in SERVICE_SCOPES:
        for profile in PROFILES:
            _run_scope_profile(scope, profile)


if __name__ == "__main__":
    main()
