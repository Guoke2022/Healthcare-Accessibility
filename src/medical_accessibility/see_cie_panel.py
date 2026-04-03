from __future__ import annotations

from pathlib import Path

import pandas as pd


CITY_LEVEL_MERGE_MAP = {
    "Type I Large City": "Large City",
    "Type II Large City": "Large City",
    "Medium-sized City": "Medium/Small City",
    "Small City": "Medium/Small City",
}

DIRECT_MUNICIPALITIES = ["北京市", "上海市", "天津市", "重庆市"]

BASELINE_SOURCE_COLS = ["pop_median", "pop_gini", "pop_theil", "pop_MLD", "p90_p10", "p80_p20"]
BASELINE_RENAME_MAP = {
    "pop_median": "acc_2014",
    "pop_gini": "gini_2014",
    "pop_theil": "theil_2014",
    "pop_MLD": "MLD_2014",
    "p90_p10": "p90_p10_2014",
    "p80_p20": "p80_p20_2014",
}


def _load_yearly_city_panel(yearly_city_dir: Path, start_year: int, end_year: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        file_path = yearly_city_dir / f"city_SEE_CIE_{year}.csv"
        if not file_path.exists():
            continue
        df = pd.read_csv(file_path)
        df["city_level"] = df["city_level"].replace(CITY_LEVEL_MERGE_MAP)
        frames.append(
            df[
                [
                    "省级",
                    "地级",
                    "city_level",
                    "city_pop",
                    "new_hosp_beds",
                    "expanded_beds",
                    "acc_delta",
                    "gini_delta",
                    "theil_delta",
                    "MLD_delta",
                    "p90_p10_delta",
                    "p80_p20_delta",
                ]
            ].copy()
        )

    if not frames:
        raise FileNotFoundError(f"No city_SEE_CIE_YYYY.csv files found in {yearly_city_dir}")

    df_all = pd.concat(frames, ignore_index=True)
    aggregated = (
        df_all.groupby(["省级", "地级", "city_level"], as_index=False)
        .agg(
            city_pop=("city_pop", "mean"),
            new_hosp_beds=("new_hosp_beds", "sum"),
            expanded_beds=("expanded_beds", "sum"),
            acc_delta=("acc_delta", "sum"),
            gini_delta=("gini_delta", "sum"),
            theil_delta=("theil_delta", "sum"),
            MLD_delta=("MLD_delta", "sum"),
            p90_p10_delta=("p90_p10_delta", "sum"),
            p80_p20_delta=("p80_p20_delta", "sum"),
        )
    )
    aggregated["city_SEE"] = aggregated["new_hosp_beds"] / aggregated["city_pop"] * 10000
    aggregated["city_CIE"] = aggregated["expanded_beds"] / aggregated["city_pop"] * 10000
    mask = aggregated["city_SEE"].notna() | aggregated["city_CIE"].notna()
    aggregated.loc[mask, "Dominance"] = aggregated.loc[mask, "city_SEE"].fillna(0) - aggregated.loc[mask, "city_CIE"].fillna(0)
    return aggregated


def _merge_gdp_controls(df: pd.DataFrame, gdp_csv: Path) -> pd.DataFrame:
    gdp = pd.read_csv(gdp_csv)
    gdp_growth = (
        gdp[gdp["year"].isin([2014, 2023])]
        .pivot(index="城市", columns="year", values="GDP_per")
        .assign(
            GDP_growth=lambda x: x[2023] - x[2014],
            GDP_growth_pct=lambda x: (x[2023] - x[2014]) / x[2014] * 100,
        )
        .rename(columns={2014: "GDP_2014"})
        .reset_index()[["城市", "GDP_2014", "GDP_growth", "GDP_growth_pct"]]
    )
    return df.merge(gdp_growth, left_on="地级", right_on="城市", how="left").drop(columns=["城市"])


def _merge_population_flow_controls(df: pd.DataFrame, population_flow_csv: Path) -> pd.DataFrame:
    popflow = pd.read_csv(population_flow_csv).rename(columns={"年份": "year"})
    years = list(range(2014, 2024))

    df_netin = (
        popflow[popflow["year"].isin(years)]
        .pivot(index="城市", columns="year", values="人口净流入(万人)")
        .assign(Ppo_NetIn=lambda x: x[years].mean(axis=1))
        .reset_index()[["城市", "Ppo_NetIn"]]
    )

    df_netin_rate = (
        popflow[popflow["year"].isin(years)]
        .groupby("城市", as_index=False)
        .apply(
            lambda group: pd.Series(
                {
                    "Ppo_NetIn_rate": group["人口净流入(万人)"].sum() / group["户籍人口数(万人)"].sum(),
                }
            )
        )
        .reset_index(drop=True)
    )

    df_respop = (
        popflow[popflow["year"].isin(years)]
        .sort_values(["城市", "year"])
        .groupby("城市", as_index=False)
        .apply(
            lambda group: pd.Series(
                {
                    "ResPop_2014": group["常住人口数(万人)"].iloc[0],
                    "ResPop_growth": group["常住人口数(万人)"].iloc[-1] - group["常住人口数(万人)"].iloc[0],
                    "ResPop_growth_rate": (
                        group["常住人口数(万人)"].iloc[-1] - group["常住人口数(万人)"].iloc[0]
                    )
                    / group["常住人口数(万人)"].iloc[0],
                }
            )
        )
        .reset_index(drop=True)
    )

    city_popflow = df_netin.merge(df_netin_rate, on="城市").merge(df_respop, on="城市")
    return df.merge(city_popflow, left_on="地级", right_on="城市", how="left").drop(columns=["城市"])


def _merge_population_density(df: pd.DataFrame, population_density_xlsx: Path) -> pd.DataFrame:
    pop_density = pd.read_excel(population_density_xlsx)
    density_year = (
        pop_density[pop_density["年份"].isin(range(2014, 2025))]
        .groupby("城市", as_index=False)
        .agg(pop_density_mean=("人口密度(人／平方公里)", "mean"))
    )
    return df.merge(density_year, left_on="地级", right_on="城市", how="left").drop(columns=["城市"])


def _merge_baseline_accessibility(
    df: pd.DataFrame,
    provincial_stats_csv: Path,
    city_stats_csv: Path,
    county_stats_csv: Path,
) -> pd.DataFrame:
    df_acc_provincial = pd.read_csv(provincial_stats_csv, encoding="utf-8")
    df_acc_city = pd.read_csv(city_stats_csv, encoding="utf-8")
    df_acc_county = pd.read_csv(county_stats_csv, encoding="utf-8")

    df_acc_2014 = df_acc_city[df_acc_city["Year"] == 2014][["地级"] + BASELINE_SOURCE_COLS]
    result = df.merge(df_acc_2014, on="地级", how="left").rename(columns=BASELINE_RENAME_MAP)

    mask_muni = result["acc_2014"].isna() & result["地级"].isin(DIRECT_MUNICIPALITIES)
    if mask_muni.any():
        df_acc_prov_year = df_acc_provincial[df_acc_provincial["Year"] == 2014]
        prov_lookup = df_acc_prov_year.set_index("省级")[BASELINE_SOURCE_COLS].rename(columns=BASELINE_RENAME_MAP)
        city_names = result.loc[mask_muni, "地级"]
        result.loc[mask_muni, list(prov_lookup.columns)] = prov_lookup.reindex(city_names).values

    mask_county = result["acc_2014"].isna()
    if mask_county.any():
        df_acc_county_year = df_acc_county[df_acc_county["Year"] == 2014]
        county_lookup = (
            df_acc_county_year.drop_duplicates(subset=["县级"])
            .set_index("县级")[BASELINE_SOURCE_COLS]
            .rename(columns=BASELINE_RENAME_MAP)
        )
        missing_names = result.loc[mask_county, "地级"]
        result.loc[mask_county, list(county_lookup.columns)] = county_lookup.reindex(missing_names).values

    return result


def build_regression_panel(
    yearly_city_dir: Path,
    gdp_csv: Path,
    population_flow_csv: Path,
    population_density_xlsx: Path,
    provincial_stats_csv: Path,
    city_stats_csv: Path,
    county_stats_csv: Path,
    output_csv: Path,
    start_year: int = 2015,
    end_year: int = 2024,
) -> pd.DataFrame:
    """Build the city-level regression panel used in the SEE/CIE regressions."""

    panel = _load_yearly_city_panel(yearly_city_dir, start_year=start_year, end_year=end_year)
    panel = _merge_gdp_controls(panel, gdp_csv)
    panel = _merge_population_flow_controls(panel, population_flow_csv)
    panel = _merge_population_density(panel, population_density_xlsx)
    panel = _merge_baseline_accessibility(panel, provincial_stats_csv, city_stats_csv, county_stats_csv)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return panel
