# Reproduction data

This directory contains the processed analytical inputs consumed by `reproduce.py`. The files are organized by analysis module so that the main results can be regenerated without rebuilding the computationally intensive upstream geospatial products.

## Directory overview

| Directory | Content | Used for |
|---|---|---|
| `2_2_multiscale_analysis/` | Annual travel-time and accessibility summaries at multiple spatial scales | Figures 1–3 and related summary statistics |
| `3_2_ci_analysis/` | GDP-ranked concentration index estimates and plotting coordinates | Figure 4 |
| `4_2_shapley_decomposition/` | Eight counterfactual scenario summaries and reference decomposition output | Exact Shapley recomputation |
| `see_cie_regression_panel/` | City-level SEE/CIE analytical panel | Figure 5, main regressions, and robustness analyses |
| `static/administrative_boundaries/` | 2023 province, city, and county boundaries | Mapped figures and spatial analysis |

## Multiscale travel-time and accessibility summaries

`2_2_multiscale_analysis/` contains annual summary tables for 2014–2024.

- `travel_time/` contains population-weighted nearest-hospital travel-time statistics at national, provincial, city, and county scales.
- `accessibility/` contains population-weighted Ga2SFCA accessibility statistics at the same scales, together with city-size-group summaries.
- `derived_groups/` contains the grouped summaries used for regional and urban/rural comparisons.

Common fields include `Year`, administrative identifiers, population totals, population-weighted medians, coverage measures, and inequality statistics where applicable.

## Concentration index analysis

`3_2_ci_analysis/` contains the GDP-ranked concentration index (CI) results used in the manuscript.

| File | Content |
|---|---|
| `acc_CI_results_by_GDP.csv` | National annual accessibility CI estimates and population coverage |
| `acc_CI_results_by_GDP_region.csv` | Annual CI estimates by Eastern / Non-Eastern region |
| `acc_CI_results_by_GDP_city_level.csv` | Annual CI estimates by city-size group |
| `plot_inputs/kde_curves.csv` | Population-weighted distribution coordinates used in Figure 4 |
| `plot_inputs/ci_curve_points.csv` | Concentration-curve coordinates and corresponding CI values used in Figure 4 |

The CI calculations rank observations by per-capita GDP and weight accessibility by population.

## Shapley decomposition

`4_2_shapley_decomposition/scenario_stats.csv` contains the outcome statistics for all eight combinations of the three Shapley factors. Scenario codes follow the order **road–population–hospital**: `0` denotes the 2014 state and `1` the 2024 state. For example, `A101` uses 2024 road conditions, 2014 population, and 2024 hospital supply.

The decomposition covers population-weighted accessibility, Gini, Theil, and Atkinson (ε = 0.5). `shapley_summary.csv` is retained as a reference output; `reproduce.py` recalculates the exact Shapley values from `scenario_stats.csv` before plotting.

## SEE/CIE regression panel

File: `see_cie_regression_panel/city_2014_2024_index.csv`

Each row represents one city analysis unit. Expansion variables aggregate changes over 2015–2024. Accessibility and inequality variables contain the 2014 baseline, 2024 endpoint, and endpoint-minus-baseline change.

### Identification and city grouping

| Variable | Definition |
|---|---|
| `省级` | Province-level unit |
| `地级` | City/prefecture analysis unit |
| `city_level` | City-size group used in the manuscript |

### SEE/CIE measures

| Variable | Definition / unit |
|---|---|
| `new_hosp_beds` | Beds added through newly opened service sites, cumulative 2015–2024 |
| `expanded_beds` | Beds added through capacity expansion at existing service sites, cumulative 2015–2024 |
| `decreased_beds` | Beds removed through capacity contraction at existing service sites, cumulative 2015–2024 |
| `closed_beds` | Beds removed through service-site closure, cumulative 2015–2024 |
| `net_SEE_beds` | Net SEE beds = `new_hosp_beds - closed_beds` |
| `net_CIE_beds` | Net CIE beds = `expanded_beds - decreased_beds` |
| `net_total_beds` | Total net bed change = `net_SEE_beds + net_CIE_beds` |
| `city_pop` | Mean city population across 2014–2024 |
| `city_SEE` | Net SEE beds per 10,000 population |
| `city_CIE` | Net CIE beds per 10,000 population |
| `city_TotalNetExpansion` | `city_SEE + city_CIE`, per 10,000 population |
| `Dominance` | `city_SEE - city_CIE` |

For regression estimation, `code/utils/regression.py` creates `city_TotalExpansion = city_SEE + city_CIE` and standardizes the continuous variables used by the models.

### Socioeconomic and demographic covariates

| Variable | Definition |
|---|---|
| `GDP_2014` | Per-capita GDP in 2014 |
| `GDP_growth` | Change in per-capita GDP from 2014 to 2024 |
| `GDP_growth_pct` | Percentage change in per-capita GDP from 2014 to 2024 |
| `Ppo_NetIn` | Mean annual net population inflow over 2014–2024 |
| `Ppo_NetIn_rate` | Cumulative net population inflow divided by the summed registered population over 2014–2024 |
| `ResPop_2014` | Resident population in 2014 |
| `ResPop_growth` | Change in resident population from 2014 to 2024 |
| `ResPop_growth_rate` | Resident-population growth rate from 2014 to 2024 |
| `pop_density_mean` | Mean population density over 2014–2024, persons per km² |
| `FiscalRevenue_2014` | Local general-budget fiscal revenue in 2014 |
| `FiscalRevenue_pc_2014` | Baseline fiscal revenue per resident |
| `ln_FiscalRevenue_pc_2014` | Natural logarithm of baseline per-capita fiscal revenue |

### Accessibility and inequality outcomes

The suffix `_2014` denotes the baseline value, `_2024` the endpoint value, and `_delta` the 2024-minus-2014 change.

| Variable family | Definition |
|---|---|
| `acc_*` | Population-weighted Ga2SFCA accessibility |
| `gini_*` | Population-weighted Gini coefficient of accessibility |
| `theil_*` | Population-weighted Theil index of accessibility |
| `atkinson_05_*` | Population-weighted Atkinson index with ε = 0.5 |
| `zero_access_pop_pct_*` | Percentage of population with zero modeled accessibility |
| `p90_p10_*` | 90th-to-10th percentile accessibility ratio, where defined |
| `p80_p20_*` | 80th-to-20th percentile accessibility ratio, where defined |

The main SEE/CIE models use `acc_delta`, `gini_delta`, `theil_delta`, and `atkinson_05_delta`, together with their corresponding baseline values and model covariates.

## Administrative boundaries

`static/administrative_boundaries/` contains the 2023 province, city, and county shapefiles used by the map-generating scripts and the spatial-error robustness analysis.
