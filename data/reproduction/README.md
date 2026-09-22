# Public reproduction data

This directory contains the processed analytical inputs used by the compact public reproduction workflow. These files are intended to reproduce the reported summary analyses, main figures, Shapley decomposition, SEE/CIE regressions, and selected robustness analyses without redistributing the large upstream population, road-network, raster, and hospital-processing inputs.

## Directory overview

| Directory | Unit / content | Main use |
|---|---|---|
| `2_2_multiscale_analysis/` | National, province, city, county, and grouped annual summary statistics | Figures 1–3 and related national/multiscale summaries |
| `3_2_ci_analysis/` | GDP-ranked concentration-index tables and compact plotting inputs | Figure 4 |
| `4_2_shapley_decomposition/` | Outcome summaries for the eight road–population–hospital counterfactual scenarios (`A000`–`A111`) plus a reference Shapley summary | Exact three-factor Shapley recomputation |
| `see_cie_regression_panel/` | One row per city analysis unit | Main SEE/CIE regressions, Figure 5, province fixed-effects robustness, and spatial-error robustness |
| `static/administrative_boundaries/` | Fixed 2023 province/city/county boundary shapefiles | Mapped figures and spatial robustness |

## SEE/CIE regression panel

File:

`see_cie_regression_panel/city_2014_2024_index.csv`

Each row represents one city analysis unit. Expansion measures summarize changes over 2015–2024 and are normalized by the city's mean population over 2014–2024 where indicated. Accessibility and inequality columns report the 2014 baseline, 2024 endpoint, and endpoint-minus-baseline change.

### Identification and city grouping

| Variable | Definition |
|---|---|
| `省级` | Province-level unit |
| `地级` | City/prefecture analysis unit |
| `city_level` | City-size group used in the manuscript |

### Hospital expansion

| Variable | Definition / unit |
|---|---|
| `new_hosp_beds` | Beds added through newly opened eligible hospital service sites, cumulative beds |
| `expanded_beds` | Beds added through capacity expansion at existing eligible service sites, cumulative beds |
| `decreased_beds` | Beds removed through capacity contraction at existing eligible service sites, cumulative beds |
| `closed_beds` | Beds removed through closure of eligible service sites, cumulative beds |
| `net_SEE_beds` | Net spatial extensive expansion beds = `new_hosp_beds - closed_beds` |
| `net_CIE_beds` | Net capacity intensive expansion beds = `expanded_beds - decreased_beds` |
| `net_total_beds` | Total net bed change = `net_SEE_beds + net_CIE_beds` |
| `city_pop` | Mean city population across 2014–2024 |
| `city_SEE` | Net SEE beds per 10,000 population |
| `city_CIE` | Net CIE beds per 10,000 population |
| `city_TotalNetExpansion` | `city_SEE + city_CIE`, per 10,000 population |
| `Dominance` | `city_SEE - city_CIE`; positive values indicate relatively more SEE than CIE |

For model estimation, `code/utils/regression.py` creates `city_TotalExpansion = city_SEE + city_CIE` and standardizes the continuous predictors used in the regression models.

### Socioeconomic and demographic covariates

| Variable | Definition |
|---|---|
| `GDP_2014` | Baseline city GDP |
| `GDP_growth` | Change in GDP over the study-period socioeconomic series |
| `GDP_growth_pct` | GDP growth expressed as a percentage of the baseline |
| `Ppo_NetIn` | Net population inflow measure used in the city covariate dataset |
| `Ppo_NetIn_rate` | Net population inflow rate |
| `ResPop_2014` | Baseline resident population |
| `ResPop_growth` | Change in resident population |
| `ResPop_growth_rate` | Resident-population growth rate |
| `pop_density_mean` | Mean population density over 2014–2024, persons per km² |
| `FiscalRevenue_2014` | Baseline local general-budget fiscal revenue |
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

The main SEE/CIE regression workflow uses `acc_delta`, `gini_delta`, `theil_delta`, and `atkinson_05_delta`, together with their corresponding baseline values and the covariates listed above.

## Notes on scope

The public files in this directory are processed analytical products rather than raw national-scale geospatial inputs. The compact workflow does not rerun annual OSM routing, construction of the 1-km Ga2SFCA surfaces, GURS raster classification, grid-level socioeconomic matching, or the longitudinal hospital/campus reconstruction. Upstream implementation code is retained in the repository, while large externally sourced inputs are not redistributed because of data-volume and licensing constraints.

For the reviewer-facing workflow, run from the repository root:

```bash
python reproduce.py --check-only
python reproduce.py
python reproduce.py --section robustness
```
