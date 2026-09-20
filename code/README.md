# Code

`reproduce.py` in the repository root is the public reproduction entry point. Files in this directory are analysis components rather than alternative top-level runners.

## Upstream and analysis stages

The numbered scripts follow the computational sequence used in the project:

```text
0_1  prepare base data
0_2  prepare OSM routing graphs
1_1  build travel-time matrices
1_2  calculate Ga2SFCA accessibility
2_1  match multiscale administrative attributes
2_2  calculate multiscale statistics
3_1  prepare CI matched data
3_2  calculate concentration indices
4_1  prepare city dynamics covariates
4_2  run Shapley decomposition
5_1-5_5  construct hospital expansion variables and regression panel
5_6  estimate SEE/CIE regression models
5_8-5_10  spatial/province-FE robustness analyses
```

The public workflow replaces several expensive upstream stages with released intermediate data under `data/reproduction/`.

## Figure scripts

```text
fig1_1_travel_time_boxplot.py
fig1_2_travel_time_province_panels.py
fig1_3_travel_time_threshold_pies.py
fig2_1_time_accessibility_rank.py
fig2_3_city_county_accessibility_change.py
fig3_1_inequality_trends.py
fig3_2_multiscale_accessibility_gap.py
fig4_1_frequency_distribution_2014_2024.py
fig4_2_ci_plots.py
fig5_1_pathway_share_final.py
fig5_2_accessibility_scale_reversal_box.py
fig5_3_see_cie_effect_plots.py
plot_shapley_decomposition.py
```

`config.py` is a shared configuration module and is not intended to be executed directly.

`reconstruct_from_raw_inputs.py` is the optional author/HPC workflow for reconstruction from large raw inputs. It is not part of the standard public reproduction path.
