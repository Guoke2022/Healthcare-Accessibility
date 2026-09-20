# Reproduction data

This directory contains the compact inputs used by the public `reproduce.py` workflow. Large upstream routing, gridded accessibility, CI-matched grids, GURS rasters, and the complete hospital bed database are not included.

## Included inputs

```text
data/reproduction/
├─ 2_2_multiscale_analysis/          # Statistics used by Figures 1-3
├─ 3_2_ci_analysis/                  # CI tables and compact Figure 4 plot inputs
│  └─ plot_inputs/
│     ├─ kde_curves.csv
│     └─ ci_curve_points.csv
├─ 4_2_shapley_decomposition/        # Compact Shapley summaries
├─ 5_3_see_cie_annual/               # City-level annual SEE/CIE tables, 2015-2024
├─ 5_5_see_cie_regression_panel/     # Analysis-ready city regression panel
└─ static/
   └─ administrative_boundaries/
      ├─ china_province_2023.*
      ├─ china_city_2023.*
      └─ china_county_2023.*
```

`ci_curve_points.csv` and `kde_curves.csv` are compact plotting inputs derived from the non-released grid-level CI-matched data. They allow Figure 4 to be regenerated without distributing the approximately 1.16 GB matched grid dataset.

The `5_3_see_cie_annual` directory contains city-level annual SEE/CIE outputs used by Figure 5.1 and the descriptive preparation step for Figure 5.2. County-level annual SEE/CIE tables are not required by the public reproduction workflow and are therefore not included.

Run the input check from the repository root:

```bash
python reproduce.py --check-only
```
