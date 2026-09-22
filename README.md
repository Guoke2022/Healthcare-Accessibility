# Accessibility gains and uneven inequality reductions from hospital expansion in China

This repository contains the analysis source code, processed analytical inputs, and reproduction workflow for the manuscript **“Accessibility gains and uneven inequality reductions from hospital expansion in China.”** The released workflow regenerates the main accessibility and inequality summaries, concentration index results, exact Shapley decomposition, SEE/CIE regression analyses, and main-text figures from the processed data included in the repository.

## Quick start

Create the software environment from the repository root:

```bash
conda env create -f environment.yml
conda activate healthcare-accessibility
```

Validate the released reproduction inputs:

```bash
python reproduce.py --check-only
```

Run the main reproduction workflow:

```bash
python reproduce.py
```

Analyses can also be run separately:

```bash
python reproduce.py --section figures      # Figures 1–4 and Shapley decomposition
python reproduce.py --section see-cie      # SEE/CIE regressions and Figure 5
python reproduce.py --section robustness   # Province fixed-effects and spatial-error analyses
python reproduce.py --section all          # Main analyses plus robustness analyses
```

Generated files are written to `outputs/`. To remove generated outputs and temporary working files:

```bash
python reproduce.py --clean-only
```

## Reproduction analyses

### Accessibility and multiscale analysis

`data/reproduction/2_2_multiscale_analysis/` contains annual travel-time and Ga2SFCA accessibility summaries at national, provincial, city, and county scales, together with the grouped summaries used in the manuscript. These data are used to regenerate Figures 1–3 and the associated temporal and multiscale comparisons.

### Concentration index analysis

`data/reproduction/3_2_ci_analysis/` contains the GDP-ranked concentration index estimates used in the manuscript at national, regional, and city-size-group levels. The directory also contains the distribution and concentration-curve coordinates used to regenerate Figure 4.

### Shapley decomposition

`data/reproduction/4_2_shapley_decomposition/` contains outcome statistics for the eight combinations of 2014/2024 road-network conditions, population distribution, and hospital supply (`A000`–`A111`). `code/recompute_shapley_from_scenarios.py` recalculates the exact three-factor Shapley decomposition from these eight scenarios, verifies the efficiency property, and compares the recalculated values with the released reference summary before plotting.

### SEE/CIE analysis

`data/reproduction/see_cie_regression_panel/city_2014_2024_index.csv` is the analysis-ready city-level panel used for the Spatial Extensive Expansion (SEE) and Capacity Intensive Expansion (CIE) analyses. It contains cumulative expansion measures, baseline and endpoint accessibility and inequality outcomes, city-size groups, and model covariates. The workflow re-estimates the main SEE/CIE models and regenerates Figure 5.

The robustness workflow uses the same city-level panel to estimate province fixed-effects specifications and spatial-error models:

```bash
python reproduce.py --section robustness
```

A concise description of the released input files and variables is provided in `data/reproduction/README.md`.

## Reproduction data

```text
data/reproduction/
├─ 2_2_multiscale_analysis/          # Travel-time and accessibility summary statistics
├─ 3_2_ci_analysis/                  # GDP-ranked concentration index results and plotting inputs
├─ 4_2_shapley_decomposition/        # Eight counterfactual scenario summaries
├─ see_cie_regression_panel/         # Analysis-ready city-level SEE/CIE panel
└─ static/administrative_boundaries/ # 2023 province/city/county boundaries
```

The reproduction workflow starts from these processed analytical inputs. Computationally intensive upstream stages are implemented in the repository but require the original source datasets and are separate from the released reproduction workflow.

## Repository structure

```text
.
├─ reproduce.py                      # Reproduction entry point
├─ environment.yml                   # Conda environment
├─ code/
│  ├─ config.py                      # Shared configuration
│  ├─ 0_*.py ... 4_*.py             # Upstream analysis stages
│  ├─ recompute_shapley_from_scenarios.py
│  ├─ build_see_cie_regression_panel.py
│  ├─ run_see_cie_regressions.py
│  ├─ run_province_fe_robustness.py
│  ├─ run_spatial_robustness.py
│  ├─ fig*.py                        # Main-text figure scripts
│  └─ utils/                         # Shared analysis utilities
├─ data/reproduction/                # Released reproduction inputs
├─ tests/                            # Unit tests
├─ osm_batch_router_v2/              # Rust routing source for the upstream travel-time workflow
└─ outputs/                          # Generated reproduction outputs
```

## Outputs

The main output groups are:

- `outputs/figures/` — regenerated main-text figures and supporting figure files;
- `outputs/regression/` — main SEE/CIE model outputs and province fixed-effects robustness results;
- `outputs/robustness/spatial_error/` — spatial-error model results and spatial diagnostics;
- `outputs/map_layers/` — GIS layers exported by map-generating scripts;
- `outputs/generated_files.txt` — index of files currently collected under `outputs/`.

## Upstream workflow

The repository also includes the analysis code used before the released processed inputs are produced. `code/reconstruct_from_raw_inputs.py` coordinates the larger upstream workflow, and `osm_batch_router_v2/` contains the Rust router used for road-network travel-time calculations. These stages require the corresponding source datasets and substantially greater computation than the released reproduction workflow.

Machine-dependent paths and compute settings can be configured through the `NC_*` environment variables defined in `code/config.py`. `NC_CPU_BUDGET` controls the default CPU budget for applicable upstream stages.

## Tests

Run the test suite from the repository root:

```bash
python -m pytest -q
```

The tests include population-weighted inequality metrics and synthetic verification of the exact Shapley implementation.

## License

The project code is released under the MIT License. Third-party data remain subject to their original licenses and terms of use.
