# Accessibility gains and uneven equity returns from healthcare expansion in China

Code and reproduction data for the manuscript **“Accessibility gains and uneven equity returns from healthcare expansion in China.”**

This repository provides a lightweight workflow for reproducing the manuscript figures and downstream SEE/CIE regression analyses from released intermediate data. The computationally intensive reconstruction from historical OSM networks, population rasters, and hospital-level inputs is retained separately for methodological transparency and is not required for the standard reproduction workflow.

## Reproduce the manuscript results

Create the software environment and activate it:

```bash
conda env create -f environment.yml
conda activate healthcare-accessibility
```

Check that all released inputs are present:

```bash
python reproduce.py --check-only
```

Run the complete public reproduction:

```bash
python reproduce.py
```

Optional subsets are available:

```bash
python reproduce.py --section figures
python reproduce.py --section see-cie
```

Generated files are written to `outputs/`. Temporary staged inputs and intermediate runtime files are written to `.reproduction_work/` and removed automatically after a successful run.

To remove all generated files:

```bash
python reproduce.py --clean-only
```

After each run, `outputs/generated_files.txt` lists the files created by the workflow. The generated outputs are ignored by Git and are not intended to be committed.

## Output structure

```text
outputs/
├─ figures/
│  ├─ Figure_1/
│  ├─ Figure_2/
│  ├─ Figure_3/
│  ├─ Figure_4/
│  │  ├─ CI_plots/
│  │  └─ KDE/
│  ├─ Figure_5/
│  └─ Shapley_decomposition/
├─ map_layers/
├─ regression/
├─ tables/
└─ generated_files.txt
```

Only manuscript-facing figures, compact descriptive tables, regression tables, and machine-readable regression results are copied to `outputs/`. Large staged inputs and internal intermediate files are kept out of the final output directory.

## Repository structure

```text
.
├─ reproduce.py                  # Public reproduction entry point
├─ environment.yml              # Reproduction environment
├─ code/                        # Analysis and figure scripts
│  ├─ config.py                 # Shared configuration module
│  ├─ reconstruct_from_raw_inputs.py
│  ├─ 0_*.py ... 5_*.py        # Numbered analysis/reconstruction stages
│  └─ fig*.py                   # Manuscript figure scripts
├─ data/
│  ├─ reproduction/             # Released inputs used by reproduce.py
│  ├─ example/                  # Illustrative raw-data examples
│  └─ external/                 # Optional large raw inputs; ignored by Git
├─ osm_batch_router_v2/         # Rust routing source
├─ docs/                        # Additional reproduction notes
└─ outputs/                     # Generated public outputs; ignored by Git
```

## Public reproduction boundary

The public workflow begins from compact analysis-ready stages rather than from national-scale raw geospatial data:

| Released input | Used for |
|---|---|
| `2_2_multiscale_analysis` | Figures 1–3 |
| `3_2_ci_analysis` and `plot_inputs` | Figure 4 |
| `4_2_shapley_decomposition` | Shapley decomposition figure |
| `5_3_see_cie_annual` | Figure 5 pathway and accessibility-change panels |
| `5_5_see_cie_regression_panel` | SEE/CIE regressions and Figure 5 effect panels |
| fixed 2023 administrative boundaries | Map layers used by the figure workflow |

The default workflow does **not** rerun national historical routing, travel-matrix construction, Ga2SFCA accessibility construction, GURS raster classification, grid-level CI matching, or hospital-level SEE/CIE construction.

The corresponding upstream code remains available in `code/` so that the computational definitions can be inspected. Advanced reconstruction from raw inputs can be initiated with:

```bash
python code/reconstruct_from_raw_inputs.py
```

This optional workflow requires separately obtained large geospatial datasets and substantially greater computing resources than the public reproduction workflow.

## Data availability

The released reproduction inputs are stored under `data/reproduction/`. Large upstream datasets are not included in the Git repository.

The complete annual hospital bed/campus database is not included in the initial public release because it supports ongoing research. The public workflow instead uses compact city-level SEE/CIE outputs and the analysis-ready regression panel required to reproduce the reported downstream analyses. Small illustrative hospital records may be provided under `data/example/` to document the raw input schema; such examples are not the data used to produce the national estimates.

See [`data/reproduction/README.md`](data/reproduction/README.md) and [`docs/reproduction.md`](docs/reproduction.md) for details.

## Figure and analysis scripts

The main plotting scripts are separated from the analysis stages that produce their inputs. In particular:

- `3_2_calculate_ci.py` computes CI results; `fig4_1_frequency_distribution_2014_2024.py` and `fig4_2_ci_plots.py` render Figure 4 outputs.
- `5_4_see_cie_descriptives.py` prepares the compact descriptive table used by `fig5_2_accessibility_scale_reversal_box.py`.
- `5_6_see_cie_regression.py` estimates the SEE/CIE models and writes compact effect tables used by `fig5_3_see_cie_effect_plots.py`.

## Citation

If you use this repository, please cite the associated manuscript:

> *Accessibility gains and uneven equity returns from healthcare expansion in China.*

A formal article citation and DOI can be added here upon publication.
