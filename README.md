# Healthcare expansion improves accessibility but yields uneven equity returns across Chinese cities

Code and sample data for the manuscript:

**“Healthcare expansion improves accessibility but yields uneven equity returns across Chinese cities”**

## Overview

This repository contains the public code release for the main analytical workflows used in the manuscript. It provides reusable functions, executable scripts, and lightweight sample inputs to demonstrate the core workflow structure.

## Repository structure

```text
src/
  medical_accessibility/   # reusable analysis functions
scripts/                   # public workflow entry points
data/
  sample/                  # lightweight sample inputs
```

## Included workflows

This public repository includes code for the following analytical components:

1. Accessibility matching and grouped summary statistics  
2. Concentration index analysis  
3. SEE/CIE panel construction and regression analysis  
4. Shapley decomposition and waterfall plotting  
5. Accessibility dataset production

## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

Some workflows may require additional geospatial dependencies for full-scale reproduction beyond the bundled sample files.

## Quick start

### 1. Concentration index analysis

Input directory:

- `data/sample/ci_input/`

Bundled sample files:

- `national_2014_sample.csv`
- `national_2023_sample.csv`

Run:

```bash
python scripts/run_ci_analysis.py --input-dir data/sample/ci_input --output-dir results/ci
```

### 2. SEE/CIE panel construction and regression analysis

Run the workflow in three steps:

```bash
python scripts/build_see_cie_panel.py --yearly-city-dir <dir> --gdp-csv <file> --population-flow-csv <file> --population-density-xlsx <file> --provincial-stats-csv <file> --city-stats-csv <file> --county-stats-csv <file> --output-csv <panel.csv>

python scripts/run_see_cie_analysis.py --input-csv <panel.csv> --output-dir results/see_cie

python scripts/plot_see_cie_effects.py --input-dir results/see_cie --output-path results/see_cie/see_cie_effects.png
```

Sample inputs for this workflow are not provided as a full reproduction package. Users should prepare the required input tables from the original data sources.

### 3. Shapley decomposition

Run:

```bash
python scripts/run_shapley_decomposition.py --config-json data/sample/shapley/tasks.json --output-dir results/shapley

python scripts/plot_shapley_waterfalls.py --input-csv results/shapley/shapley_all_results.csv --output-dir results/shapley/figures
```

### 4. Accessibility dataset production on Linux

Because this workflow relies on extremely large raw datasets and production-scale geospatial inputs, no bundled sample inputs are provided in this repository.

Run:

```bash
python scripts/build_accessibility_dataset_linux.py --help
```

## Notes on sample data

The sample files provided in this repository are intended only to illustrate workflow usage.

- Some files are trimmed subsets of much larger analytical tables.
- CI sample files are reduced versions prepared for demonstration.
- Accessibility-summary helper functions are included, but no bundled sample input is provided for that workflow.
- The SEE/CIE analysis code is public, but exact numerical reproduction may depend on the original project environment and full input data.
- The Linux accessibility dataset production workflow is included as code only. Because it depends on extremely large raw datasets and production-scale geospatial inputs, bundled sample inputs are not provided in this repository.

## Citation

If you use this repository, please cite the associated manuscript.

## License

Please see the [LICENSE](LICENSE) file for reuse terms.
