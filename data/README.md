# Data layout

This repository separates **paper-reproduction inputs** from **illustrative raw-data examples** and from **external large raw datasets**.

## `reproduction/`

Publication-ready intermediate/analysis-ready data required by the default reproduction workflow belong here. These files are intended to support rapid regeneration of the reported statistics, regressions, and figures without rerunning national-scale routing from raw OSM and population rasters.

The release manifest is defined in `reproduction/manifest.json`; the public bundle uses a flat directory layout and excludes large upstream grids, routing outputs, and pre-rendered figures.

## `example/`

Small illustrative raw-data subsets belong here. For example, a deterministic sample of hospital-campus records may be supplied to document the raw schema and demonstrate preprocessing behavior.

**Example data are not the data used to reproduce the paper's reported estimates.** They must never be presented as a substitute for the analysis-ready reproduction data.

## `external/`

Large or separately distributed raw inputs belong here when a user performs optional raw-data reconstruction. These files are intentionally excluded from Git version control. Examples include historical OSM PBF files, population rasters, and GURS rasters.
