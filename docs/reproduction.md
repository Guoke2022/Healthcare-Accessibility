# Reproduction notes

## Public workflow

`reproduce.py` is the only reviewer-facing entry point. It validates the released inputs, stages them in a temporary runtime directory, executes the downstream figure and SEE/CIE analysis scripts, and copies only manuscript-facing outputs to `outputs/`.

The runtime staging directory `.reproduction_work/` is removed automatically after a successful run unless `--keep-work` is supplied. Both the runtime directory and generated outputs are ignored by Git.

## Released inputs

The public reproduction starts from compact intermediate data:

- multiscale accessibility and travel-time statistics for Figures 1–3;
- compact CI tables, KDE curves, and concentration-curve points for Figure 4;
- compact Shapley endpoint/decomposition tables;
- annual city-level SEE/CIE tables for the Figure 5 pathway and accessibility-change panels;
- an analysis-ready city regression panel for the SEE/CIE models;
- fixed province, city, and county boundary layers.

These inputs replace computationally large upstream outputs that are not practical to distribute through GitHub.

## Optional raw reconstruction

`code/reconstruct_from_raw_inputs.py` retains the upstream reconstruction workflow. It is intended for authors or advanced users with the required raw inputs and computing resources. It is not needed to reproduce the released manuscript figures and downstream regression results.

The upstream workflow includes population/hospital preparation, OSM routing graph preparation, travel-time matrices, Ga2SFCA accessibility, multiscale matching/statistics, CI preparation, Shapley counterfactual scenarios, and hospital expansion construction.

## Hospital data boundary

The complete hospital-level annual bed/campus dataset is not part of the initial public release. The public workflow therefore begins from derived city-level SEE/CIE tables and the regression panel. Any records supplied under `data/example/` are illustrative schema examples only.
