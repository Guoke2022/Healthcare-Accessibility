# Generated outputs

`python reproduce.py` writes generated results to this directory.

- `figures/` — regenerated main-text figures and supporting figure files.
- `regression/` — main SEE/CIE model outputs and province fixed-effects robustness results.
- `robustness/spatial_error/` — spatial-error model results and spatial diagnostics.
- `map_layers/` — GIS layers exported by map-generating scripts.
- `generated_files.txt` — index of files currently collected under `outputs/`.

Use `python reproduce.py --clean-only` to remove generated outputs and temporary working files.
