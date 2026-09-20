# Generated outputs

This directory is populated by `python reproduce.py`.

- `figures/` contains manuscript figure outputs.
- `regression/` contains the main SEE/CIE model tables and machine-readable estimates.
- `map_layers/` contains GIS layers exported by map-generating scripts.
- `generated_files.txt` lists every generated file from the most recent run.

Generated files are ignored by Git. Run `python reproduce.py --clean-only` to remove them.
