# Hospital input example

`hospital_records_2024_schema.csv` documents the hospital-level input fields used by the upstream hospital-expansion workflow. It contains headers only and is not used by `reproduce.py`.

The complete longitudinal hospital bed database is not distributed with this release because it supports ongoing research. Authors can create a fixed-seed 100-record example from the private 2024 table with:

```bash
python tools/create_hospital_example.py --input /path/to/2024.csv
```

The generated file is intended only to illustrate the input schema and preprocessing interface; it cannot reproduce the national estimates in the manuscript.
