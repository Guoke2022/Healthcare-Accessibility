# CI Sample Note

The files in this folder are lightweight public samples for the GDP-based concentration-index workflow.

Included files:

- `national_2014_sample.csv`
- `national_2023_sample.csv`

These files were reduced from larger CI-ready tables using the following steps:

1. Keep only the columns required by the public GDP-based CI workflow:
   - `pop`
   - `acc`
   - `GDP_per`
2. Drop rows that would not enter the public GDP-based CI calculation:
   - `pop <= 0`
   - `GDP_per <= 0`
   - missing values in `pop`, `acc`, or `GDP_per`
3. Apply deterministic row sampling with fixed random seeds so that each file stays below 100 MB for GitHub hosting.

Final sample sizes:

- `national_2014_sample.csv`: 4,866,826 rows, 98,801,425 bytes
- `national_2023_sample.csv`: 4,516,354 rows, 99,301,433 bytes

These files are intended for code demonstration and smoke testing, not for exact reproduction of the full-paper CI estimates.
