from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medical_accessibility.see_cie_panel import build_regression_panel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the cleaned city-level regression panel used in the SEE/CIE regression workflow.",
    )
    parser.add_argument("--yearly-city-dir", type=Path, required=True, help="Directory containing city_SEE_CIE_YYYY.csv files.")
    parser.add_argument("--gdp-csv", type=Path, required=True, help="City-level GDP panel CSV used for GDP_2014 and GDP growth controls.")
    parser.add_argument("--population-flow-csv", type=Path, required=True, help="City-level population-flow CSV used for migration and resident-population controls.")
    parser.add_argument("--population-density-xlsx", type=Path, required=True, help="Workbook containing yearly city-level population density values.")
    parser.add_argument("--provincial-stats-csv", type=Path, required=True, help="Provincial accessibility statistics CSV.")
    parser.add_argument("--city-stats-csv", type=Path, required=True, help="City accessibility statistics CSV.")
    parser.add_argument("--county-stats-csv", type=Path, required=True, help="County accessibility statistics CSV.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output path for the cleaned regression panel.")
    parser.add_argument("--start-year", type=int, default=2015, help="First yearly city_SEE_CIE file to include.")
    parser.add_argument("--end-year", type=int, default=2024, help="Last yearly city_SEE_CIE file to include.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_regression_panel(
        yearly_city_dir=args.yearly_city_dir,
        gdp_csv=args.gdp_csv,
        population_flow_csv=args.population_flow_csv,
        population_density_xlsx=args.population_density_xlsx,
        provincial_stats_csv=args.provincial_stats_csv,
        city_stats_csv=args.city_stats_csv,
        county_stats_csv=args.county_stats_csv,
        output_csv=args.output_csv,
        start_year=args.start_year,
        end_year=args.end_year,
    )


if __name__ == "__main__":
    main()
