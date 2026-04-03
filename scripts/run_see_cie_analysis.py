from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medical_accessibility.see_cie_analysis import load_city_panel, run_see_cie_models  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the public SEE/CIE regression workflow on the cleaned city-level panel.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "sample" / "see_cie" / "city_panel.csv",
        help="City-level panel CSV used for SEE/CIE regressions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "see_cie",
        help="Directory for model outputs, tables, and intermediate files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_city_panel(args.input_csv)
    run_see_cie_models(df, args.output_dir)


if __name__ == "__main__":
    main()

