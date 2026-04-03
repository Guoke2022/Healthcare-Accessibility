from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medical_accessibility.preprocessing import convert_bd09_columns_to_wgs84  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Convert BD-09 coordinates to WGS84.")
    parser.add_argument("--input-csv", type=Path, default=PROJECT_ROOT / "results" / "raw_preprocessing" / "results_baidu.csv")
    parser.add_argument("--output-csv", type=Path, default=PROJECT_ROOT / "results" / "raw_preprocessing" / "results_wgs84.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    convert_bd09_columns_to_wgs84(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()

