from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medical_accessibility.preprocessing import geocode_addresses  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Geocode hospital addresses with the Baidu API.")
    parser.add_argument("--input-csv", type=Path, default=PROJECT_ROOT / "results" / "raw_preprocessing" / "results_cleaned.csv")
    parser.add_argument("--output-csv", type=Path, default=PROJECT_ROOT / "results" / "raw_preprocessing" / "results_baidu.csv")
    parser.add_argument("--ak", default=os.environ.get("BAIDU_MAP_AK", ""))
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.ak:
        raise ValueError("A Baidu API key is required. Pass --ak or set BAIDU_MAP_AK.")
    geocode_addresses(args.input_csv, args.output_csv, args.ak)


if __name__ == "__main__":
    main()

