from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medical_accessibility.shapley_analysis import plot_shapley_waterfalls  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Plot the public Shapley waterfall figures.")
    parser.add_argument("--input-csv", type=Path, default=PROJECT_ROOT / "results" / "shapley" / "shapley_all_results.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "shapley" / "figures")
    return parser.parse_args()


def main():
    args = parse_args()
    plot_shapley_waterfalls(args.input_csv, args.output_dir)


if __name__ == "__main__":
    main()

