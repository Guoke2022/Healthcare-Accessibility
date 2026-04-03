from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medical_accessibility.accessibility_stats import process_yearly_files  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description='Compute public accessibility summary statistics from yearly matched CSV files.')
    parser.add_argument('--input-dir', type=Path, default=PROJECT_ROOT / 'data' / 'sample' / 'matched_accessibility')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'results' / 'accessibility_stats')
    parser.add_argument('--years', nargs='+', default=[str(year) for year in range(2014, 2025)])
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    process_yearly_files(args.input_dir, args.years, args.output_dir / 'national_acc_stats.csv')
    process_yearly_files(args.input_dir, args.years, args.output_dir / 'provincial_acc_stats.csv', groupby_col='省级')
    process_yearly_files(args.input_dir, args.years, args.output_dir / 'city_acc_stats.csv', groupby_col='地级')
    process_yearly_files(args.input_dir, args.years, args.output_dir / 'county_acc_stats.csv', groupby_col='县级码')


if __name__ == '__main__':
    main()
