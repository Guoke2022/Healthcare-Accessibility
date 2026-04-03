from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medical_accessibility.ci_analysis import (  # noqa: E402
    AnalysisSpec,
    compute_ci_series,
    make_gdp_formatter,
    plot_ci_trend,
    plot_concentration_curves,
    plot_weighted_kde,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the public GDP-based concentration-index workflow on yearly national CSV files.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "sample" / "ci_input",
        help="Directory containing national_YYYY.csv or national_YYYY_sample.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "ci",
        help="Directory for CSV outputs and figures.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2014,
        help="First analysis year.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2023,
        help="Last analysis year.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    years = list(range(args.start_year, args.end_year + 1))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gdp_spec = AnalysisSpec(
        name="gdp",
        rank_var="GDP_per",
        valid_rank_rule="gt0",
        xlabel="Cumulative share of population by SES",
        xlim=(0.18, 0.30),
    )
    ci_df = compute_ci_series(args.input_dir, years, gdp_spec)
    ci_csv = args.output_dir / "acc_ci_by_gdp.csv"
    ci_df.to_csv(ci_csv, index=False, encoding="utf-8-sig")

    plot_concentration_curves(
        input_dir=args.input_dir,
        ci_df=ci_df,
        output_path=args.output_dir / "acc_concentration_curve_gdp.png",
        spec=gdp_spec,
        years=[years[0], years[-1]],
    )
    plot_ci_trend(
        ci_df=ci_df,
        output_path=args.output_dir / "acc_ci_trend_gdp.png",
        xlim=gdp_spec.xlim,
    )

    plot_weighted_kde(
        input_dir=args.input_dir,
        output_path=args.output_dir / "gdp_kde.png",
        x_var="GDP_per",
        x_label="GDP per capita (10k RMB)",
        years=years,
        filter_fn=lambda df: df[(df["GDP_per"] > 0) & (df["pop"] > 0)].copy(),
        bw_adjust=5.0,
        formatter=make_gdp_formatter(),
    )

if __name__ == "__main__":
    main()
