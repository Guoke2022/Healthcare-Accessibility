from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medical_accessibility.see_cie_analysis import plot_see_cie_effects  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot public SEE/CIE effect figures from exported regression outputs.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "see_cie",
        help="Directory containing exported SEE/CIE model results.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "see_cie" / "see_cie_effects.png",
        help="Path for the combined output figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_see_cie_effects(args.input_dir, args.output_path)


if __name__ == "__main__":
    main()

