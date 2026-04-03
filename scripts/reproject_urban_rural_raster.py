from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medical_accessibility.preprocessing import reproject_raster_to_epsg4326  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Reproject an urban-rural raster to EPSG:4326.")
    parser.add_argument("--src-path", type=Path, default=PROJECT_ROOT / "data" / "sample" / "raw_preprocessing" / "GURS_2015.tif")
    parser.add_argument("--dst-path", type=Path, default=PROJECT_ROOT / "results" / "raw_preprocessing" / "GURS_2015_EPSG4326.tif")
    return parser.parse_args()


def main():
    args = parse_args()
    reproject_raster_to_epsg4326(args.src_path, args.dst_path)


if __name__ == "__main__":
    main()
