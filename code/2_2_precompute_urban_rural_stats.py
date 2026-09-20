# -*- coding: utf-8 -*-
"""Precompute the historical Urban/Rural statistics required by Figure 3.

This stage is deliberately separated from plotting. It uses the historical 10x10 GURS
majority definition, but performs raster IO city-by-city and caches each year so an
interrupted run can resume without repeating completed years.
"""
from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Precompute Figure-3 Urban/Rural grouped statistics.")
    ap.add_argument("--scope", default=None, help="Override NC_FIGURE_SERVICE_SCOPE.")
    ap.add_argument("--profile", default=None, help="Override NC_FIGURE_SPEED_PROFILE.")
    ap.add_argument("--force", action="store_true", help="Ignore valid yearly caches and rebuild all six years.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    # Set overrides before importing config-dependent compatibility helpers.
    if args.scope:
        os.environ["NC_FIGURE_SERVICE_SCOPE"] = args.scope
    if args.profile:
        os.environ["NC_FIGURE_SPEED_PROFILE"] = args.profile

    from utils.figure_data import precompute_urban_rural_stats

    print("=" * 80, flush=True)
    print("Figure 3 Urban/Rural 预统计", flush=True)
    print("口径：旧版 GURS 中心点周围 10x10 像元多数类别；Urban=1, Rural=2", flush=True)
    print("年份：2014, 2015, 2016 使用 GURS 2015；2019, 2020, 2021 使用 GURS 2020", flush=True)
    print("每年完成后都会单独缓存，可中断后继续。", flush=True)
    print("=" * 80, flush=True)
    paths = precompute_urban_rural_stats(force=args.force)
    print("\n完成。Figure 3 之后只读取缓存，不再现场计算城乡分类。", flush=True)
    for kind, path in paths.items():
        print(f"  {kind}: {path}", flush=True)


if __name__ == "__main__":
    main()
