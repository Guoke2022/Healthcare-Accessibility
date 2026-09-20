# -*- coding: utf-8 -*-
"""5_8 Global Moran's I diagnostics for Stage-3 SEE/CIE OLS models.

Purpose
-------
This is a deliberately minimal spatial-diagnostics script. It does NOT replace OLS
with a spatial model. It asks one question first; 5_9 then re-estimates the same
Stage-3 specifications with a spatial error model as a robustness analysis:

    Do the fully adjusted Stage-3 OLS residuals retain global spatial autocorrelation?

For each of the seven main Stage-3 specifications used in Figure 5 / Tables S6-S12,
the script reports Global Moran's I for:
  1) the outcome itself; and
  2) the OLS residuals.

Primary spatial weights: 4-nearest-neighbour (KNN4), row-standardized.
The city geometry is built from the same county shapefile used elsewhere in the project.

Outputs
-------
result/5_8_spatial_diagnostics/
  - moran_stage3_knn4.csv
  - model_sample_and_geometry_match.csv
  - unmatched_regression_cities.csv

Dependencies
------------
pip install geopandas libpysal esda statsmodels
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    BASE_YEAR,
    END_YEAR,
    SEE_CIE_REGRESSION_ROOT,
    COUNTY_SHP,
    RESULT_ROOT,
    CITY_ORDER_4,
)
from utils.extended_analysis import read_csv_robust

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
K_NEIGHBORS = 4
N_PERMUTATIONS = 9999
RANDOM_SEED = 20260828

OUT_ROOT = RESULT_ROOT / "5_8_spatial_diagnostics"

ACC_BASE_COL = f"acc_{BASE_YEAR}"
GINI_BASE_COL = f"gini_{BASE_YEAR}"
THEIL_BASE_COL = f"theil_{BASE_YEAR}"
ATKINSON_BASE_COL = f"atkinson_05_{BASE_YEAR}"
GDP_BASE_COL = f"GDP_{BASE_YEAR}"
RESPOP_BASE_COL = f"ResPop_{BASE_YEAR}"
FISCAL_BASE_COL = f"ln_FiscalRevenue_pc_{BASE_YEAR}"


def _norm_text(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


def build_city_geometry():

    try:
        import geopandas as gpd
    except ImportError as e:
        raise ImportError("Please install geopandas: pip install geopandas") from e

    if not Path(COUNTY_SHP).exists():
        raise FileNotFoundError(f"COUNTY_SHP not found: {COUNTY_SHP}")

    gdf = gpd.read_file(COUNTY_SHP)
    required = {"地级", "省级", "县级", "geometry"}
    missing = required - set(gdf.columns)
    if missing:
        raise ValueError(f"COUNTY_SHP is missing required columns: {sorted(missing)}")

    gdf = gdf[["地级", "省级", "县级", "geometry"]].copy()
    gdf["city_name"] = _norm_text(gdf["地级"])
    gdf["province_name"] = _norm_text(gdf["省级"])
    gdf["county_name"] = _norm_text(gdf["县级"])

    use_province = gdf["city_name"].eq("不统计") | gdf["city_name"].isna()
    gdf.loc[use_province, "city_name"] = gdf.loc[use_province, "province_name"]

    direct_admin = gdf["city_name"].isin(["海南省", "湖北省"])
    gdf.loc[direct_admin, "city_name"] = gdf.loc[direct_admin, "county_name"]

    # Repair invalid polygons if needed; buffer(0) is only a geometry-cleaning step.
    bad = ~gdf.geometry.is_valid
    if bad.any():
        warnings.warn(f"Repairing {int(bad.sum())} invalid county geometries with buffer(0).")
        gdf.loc[bad, "geometry"] = gdf.loc[bad, "geometry"].buffer(0)

    city_geo = gdf[["city_name", "geometry"]].dissolve(by="city_name", as_index=False)
    city_geo = city_geo[~city_geo.geometry.is_empty & city_geo.geometry.notna()].copy()
    return city_geo


def prepare_regression_data() -> pd.DataFrame:
    """Load the exact standardized dataset written by 5_6."""
    path = SEE_CIE_REGRESSION_ROOT / "regression_standardized_data.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Regression input not found: {path}\n"
            "Please run 5_6_see_cie_regression.py first."
        )

    df = read_csv_robust(path)
    needed = {"地级", "city_level"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Regression dataset missing columns: {sorted(missing)}")

    df = df[df["city_level"].isin(CITY_ORDER_4)].copy()
    df["地级"] = _norm_text(df["地级"])
    df["city_level_4"] = pd.Categorical(
        df["city_level"], categories=CITY_ORDER_4, ordered=True
    )
    return df


def stage3_specs():
    """Return the seven fully adjusted Stage-3 formulas used in 5_6."""
    city_fe = 'C(city_level_4, Treatment(reference="Medium/Small City"))'
    controls = (
        f"pop_density_mean + {GDP_BASE_COL} + GDP_growth_pct + {RESPOP_BASE_COL} "
        "+ Ppo_NetIn_rate + ResPop_growth_rate + " + FISCAL_BASE_COL
    )

    specs = []

    # Accessibility: SEE + CIE jointly
    specs.append(
        {
            "model": "accessibility_SEE_CIE_stage3",
            "outcome": "acc_delta",
            "family": "SEE_CIE",
            "formula": (
                f"acc_delta ~ {ACC_BASE_COL} + {controls} "
                f"+ city_SEE*{city_fe} + city_CIE*{city_fe}"
            ),
            "columns": [
                "acc_delta", ACC_BASE_COL, "pop_density_mean", GDP_BASE_COL,
                "GDP_growth_pct", RESPOP_BASE_COL, "Ppo_NetIn_rate",
                "ResPop_growth_rate", FISCAL_BASE_COL, "city_SEE", "city_CIE", "city_level_4",
            ],
        }
    )

    inequality = {
        "gini": ("gini_delta", GINI_BASE_COL),
        "theil": ("theil_delta", THEIL_BASE_COL),
        "atkinson_05": ("atkinson_05_delta", ATKINSON_BASE_COL),
    }

    for label, (y, y0) in inequality.items():
        common_cols = [
            y, y0, "pop_density_mean", GDP_BASE_COL, "GDP_growth_pct",
            RESPOP_BASE_COL, "Ppo_NetIn_rate", "ResPop_growth_rate", FISCAL_BASE_COL,
            "city_level_4",
        ]
        specs.append(
            {
                "model": f"{label}_SEE_CIE_stage3",
                "outcome": y,
                "family": "SEE_CIE",
                "formula": (
                    f"{y} ~ {y0} + {controls} "
                    f"+ city_SEE*{city_fe} + city_CIE*{city_fe}"
                ),
                "columns": common_cols + ["city_SEE", "city_CIE"],
            }
        )
        specs.append(
            {
                "model": f"{label}_TotalExpansion_stage3",
                "outcome": y,
                "family": "TotalExpansion",
                "formula": (
                    f"{y} ~ {y0} + {controls} "
                    f"+ city_TotalExpansion*{city_fe}"
                ),
                "columns": common_cols + ["city_TotalExpansion"],
            }
        )

    return specs


def build_knn_weights(city_geo, city_order, k=4):
    """Build row-standardized KNN weights for one model sample.

    We project the nationwide city polygons to a China-centered Albers equal-area CRS
    before deriving city centroids, so KNN is not computed directly in longitude/latitude.
    """
    try:
        from libpysal.weights import KNN
    except ImportError as e:
        raise ImportError("Please install libpysal: pip install libpysal") from e

    # Keep exactly the regression-model order.
    q = city_geo.set_index("city_name").loc[list(city_order)].copy()

    if q.crs is None:
        raise ValueError("COUNTY_SHP has no CRS; cannot construct defensible KNN distances.")

    # China-centered Albers equal-area projection, meters.
    china_aea = (
        "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 "
        "+datum=WGS84 +units=m +no_defs"
    )
    q_proj = q.to_crs(china_aea)
    cent = q_proj.geometry.centroid
    coords = np.column_stack([cent.x.to_numpy(), cent.y.to_numpy()])

    if len(coords) <= k:
        raise ValueError(f"Model sample N={len(coords)} must be greater than k={k}.")

    w = KNN.from_array(coords, k=k, ids=list(city_order))
    w.transform = "R"
    return w


def moran_stats(values, w):
    try:
        from esda.moran import Moran
    except ImportError as e:
        raise ImportError("Please install esda: pip install esda") from e

    arr = np.asarray(values, dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("Moran input contains NaN or infinite values after model filtering.")
    if np.nanstd(arr) == 0:
        raise ValueError("Moran input has zero variance.")

    # esda uses permutation inference; seed NumPy for reproducibility.
    np.random.seed(RANDOM_SEED)
    mi = Moran(arr, w, permutations=N_PERMUTATIONS, two_tailed=True)

    return {
        "moran_I": float(mi.I),
        "expected_I": float(mi.EI),
        "p_perm": float(mi.p_sim),
        "z_sim": float(mi.z_sim),
        "p_z_sim": float(mi.p_z_sim),
        "permutations": int(N_PERMUTATIONS),
    }


def main():
    try:
        import statsmodels.formula.api as smf
    except ImportError as e:
        raise ImportError("Please install statsmodels: pip install statsmodels") from e

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    df = prepare_regression_data()
    city_geo = build_city_geometry()
    geo_names = set(city_geo["city_name"].astype(str))

    # Global diagnostic: cities in the regression file that cannot be mapped at all.
    all_reg_cities = sorted(set(df["地级"].dropna().astype(str)))
    unmatched_all = sorted(set(all_reg_cities) - geo_names)
    pd.DataFrame({"地级": unmatched_all}).to_csv(
        OUT_ROOT / "unmatched_regression_cities.csv", index=False, encoding="utf-8-sig"
    )

    rows = []
    match_rows = []

    for spec in stage3_specs():
        required = list(dict.fromkeys(spec["columns"] + ["地级"]))
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            raise ValueError(f"{spec['model']} missing columns: {missing_cols}")

        d = df[required].dropna().copy()
        d["地级"] = _norm_text(d["地级"])

        # Ensure one observation per city.
        dup = d["地级"].duplicated(keep=False)
        if dup.any():
            names = sorted(d.loc[dup, "地级"].astype(str).unique())
            raise ValueError(f"{spec['model']} has duplicate city observations: {names[:20]}")

        matched = d["地级"].isin(geo_names)
        for city, ok in zip(d["地级"], matched):
            match_rows.append({"model": spec["model"], "地级": city, "geometry_matched": bool(ok)})

        if not matched.all():
            bad = d.loc[~matched, "地级"].astype(str).tolist()
            raise ValueError(
                f"{spec['model']} has {len(bad)} cities unmatched to geometry: {bad}\n"
                f"See {OUT_ROOT / 'unmatched_regression_cities.csv'}"
            )

        # Refit the exact Stage-3 mean specification. Robust covariance is irrelevant
        # for fitted values/residuals; the residuals are identical to HC1/HC3 fits.
        model = smf.ols(spec["formula"], data=d).fit()

        # patsy/statsmodels can in principle drop rows; enforce exact row alignment.
        used_idx = model.model.data.row_labels
        d_used = d.loc[used_idx].copy()
        if len(d_used) != int(model.nobs):
            raise RuntimeError(f"Row alignment failed for {spec['model']}.")

        city_order = d_used["地级"].astype(str).tolist()
        w = build_knn_weights(city_geo, city_order, k=K_NEIGHBORS)

        # Outcome Moran's I: useful context; residual Moran's I is the key diagnostic.
        outcome_res = moran_stats(d_used[spec["outcome"]].to_numpy(), w)
        rows.append({
            "model": spec["model"],
            "family": spec["family"],
            "outcome": spec["outcome"],
            "statistic": "outcome",
            "n": int(model.nobs),
            "weights": f"KNN{K_NEIGHBORS}",
            "k": K_NEIGHBORS,
            **outcome_res,
        })

        resid_res = moran_stats(model.resid.to_numpy(), w)
        rows.append({
            "model": spec["model"],
            "family": spec["family"],
            "outcome": spec["outcome"],
            "statistic": "OLS_residual",
            "n": int(model.nobs),
            "weights": f"KNN{K_NEIGHBORS}",
            "k": K_NEIGHBORS,
            **resid_res,
        })

        print(
            f"{spec['model']}: N={int(model.nobs)} | "
            f"outcome I={outcome_res['moran_I']:.4f}, p={outcome_res['p_perm']:.4g} | "
            f"residual I={resid_res['moran_I']:.4f}, p={resid_res['p_perm']:.4g}"
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_ROOT / "moran_stage3_knn4.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(match_rows).drop_duplicates().to_csv(
        OUT_ROOT / "model_sample_and_geometry_match.csv", index=False, encoding="utf-8-sig"
    )

    print("\nDone.")
    print(f"Main results: {OUT_ROOT / 'moran_stage3_knn4.csv'}")
    print("Interpretation focus: rows where statistic == 'OLS_residual'.")
    print("A small permutation p-value indicates residual spatial autocorrelation remains.")
    print("If residual spatial autocorrelation is detected, inspect 5_9 spatial-error robustness outputs.")


if __name__ == "__main__":
    main()
