# -*- coding: utf-8 -*-
"""5_9 Province fixed-effects robustness for SEE/CIE regressions.

Purpose
-------
The formal 5_6 model already includes baseline fiscal capacity
ln(FiscalRevenue_pc_2014) as a control.  This script asks a separate robustness
question: do the *overall* pathway associations persist after absorbing common
province-level institutional/policy environments?

Accordingly, this script deliberately does NOT re-estimate SEE/CIE x city-size
interactions with province fixed effects.  Four of the seven mega cities are
province-level municipalities and provide no within-province city variation,
which makes mega-city interactions weakly identified under province FE.

Specifications (same complete-case sample within each outcome/model family):
  Main_HC1:
      baseline outcome + controls + city-size FE + expansion variable(s)
  ProvinceFE_HC1:
      Main_HC1 + province FE
  ProvinceFE_cluster:
      ProvinceFE model with province-clustered standard errors

Coefficients remain adjusted associations, not causal effects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import BASE_YEAR, END_YEAR, SEE_CIE_REGRESSION_ROOT, CITY_ORDER_4
from utils.extended_analysis import read_csv_robust, p_to_star

ACC_BASE_COL = f"acc_{BASE_YEAR}"
GINI_BASE_COL = f"gini_{BASE_YEAR}"
THEIL_BASE_COL = f"theil_{BASE_YEAR}"
ATKINSON_BASE_COL = f"atkinson_05_{BASE_YEAR}"
GDP_BASE_COL = f"GDP_{BASE_YEAR}"
RESPOP_BASE_COL = f"ResPop_{BASE_YEAR}"
FISCAL_BASE_COL = f"ln_FiscalRevenue_pc_{BASE_YEAR}"

OUT_ROOT = SEE_CIE_REGRESSION_ROOT / "province_fe_robustness"


def _fit(smf, formula: str, data: pd.DataFrame, covariance: str):
    if covariance == "HC1":
        return smf.ols(formula, data=data).fit(cov_type="HC1")
    if covariance == "province_cluster":
        return smf.ols(formula, data=data).fit(
            cov_type="cluster",
            cov_kwds={
                "groups": data["province_fe"],
                "use_correction": True,
                "df_correction": True,
            },
            use_t=True,
        )
    raise ValueError(covariance)


def _collect_effect(model, *, outcome, model_family, expansion, specification, covariance):
    term = {"SEE": "city_SEE", "CIE": "city_CIE", "TotalExpansion": "city_TotalExpansion"}[expansion]
    coef = float(model.params.get(term, np.nan))
    se = float(model.bse.get(term, np.nan))
    p = float(model.pvalues.get(term, np.nan))
    return {
        "outcome": outcome,
        "model_family": model_family,
        "expansion": expansion,
        "specification": specification,
        "covariance": covariance,
        "effect": coef,
        "se": se,
        "ci_low": coef - 1.96 * se if np.isfinite(coef) and np.isfinite(se) else np.nan,
        "ci_high": coef + 1.96 * se if np.isfinite(coef) and np.isfinite(se) else np.nan,
        "p": p,
        "significance": p_to_star(p),
        "n": int(model.nobs),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
    }


def _collect_coefficients(model, *, outcome, model_family, specification, covariance):
    rows = []
    for term, coef in model.params.items():
        rows.append({
            "outcome": outcome,
            "model_family": model_family,
            "specification": specification,
            "covariance": covariance,
            "term": term,
            "coef": float(coef),
            "se": float(model.bse.get(term, np.nan)),
            "p": float(model.pvalues.get(term, np.nan)),
            "n": int(model.nobs),
            "r2": float(model.rsquared),
            "adj_r2": float(model.rsquared_adj),
        })
    return rows


def _display(effect: float, p: float) -> str:
    if not np.isfinite(effect):
        return ""
    return f"{effect:.4f}{p_to_star(p)}"


def main():
    try:
        import statsmodels.formula.api as smf
    except ImportError as e:
        raise ImportError("5_6_3 requires statsmodels") from e

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = SEE_CIE_REGRESSION_ROOT / "regression_standardized_data.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run 5_6_see_cie_regression.py first: {path}")

    df = read_csv_robust(path)
    required = {"省级", "city_level", FISCAL_BASE_COL, "city_SEE", "city_CIE", "city_TotalExpansion"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Regression data missing required columns: {sorted(missing)}")

    df = df[df["city_level"].isin(CITY_ORDER_4)].copy()
    df["city_level_4"] = pd.Categorical(df["city_level"], categories=CITY_ORDER_4, ordered=True)
    df["province_fe"] = df["省级"].astype("string").str.strip().astype("category")

    # These variables have already been standardized separately in 5_6.
    # In particular, city_TotalExpansion is z(raw SEE + raw CIE),
    # so it must NOT be reconstructed here as z(SEE) + z(CIE).
    df["city_SEE"] = pd.to_numeric(df["city_SEE"], errors="coerce")
    df["city_CIE"] = pd.to_numeric(df["city_CIE"], errors="coerce")
    df["city_TotalExpansion"] = pd.to_numeric(
        df["city_TotalExpansion"], errors="coerce"
    )

    city_fe = 'C(city_level_4, Treatment(reference="Medium/Small City"))'
    province_fe = "C(province_fe)"
    controls = (
        f"pop_density_mean + {GDP_BASE_COL} + GDP_growth_pct + {RESPOP_BASE_COL} + "
        f"Ppo_NetIn_rate + ResPop_growth_rate + {FISCAL_BASE_COL}"
    )

    effect_rows = []
    coef_rows = []
    sample_rows = []

    families = [
        {
            "outcome": "accessibility",
            "model_family": "SEE_CIE",
            "y": "acc_delta",
            "y0": ACC_BASE_COL,
            "expansions": ["SEE", "CIE"],
            "terms": "city_SEE + city_CIE",
        },
    ]
    for label, y, y0 in [
        ("gini", "gini_delta", GINI_BASE_COL),
        ("theil", "theil_delta", THEIL_BASE_COL),
        ("atkinson_05", "atkinson_05_delta", ATKINSON_BASE_COL),
    ]:
        families.append({
            "outcome": label,
            "model_family": "SEE_CIE",
            "y": y,
            "y0": y0,
            "expansions": ["SEE", "CIE"],
            "terms": "city_SEE + city_CIE",
        })
        families.append({
            "outcome": label,
            "model_family": "TotalExpansion",
            "y": y,
            "y0": y0,
            "expansions": ["TotalExpansion"],
            "terms": "city_TotalExpansion",
        })

    base_cols = [
        "省级", "province_fe", "city_level_4", "pop_density_mean", GDP_BASE_COL,
        "GDP_growth_pct", RESPOP_BASE_COL, "Ppo_NetIn_rate", "ResPop_growth_rate",
        FISCAL_BASE_COL,
    ]

    for spec in families:
        needed = base_cols + [spec["y"], spec["y0"]]
        if spec["model_family"] == "SEE_CIE":
            needed += ["city_SEE", "city_CIE"]
        else:
            needed += ["city_TotalExpansion"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise KeyError(f"{spec['outcome']} {spec['model_family']} missing columns: {missing}")
        d = df[needed].dropna().copy()
        if len(d) < 20:
            print(f"Warning: skip {spec['outcome']} {spec['model_family']}, N={len(d)}")
            continue

        counts = d.groupby("省级", observed=False).size().sort_values()
        singleton = counts[counts == 1].index.tolist()
        sample_rows.append({
            "outcome": spec["outcome"],
            "model_family": spec["model_family"],
            "N": len(d),
            "n_provinces": int(d["省级"].nunique()),
            "n_singleton_provinces": len(singleton),
            "singleton_provinces": "; ".join(singleton),
            **{f"N_{level}": int((d["city_level_4"] == level).sum()) for level in CITY_ORDER_4},
        })

        main_formula = (
            f"{spec['y']} ~ {spec['y0']} + {controls} + {spec['terms']} + {city_fe}"
        )
        province_formula = main_formula + f" + {province_fe}"

        models = [
            ("Main", "HC1", _fit(smf, main_formula, d, "HC1")),
            ("ProvinceFE", "HC1", _fit(smf, province_formula, d, "HC1")),
            ("ProvinceFE", "province_cluster", _fit(smf, province_formula, d, "province_cluster")),
        ]

        for specification, covariance, model in models:
            coef_rows.extend(_collect_coefficients(
                model, outcome=spec["outcome"], model_family=spec["model_family"],
                specification=specification, covariance=covariance,
            ))
            for expansion in spec["expansions"]:
                effect_rows.append(_collect_effect(
                    model, outcome=spec["outcome"], model_family=spec["model_family"],
                    expansion=expansion, specification=specification, covariance=covariance,
                ))

    effects = pd.DataFrame(effect_rows)
    coefs = pd.DataFrame(coef_rows)
    samples = pd.DataFrame(sample_rows)

    effects.to_csv(OUT_ROOT / "province_fe_overall_effects_long.csv", index=False, encoding="utf-8-sig")
    coefs.to_csv(OUT_ROOT / "province_fe_model_coefficients_long.csv", index=False, encoding="utf-8-sig")
    samples.to_csv(OUT_ROOT / "province_fe_sample_accounting.csv", index=False, encoding="utf-8-sig")

    # Wide reviewer/SI-friendly comparison table.
    wide = effects.copy()
    wide["display"] = [_display(e, p) for e, p in zip(wide["effect"], wide["p"])]
    wide["column"] = np.select(
        [
            (wide["specification"] == "Main") & (wide["covariance"] == "HC1"),
            (wide["specification"] == "ProvinceFE") & (wide["covariance"] == "HC1"),
            (wide["specification"] == "ProvinceFE") & (wide["covariance"] == "province_cluster"),
        ],
        ["Main_HC1", "ProvinceFE_HC1", "ProvinceFE_cluster"],
        default="other",
    )
    table = wide.pivot_table(
        index=["outcome", "model_family", "expansion"],
        columns="column", values="display", aggfunc="first",
    ).reset_index()
    table.to_csv(OUT_ROOT / "Table_province_FE_overall_robustness.csv", index=False, encoding="utf-8-sig")

    # R2 comparison (cluster covariance does not alter fitted values/R2, so HC1 rows suffice).
    r2 = coefs.drop_duplicates(["outcome", "model_family", "specification", "covariance"])
    r2 = r2[r2["covariance"].eq("HC1")][["outcome", "model_family", "specification", "n", "r2", "adj_r2"]]
    r2.to_csv(OUT_ROOT / "province_FE_R2_comparison.csv", index=False, encoding="utf-8-sig")

    readme = f"""Province fixed-effects robustness (overall pathway associations only)\n\nMain model already controls for:\n  population density; {GDP_BASE_COL}; GDP growth; {RESPOP_BASE_COL};\n  net population inflow; resident population growth; and {FISCAL_BASE_COL}.\n\nProvince-FE models add C(province_fe) and are also reported with province-clustered SE.\nNo SEE/CIE x city-size interactions are estimated here because four of seven mega cities\nare province-level municipalities and provide no within-province city variation for\nidentifying mega-city interaction effects.\n\nInterpretation: province FE absorb common province-level institutional/policy environments,\nbut do not eliminate city-level endogeneity. Coefficients remain adjusted associations.\n"""
    (OUT_ROOT / "README.txt").write_text(readme, encoding="utf-8")

    print("=" * 88)
    print("5_6_3 Province FE overall robustness complete")
    print(f"Output: {OUT_ROOT}")
    print(table.to_string(index=False))
    print("=" * 88)


if __name__ == "__main__":
    main()
