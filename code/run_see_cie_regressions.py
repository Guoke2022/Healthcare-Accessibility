"""Estimate the adjusted SEE/CIE regression models reported in the manuscript."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from config import BASE_YEAR, SEE_CIE_REGRESSION_ROOT, CITY_ORDER_4
from utils.extended_analysis import p_to_star, summary_col_standard_stars
from utils.regression import (
    load_standardized_see_cie_panel,
    ACC_BASE_COL,
    GINI_BASE_COL,
    THEIL_BASE_COL,
    ATKINSON_BASE_COL,
    GDP_BASE_COL,
    RESPOP_BASE_COL,
    FISCAL_BASE_COL,
)


def _find_interaction_name(params_index, var_prefix, city, fe_col="city_level_4"):
    hits = [
        p for p in params_index
        if var_prefix in p and f"[T.{city}]" in p and f"C({fe_col}" in p and ":" in p
    ]
    return hits[0] if hits else None


def _total_effect_stats(model, var_prefix, city, ref="Medium/Small City", fe_col="city_level_4"):
    params = model.params
    if var_prefix not in params.index:
        return None, None, None
    names = list(params.index)
    r = np.zeros((1, len(names)))
    r[0, names.index(var_prefix)] = 1
    if city != ref:
        inter = _find_interaction_name(params.index, var_prefix, city, fe_col)
        if inter is not None:
            r[0, names.index(inter)] = 1
    effect = float((r @ params.to_numpy(dtype=float)).item())
    cov = model.cov_params().to_numpy(dtype=float)
    se = float(np.sqrt((r @ cov @ r.T).item()))
    p = float(model.wald_test(r, scalar=True).pvalue)
    return effect, se, p


def _collect_model_coefficients(model, *, outcome, model_family, stage):
    rows = []
    for term, coef in model.params.items():
        rows.append({
            "outcome": outcome,
            "model_family": model_family,
            "stage": stage,
            "term": term,
            "coef": float(coef),
            "se": float(model.bse.get(term, np.nan)),
            "p": float(model.pvalues.get(term, np.nan)),
            "n": int(model.nobs),
            "r2": float(model.rsquared),
        })
    return rows


def _collect_city_effects(model, *, outcome, expansion, var_prefix):
    rows = []
    tcrit = float(stats.t.ppf(0.975, df=float(model.df_resid)))
    for city in CITY_ORDER_4:
        effect, se, p = _total_effect_stats(model, var_prefix, city)
        if effect is None:
            continue
        rows.append({
            "outcome": outcome,
            "expansion": expansion,
            "city_level": city,
            "effect": effect,
            "se": se,
            "p": p,
            "ci_low": effect - tcrit * se,
            "ci_high": effect + tcrit * se,
            "df_resid": float(model.df_resid),
            "significance": p_to_star(p),
            "n": int(model.nobs),
        })
    return rows


def _write_model_table(models, names, path):
    table = summary_col_standard_stars(
        models,
        stars=True,
        model_names=names,
        info_dict={"N": lambda x: f"{int(x.nobs)}", "R2": lambda x: f"{x.rsquared:.3f}"},
        float_format="%0.4f",
    )
    path.write_text(table.as_text(), encoding="utf-8")


def main():
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise ImportError("statsmodels is required for the SEE/CIE regressions") from exc

    SEE_CIE_REGRESSION_ROOT.mkdir(parents=True, exist_ok=True)
    df = load_standardized_see_cie_panel()
    city_fe = 'C(city_level_4, Treatment(reference="Medium/Small City"))'
    controls = (
        f"pop_density_mean + {GDP_BASE_COL} + GDP_growth_pct + {RESPOP_BASE_COL} + "
        f"Ppo_NetIn_rate + ResPop_growth_rate + {FISCAL_BASE_COL}"
    )

    coef_rows = []
    effect_rows = []

    acc_cols = [
        "acc_delta", ACC_BASE_COL, "pop_density_mean", "city_SEE", "city_CIE",
        GDP_BASE_COL, "GDP_growth_pct", RESPOP_BASE_COL, "Ppo_NetIn_rate",
        "ResPop_growth_rate", FISCAL_BASE_COL, "city_level_4",
    ]
    df_acc = df[acc_cols].dropna().copy()
    if len(df_acc) < 20:
        raise ValueError(f"Too few complete cases for accessibility regression: N={len(df_acc)}")

    f1 = (
        f"acc_delta ~ {ACC_BASE_COL} + pop_density_mean + city_SEE + city_CIE + "
        f"{GDP_BASE_COL} + GDP_growth_pct + {RESPOP_BASE_COL} + Ppo_NetIn_rate + "
        f"ResPop_growth_rate + {FISCAL_BASE_COL}"
    )
    f2 = f1 + f" + {city_fe}"
    f3 = f"acc_delta ~ {ACC_BASE_COL} + {controls} + city_SEE*{city_fe} + city_CIE*{city_fe}"
    m1 = smf.ols(f1, data=df_acc).fit(cov_type="HC1")
    m2 = smf.ols(f2, data=df_acc).fit(cov_type="HC1")
    m3 = smf.ols(f3, data=df_acc).fit(cov_type="HC1")

    for stage, model in enumerate([m1, m2, m3], start=1):
        coef_rows += _collect_model_coefficients(model, outcome="accessibility", model_family="SEE_CIE", stage=stage)
    effect_rows += _collect_city_effects(m3, outcome="accessibility", expansion="SEE", var_prefix="city_SEE")
    effect_rows += _collect_city_effects(m3, outcome="accessibility", expansion="CIE", var_prefix="city_CIE")
    _write_model_table(
        [m1, m2, m3],
        ["OLS", "OLS + city-size FE", "OLS + expansion x city size"],
        SEE_CIE_REGRESSION_ROOT / "accessibility_models.txt",
    )

    specs = {
        "gini": ("gini_delta", GINI_BASE_COL),
        "theil": ("theil_delta", THEIL_BASE_COL),
        "atkinson_05": ("atkinson_05_delta", ATKINSON_BASE_COL),
    }
    for label, (y, y0) in specs.items():
        cols = [
            y, y0, "pop_density_mean", GDP_BASE_COL, "GDP_growth_pct", RESPOP_BASE_COL,
            "Ppo_NetIn_rate", "ResPop_growth_rate", FISCAL_BASE_COL, "city_SEE",
            "city_CIE", "city_TotalExpansion", "city_level_4",
        ]
        d = df[cols].dropna().copy()
        if len(d) < 20:
            raise ValueError(f"Too few complete cases for {label}: N={len(d)}")

        a1 = smf.ols(f"{y} ~ {y0} + {controls} + city_SEE + city_CIE", data=d).fit(cov_type="HC1")
        a2 = smf.ols(f"{y} ~ {y0} + {controls} + city_SEE + city_CIE + {city_fe}", data=d).fit(cov_type="HC1")
        a3 = smf.ols(f"{y} ~ {y0} + {controls} + city_SEE*{city_fe} + city_CIE*{city_fe}", data=d).fit(cov_type="HC1")
        t1 = smf.ols(f"{y} ~ {y0} + {controls} + city_TotalExpansion", data=d).fit(cov_type="HC1")
        t2 = smf.ols(f"{y} ~ {y0} + {controls} + city_TotalExpansion + {city_fe}", data=d).fit(cov_type="HC1")
        t3 = smf.ols(f"{y} ~ {y0} + {controls} + city_TotalExpansion*{city_fe}", data=d).fit(cov_type="HC1")

        for stage, model in enumerate([a1, a2, a3], start=1):
            coef_rows += _collect_model_coefficients(model, outcome=label, model_family="SEE_CIE", stage=stage)
        for stage, model in enumerate([t1, t2, t3], start=1):
            coef_rows += _collect_model_coefficients(model, outcome=label, model_family="TotalExpansion", stage=stage)

        effect_rows += _collect_city_effects(a3, outcome=label, expansion="SEE", var_prefix="city_SEE")
        effect_rows += _collect_city_effects(a3, outcome=label, expansion="CIE", var_prefix="city_CIE")
        effect_rows += _collect_city_effects(t3, outcome=label, expansion="TotalExpansion", var_prefix="city_TotalExpansion")

        _write_model_table(
            [a1, a2, a3],
            ["OLS", "OLS + city-size FE", "OLS + expansion x city size"],
            SEE_CIE_REGRESSION_ROOT / f"inequality_models_{label}.txt",
        )
        _write_model_table(
            [t1, t2, t3],
            ["OLS", "OLS + city-size FE", "OLS + total expansion x city size"],
            SEE_CIE_REGRESSION_ROOT / f"total_expansion_models_{label}.txt",
        )

    pd.DataFrame(coef_rows).to_csv(
        SEE_CIE_REGRESSION_ROOT / "model_coefficients.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(effect_rows).to_csv(
        SEE_CIE_REGRESSION_ROOT / "city_size_effects.csv", index=False, encoding="utf-8-sig"
    )

    names = list(m3.params.index)
    wald_rows = []
    for city in CITY_ORDER_4:
        r = np.zeros((1, len(names)))
        def add(term, weight):
            if term in names:
                r[0, names.index(term)] += weight
        add("city_SEE", 1)
        add("city_CIE", -1)
        if city != "Medium/Small City":
            add(f'city_SEE:C(city_level_4, Treatment(reference="Medium/Small City"))[T.{city}]', 1)
            add(f'city_CIE:C(city_level_4, Treatment(reference="Medium/Small City"))[T.{city}]', -1)
        test = m3.wald_test(r, scalar=True)
        wald_rows.append({"city_level": city, "wald_stat": float(test.statistic), "p": float(test.pvalue)})
    pd.DataFrame(wald_rows).to_csv(
        SEE_CIE_REGRESSION_ROOT / "see_vs_cie_wald_accessibility.csv", index=False, encoding="utf-8-sig"
    )


if __name__ == "__main__":
    main()
